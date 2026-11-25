package main

// direct.go — talks to the ML AI Code gateway directly (real token streaming).
//
// The gateway (POST /api/model/v1/chat/completions) accepts a per-run bearer
// token ("mlrun_…") that the CLI mints for a live run. We can't mint it, so the
// bridge keeps one run open with a long `mlai --bg` task and reuses that run's
// token, refreshing whenever the gateway says the run has ended.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	gatewayURL   = "https://code.ml.ai/api/model/v1/chat/completions"
	keeperPrompt = "Run the shell command `sleep 5400` with the Bash tool and wait for it to finish, then reply with the single word: done."
)

// tokenTTL is shorter than the keeper's 90 minute sleep; 401/409 still force a refresh.
const tokenTTL = 80 * time.Minute

// keeperCooldown is the minimum time between two keeper spawns, success or not. It
// doubles with every failure in a row (up to 32x).
var keeperCooldown = 60 * time.Second

// closeWait bounds how long Close waits for a keeper run that is being opened.
var closeWait = 15 * time.Second

// gatewayClient never follows redirects: the run token must not leave the gateway.
var gatewayClient = &http.Client{
	CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
}

// runKeeper holds the run token. session is the id of the session it came from.
// refresh is non-nil while a refresh runs and is closed when it ends; lastErr is
// that refresh's result.
type runKeeper struct {
	mu        sync.Mutex
	token     string
	at        time.Time
	session   string
	dir       string
	mlai      string
	lastSpawn time.Time
	lastErr   error
	fails     int
	refresh   chan struct{}
	closed    bool
}

func newRunKeeper(mlaiBin, workdir string) *runKeeper {
	dir := filepath.Join(workdir, ".keeper")
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	os.MkdirAll(dir, 0o700)
	return &runKeeper{dir: dir, mlai: mlaiBin}
}

// sessionsDir is where the CLI writes per-run state (and the run token).
func sessionsDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".mlai", "sessions")
	}
	return ""
}

func canonDir(p string) string {
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	if rp, err := filepath.EvalSymlinks(p); err == nil {
		p = rp
	}
	return filepath.Clean(p)
}

func sameDir(a, b string) bool {
	a, b = canonDir(a), canonDir(b)
	if runtime.GOOS == "windows" {
		return strings.EqualFold(a, b)
	}
	return a == b
}

// gitRoot is the enclosing git work tree of dir, or dir itself: the CLI records
// that as a session's root.
func gitRoot(dir string) string {
	for d := canonDir(dir); ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, ".git")); err == nil {
			return d
		}
		if d == filepath.Dir(d) {
			return dir
		}
	}
}

// sessionRunsIn reports whether the session's recorded root is dir or its git root. A missing or
// unreadable session.json counts as no.
func sessionRunsIn(sessionDir, dir string) bool {
	raw, err := os.ReadFile(filepath.Join(sessionDir, "session.json"))
	if err != nil {
		return false
	}
	var doc struct {
		Root string `json:"root"`
	}
	if json.Unmarshal(raw, &doc) != nil || doc.Root == "" {
		return false
	}
	return sameDir(doc.Root, dir) || sameDir(doc.Root, gitRoot(dir))
}

// tokenFromNewestSession returns the newest mcp.json token (and its session id)
// modified after `since` from a session that runs in dir, ignoring the sessions in seen.
func tokenFromNewestSession(since time.Time, dir string, seen map[string]bool) (string, string, bool) {
	root := sessionsDir()
	if root == "" {
		return "", "", false
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", "", false
	}
	type cand struct {
		path string
		id   string
		mod  time.Time
	}
	var cands []cand
	for _, e := range entries {
		if seen[e.Name()] {
			continue
		}
		p := filepath.Join(root, e.Name(), "engine", "mcp.json")
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		if st.ModTime().After(since) && sessionRunsIn(filepath.Join(root, e.Name()), dir) {
			cands = append(cands, cand{p, e.Name(), st.ModTime()})
		}
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].mod.After(cands[j].mod) })
	for _, c := range cands {
		raw, err := os.ReadFile(c.path)
		if err != nil {
			continue
		}
		var doc struct {
			MCPServers map[string]struct {
				Env map[string]string `json:"env"`
			} `json:"mcpServers"`
		}
		if json.Unmarshal(raw, &doc) != nil {
			continue
		}
		for _, srv := range doc.MCPServers {
			if tok := srv.Env["MODEL_API_KEY"]; strings.HasPrefix(tok, "mlrun_") {
				return tok, c.id, true
			}
		}
	}
	return "", "", false
}

// Token returns a run token, opening a new run when the current one is stale.
func (k *runKeeper) Token(ctx context.Context) (string, error) {
	k.mu.Lock()
	tok, fresh := k.token, time.Since(k.at) < tokenTTL
	k.mu.Unlock()
	if tok != "" && fresh {
		return tok, nil
	}
	return k.Refresh(ctx, tok)
}

// Refresh replaces the run behind `stale`. If the keeper already holds a different
// token, that one is returned. At most one refresh runs at a time and it is not
// tied to any caller's context: callers wait for it or for their own ctx.
func (k *runKeeper) Refresh(ctx context.Context, stale string) (string, error) {
	k.mu.Lock()
	if k.token != "" && k.token != stale {
		tok := k.token
		k.mu.Unlock()
		return tok, nil
	}
	if k.closed {
		k.mu.Unlock()
		return "", errors.New("keeper closed")
	}
	if k.refresh == nil {
		if wait := keeperCooldown<<min(k.fails, 5) - time.Since(k.lastSpawn); wait > 0 {
			k.mu.Unlock()
			return "", fmt.Errorf("keeper cooling down for %s", wait.Round(time.Second))
		}
		k.refresh = make(chan struct{})
		k.lastSpawn = time.Now()
		go k.runRefresh(k.refresh)
	}
	done := k.refresh
	k.mu.Unlock()

	select {
	case <-done:
	case <-ctx.Done():
		return "", ctx.Err()
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.lastErr != nil {
		return "", k.lastErr
	}
	if k.token == "" {
		return "", errors.New("keeper closed")
	}
	return k.token, nil
}

func (k *runKeeper) runRefresh(done chan struct{}) {
	k.mu.Lock()
	prev := k.session
	expired := time.Since(k.at) >= tokenTTL
	k.session, k.token = "", ""
	k.mu.Unlock()
	if !expired { // the gateway rejected it; an expired run may still serve streams and ends by itself
		k.stop(prev)
	}

	tok, id, err := k.spawn()

	k.mu.Lock()
	orphan := ""
	switch {
	case err != nil:
		k.fails++
	case k.closed:
		orphan, err = id, errors.New("keeper closed")
	default:
		k.token, k.session, k.at = tok, id, time.Now()
		k.fails = 0
	}
	k.lastErr = err
	k.refresh = nil
	k.mu.Unlock()
	k.stop(orphan)
	close(done)
}

// spawn starts a keeper run and waits for its token to appear.
func (k *runKeeper) spawn() (string, string, error) {
	start := time.Now().Add(-2 * time.Second)
	seen := map[string]bool{} // only a session created from here on can be ours
	if entries, err := os.ReadDir(sessionsDir()); err == nil {
		for _, e := range entries {
			seen[e.Name()] = true
		}
	}
	cmd := exec.Command(k.mlai, "--bg", "-y", keeperPrompt)
	cmd.Dir = k.dir
	cmd.Env = childEnv()
	out := &cappedBuffer{max: 16 << 10}
	cmd.Stdout = out
	cmd.Stderr = out
	cmd.WaitDelay = 5 * time.Second
	if err := cmd.Start(); err != nil {
		return "", "", fmt.Errorf("start keeper: %w", err)
	}
	go cmd.Wait() // reap; the run lives on the server

	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		if tok, id, ok := tokenFromNewestSession(start, k.dir, seen); ok {
			logger.Printf("run token refreshed (keeper pid %d, len %d)", cmd.Process.Pid, len(tok))
			return tok, id, nil
		}
		time.Sleep(750 * time.Millisecond)
	}
	cmd.Process.Kill()
	return "", "", fmt.Errorf("no run token appeared within 90s: %s", strings.TrimSpace(out.String()))
}

// stop asks the CLI to end a keeper run. Best effort: errors are ignored.
func (k *runKeeper) stop(id string) {
	if id == "" || strings.HasPrefix(id, "-") {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, k.mlai, "agents", "stop", id)
	cmd.Dir = k.dir
	cmd.Env = childEnv()
	cmd.WaitDelay = 5 * time.Second
	cmd.Run()
}

// Close stops the current keeper run (server shutdown).
func (k *runKeeper) Close() {
	k.mu.Lock()
	k.closed = true
	id := k.session
	k.session, k.token = "", ""
	refresh := k.refresh
	k.mu.Unlock()
	k.stop(id)
	if refresh != nil { // a run is being opened: give it time to appear so it gets stopped
		select {
		case <-refresh:
		case <-time.After(closeWait):
		}
	}
}

// directRequest posts an OpenAI-shaped body to the gateway with the run token.
// The body of a >=400 answer is only logged.
func (s *server) directRequest(ctx context.Context, body []byte, stream bool) (*http.Response, error) {
	tok, err := s.keeper.Token(ctx)
	if err != nil {
		return nil, err
	}
	do := func(t string) (*http.Response, error) {
		req, err := http.NewRequestWithContext(ctx, "POST", gatewayURL, bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+t)
		req.Header.Set("Content-Type", "application/json")
		if stream {
			req.Header.Set("Accept", "text/event-stream")
		}
		req.Header.Set("User-Agent", "mlai-bridge/"+version)
		return gatewayClient.Do(req)
	}
	resp, err := do(tok)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == 401 || resp.StatusCode == 409 {
		// run ended or revoked — open a new one and retry once
		resp.Body.Close()
		tok, err = s.keeper.Refresh(ctx, tok)
		if err != nil {
			return nil, err
		}
		resp, err = do(tok)
		if err != nil {
			return nil, err
		}
	}
	if resp.StatusCode >= 400 {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
		logger.Printf("gateway %d: %q", resp.StatusCode, strings.TrimSpace(string(raw)))
		return nil, fmt.Errorf("gateway returned HTTP %d", resp.StatusCode)
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		resp.Body.Close()
		return nil, fmt.Errorf("gateway returned HTTP %d", resp.StatusCode)
	}
	return resp, nil
}
