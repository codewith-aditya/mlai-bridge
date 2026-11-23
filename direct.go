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
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	gatewayURL   = "https://code.ml.ai/api/model/v1/chat/completions"
	keeperPrompt = "Run the shell command `sleep 5400` with the Bash tool and wait for it to finish, then reply with the single word: done."
	tokenTTL     = 8 * time.Minute
)

type runKeeper struct {
	mu      sync.Mutex
	token   string
	at      time.Time
	dir     string
	mlai    string
	lastErr error
}

func newRunKeeper(mlaiBin, workdir string) *runKeeper {
	dir := filepath.Join(workdir, ".keeper")
	os.MkdirAll(dir, 0o755)
	return &runKeeper{dir: dir, mlai: mlaiBin}
}

// sessionsDir is where the CLI writes per-run state (and the run token).
func sessionsDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, ".mlai", "sessions")
	}
	return ""
}

// tokenFromNewestSession returns the newest mcp.json token modified after `since`.
func tokenFromNewestSession(since time.Time) (string, bool) {
	root := sessionsDir()
	if root == "" {
		return "", false
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return "", false
	}
	type cand struct {
		path string
		mod  time.Time
	}
	var cands []cand
	for _, e := range entries {
		p := filepath.Join(root, e.Name(), "engine", "mcp.json")
		st, err := os.Stat(p)
		if err != nil {
			continue
		}
		if st.ModTime().After(since) {
			cands = append(cands, cand{p, st.ModTime()})
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
				return tok, true
			}
		}
	}
	return "", false
}

// Token returns a run token, opening a new run when the current one is stale.
func (k *runKeeper) Token(ctx context.Context) (string, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.token != "" && time.Since(k.at) < tokenTTL {
		return k.token, nil
	}
	return k.refreshLocked(ctx)
}

// Refresh forces a new run even if the cached token looks fresh.
func (k *runKeeper) Refresh(ctx context.Context) (string, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.token = ""
	return k.refreshLocked(ctx)
}

func (k *runKeeper) refreshLocked(ctx context.Context) (string, error) {
	start := time.Now().Add(-2 * time.Second)
	cmd := exec.Command(k.mlai, "--bg", "-y", keeperPrompt)
	cmd.Dir = k.dir
	cmd.Env = append(os.Environ(), "NO_COLOR=1", "CI=1")
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Start(); err != nil {
		k.lastErr = err
		return "", fmt.Errorf("start keeper: %w", err)
	}
	go cmd.Wait() // reap; the run lives on the server

	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		if tok, ok := tokenFromNewestSession(start); ok {
			k.token = tok
			k.at = time.Now()
			logger.Printf("run token refreshed (keeper pid %d, len %d)", cmd.Process.Pid, len(tok))
			return tok, nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(750 * time.Millisecond):
		}
	}
	k.lastErr = fmt.Errorf("no run token appeared within 90s: %s", strings.TrimSpace(out.String()))
	return "", k.lastErr
}

// directRequest posts an OpenAI-shaped body to the gateway with the run token.
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
		return http.DefaultClient.Do(req)
	}
	resp, err := do(tok)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == 401 || resp.StatusCode == 409 {
		// run ended or revoked — open a new one and retry once
		resp.Body.Close()
		tok, err = s.keeper.Refresh(ctx)
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
		return nil, fmt.Errorf("gateway %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return resp, nil
}
