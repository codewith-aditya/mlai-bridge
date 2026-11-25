package main

// security_test.go: security behaviour of the bridge (auth gate, host and
// content-type checks, request limits, logging, child-process hygiene, the
// run-token keeper and the direct gateway path). The mlai CLI is replaced by
// this test binary re-executed with MLAI_STUB=1 (see TestMain).

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

const testKey = "test-bridge-key-0000"

const stubSecret = "dummy-value-the-child-must-not-see"

const gatewayOK = `{"model":"gw-model","choices":[{"message":{"content":"direct says hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":2}}`

const chatBody = `{"model":"m","messages":[{"role":"user","content":"hello"}]}`

const anthropicBody = `{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hello"}]}`

// ---------- stub CLI ----------

// stubCall is one line of the stub's log.
type stubCall struct {
	Args   []string `json:"args"`
	HasKey bool     `json:"has_key"`
}

func TestMain(m *testing.M) {
	if os.Getenv("MLAI_STUB") == "1" {
		os.Exit(runStub(os.Args[1:]))
	}
	os.Exit(m.Run())
}

func runStub(args []string) int {
	_, hasKey := os.LookupEnv("MLAI_API_KEY")
	if path := os.Getenv("STUB_LOG"); path != "" {
		rec, _ := json.Marshal(stubCall{Args: args, HasKey: hasKey})
		if f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600); err == nil {
			f.Write(append(rec, '\n'))
			f.Close()
		}
	}
	if len(args) == 0 {
		return 0
	}
	switch {
	case args[0] == "whoami":
		fmt.Println("Signed in as stub@example.test")
	case args[0] == "--bg":
		return stubBackground()
	case slices.Contains(args, "-p"):
		return stubPrompt()
	}
	return 0
}

func stubResult(text string) {
	b, _ := json.Marshal(map[string]string{"type": "result", "text": text})
	fmt.Println(string(b))
}

// stubBackground mimics `mlai --bg`: it records a session whose token the
// keeper can pick up.
func stubBackground() int {
	if os.Getenv("STUB_BG") == "fail" {
		fmt.Fprintln(os.Stderr, "stub: cannot start a background run")
		return 1
	}
	cwd, err := os.Getwd()
	if err != nil {
		return 1
	}
	id := fmt.Sprintf("stub-%d-%d", os.Getpid(), time.Now().UnixNano())
	dir := filepath.Join(os.Getenv("STUB_HOME"), ".mlai", "sessions", id)
	if err := os.MkdirAll(filepath.Join(dir, "engine"), 0o700); err != nil {
		return 1
	}
	session, _ := json.Marshal(map[string]string{"root": cwd})
	servers := map[string]any{"x": map[string]any{"env": map[string]string{"MODEL_API_KEY": "mlrun_" + id}}}
	mcp, _ := json.Marshal(map[string]any{"mcpServers": servers})
	if os.WriteFile(filepath.Join(dir, "session.json"), session, 0o600) != nil {
		return 1
	}
	if os.WriteFile(filepath.Join(dir, "engine", "mcp.json"), mcp, 0o600) != nil {
		return 1
	}
	return 0
}

// stubPrompt mimics `mlai -p`: the prompt arrives on stdin.
func stubPrompt() int {
	in, _ := io.ReadAll(os.Stdin)
	switch os.Getenv("STUB_MODE") {
	case "bigout":
		w := bufio.NewWriterSize(os.Stdout, 1<<16)
		w.WriteString(`{"type":"answer","content":"`)
		w.WriteString(strings.Repeat("x", 8<<20))
		w.WriteString("\"}\n")
		w.Flush()
	case "bigerr":
		os.Stderr.WriteString(strings.Repeat("e", 1<<20) + "\n")
		stubResult("stub:" + string(in))
	case "hang":
		time.Sleep(30 * time.Second)
	case "text":
		stubResult(os.Getenv("STUB_TEXT"))
	default:
		stubResult("stub:" + string(in))
	}
	return 0
}

// ---------- test helpers ----------

// saveGlobals restores every global the tests touch and silences the logger.
func saveGlobals(t *testing.T) {
	t.Helper()
	oldCfg, oldLogger, oldClient := cfg, logger, gatewayClient
	oldCooldown, oldBodyTimeout := keeperCooldown, bodyReadTimeout
	t.Cleanup(func() {
		cfg, logger, gatewayClient = oldCfg, oldLogger, oldClient
		keeperCooldown, bodyReadTimeout = oldCooldown, oldBodyTimeout
	})
	logger = log.New(io.Discard, "", 0)
}

type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func setTransport(t *testing.T, rt http.RoundTripper) {
	t.Helper()
	old := gatewayClient.Transport
	gatewayClient.Transport = rt
	t.Cleanup(func() { gatewayClient.Transport = old })
}

func fakeResp(r *http.Request, status int, body string) *http.Response {
	h := http.Header{"Content-Type": {"application/json"}}
	return &http.Response{StatusCode: status, Header: h, Body: io.NopCloser(strings.NewReader(body)), Request: r}
}

// stubEnv describes the sandbox every CLI-backed test runs in.
type stubEnv struct {
	home string
	log  string
}

// setupStub points cfg at the re-executed test binary, moves the home directory
// into a temp dir and blocks the network.
func setupStub(t *testing.T) *stubEnv {
	t.Helper()
	saveGlobals(t)
	env := &stubEnv{home: t.TempDir()}
	env.log = filepath.Join(t.TempDir(), "stub.log")
	t.Setenv("MLAI_STUB", "1")
	t.Setenv("HOME", env.home)
	t.Setenv("USERPROFILE", env.home)
	t.Setenv("STUB_HOME", env.home)
	t.Setenv("STUB_LOG", env.log)
	t.Setenv("STUB_MODE", "")
	t.Setenv("STUB_BG", "")
	t.Setenv("STUB_TEXT", "")
	t.Setenv("MLAI_API_KEY", stubSecret)

	cfg = config{}
	cfg.MLAI = os.Args[0]
	cfg.WorkDir = filepath.Join(t.TempDir(), "work")
	cfg.DefaultModel = "test-model"
	cfg.Concurrency = 2
	cfg.DirectConcurrency = 2
	cfg.QueueWait = 2 * time.Second
	cfg.Timeout = 30 * time.Second
	cfg.CacheSize = 16
	cfg.Bind = "127.0.0.1"
	if err := os.MkdirAll(cfg.WorkDir, 0o700); err != nil {
		t.Fatal(err)
	}
	setTransport(t, rtFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("network is disabled in tests")
	}))
	return env
}

// calls returns every recorded stub invocation and fails the test if the
// bridge's own API key reached a child.
func (e *stubEnv) calls(t *testing.T) []stubCall {
	t.Helper()
	raw, err := os.ReadFile(e.log)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var out []stubCall
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var c stubCall
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("bad stub log line %q: %v", line, err)
		}
		if c.HasKey {
			t.Errorf("MLAI_API_KEY leaked to a child process: %v", c.Args)
		}
		out = append(out, c)
	}
	return out
}

// count returns how many stub invocations had `first` as their first argument.
func (e *stubEnv) count(t *testing.T, first string) int {
	t.Helper()
	n := 0
	for _, c := range e.calls(t) {
		if len(c.Args) > 0 && c.Args[0] == first {
			n++
		}
	}
	return n
}

func waitFor(t *testing.T, limit time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(limit)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s", limit, what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func keeperState(k *runKeeper) (token string, fails int, busy bool) {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.token, k.fails, k.refresh != nil
}

// waitKeeperIdle makes sure no refresh goroutine outlives the test (and its env).
func waitKeeperIdle(t *testing.T, k *runKeeper) {
	t.Helper()
	waitFor(t, 30*time.Second, "the keeper refresh to finish", func() bool {
		_, _, busy := keeperState(k)
		return !busy
	})
}

func seedToken(s *server, tok string) {
	s.keeper.mu.Lock()
	defer s.keeper.mu.Unlock()
	s.keeper.token = tok
	s.keeper.at = time.Now()
}

func newTestServer(t *testing.T) *server {
	t.Helper()
	s := newServer()
	t.Cleanup(func() { waitKeeperIdle(t, s.keeper) })
	return s
}

func testMux(s *server) *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages/count_tokens", s.handleCountTokens)
	return mux
}

// testChain is the handler chain main() builds.
func testChain(s *server, key string) http.Handler {
	return withBodyDeadline(withMiddleware(apiKeyGate(testMux(s), key)))
}

// serve runs one request through h. An empty host means a loopback Host header.
func serve(h http.Handler, method, target, host string, hdr map[string]string, body io.Reader) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, target, body)
	req.Host = "127.0.0.1:8197"
	if host != "" {
		req.Host = host
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func parseBody(t *testing.T, rec *httptest.ResponseRecorder) any {
	t.Helper()
	var v any
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatalf("body is not JSON (%v): %q", err, rec.Body.String())
	}
	return v
}

// dig walks decoded JSON by object key (string) and array index (int).
func dig(v any, path ...any) any {
	for _, p := range path {
		switch k := p.(type) {
		case string:
			m, ok := v.(map[string]any)
			if !ok {
				return nil
			}
			v = m[k]
		case int:
			a, ok := v.([]any)
			if !ok || k >= len(a) {
				return nil
			}
			v = a[k]
		}
	}
	return v
}

func assertAnthropicError(t *testing.T, rec *httptest.ResponseRecorder, status int, errType string) {
	t.Helper()
	doc := parseBody(t, rec)
	if rec.Code != status {
		t.Errorf("status = %d, want %d (%s)", rec.Code, status, rec.Body.String())
	}
	if dig(doc, "type") != "error" || dig(doc, "error", "type") != errType || dig(doc, "error", "code") != nil {
		t.Errorf("not an Anthropic %q error: %s", errType, rec.Body.String())
	}
}

func assertOpenAIError(t *testing.T, rec *httptest.ResponseRecorder, status int, code string) {
	t.Helper()
	doc := parseBody(t, rec)
	if rec.Code != status {
		t.Errorf("status = %d, want %d (%s)", rec.Code, status, rec.Body.String())
	}
	if dig(doc, "error", "code") != code || dig(doc, "type") != nil {
		t.Errorf("not an OpenAI %q error: %s", code, rec.Body.String())
	}
}

func assertNoCORS(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	for name := range rec.Header() {
		if strings.HasPrefix(strings.ToLower(name), "access-control-") {
			t.Errorf("unexpected CORS header %s on a keyless bridge", name)
		}
	}
}

// ---------- 1. loopback detection and config validation ----------

func TestIsLoopbackHost(t *testing.T) {
	cases := []struct {
		host string
		want bool
	}{
		{"127.0.0.1", true},
		{"127.0.0.2", true},
		{"localhost", true},
		{"LOCALHOST", true},
		{"localhost.", true},
		{"::1", true},
		{"", false},
		{"0.0.0.0", false},
		{"::", false},
		{"192.168.1.5", false},
		{"10.0.0.1", false},
		{"evil.example", false},
		{"localhost.evil.example", false},
		{"127.0.0.1.evil.example", false},
	}
	for _, c := range cases {
		if got := isLoopbackHost(c.host); got != c.want {
			t.Errorf("isLoopbackHost(%q) = %v, want %v", c.host, got, c.want)
		}
	}
}

func TestValidateConfig(t *testing.T) {
	for _, bind := range []string{"127.0.0.1", "localhost", "::1"} {
		if err := validateConfig(config{Bind: bind}); err != nil {
			t.Errorf("bind %q without key: %v", bind, err)
		}
		if err := validateConfig(config{Bind: bind, APIKey: testKey}); err != nil {
			t.Errorf("bind %q with key: %v", bind, err)
		}
	}
	for _, bind := range []string{"", "0.0.0.0", "::", "192.168.1.5"} {
		err := validateConfig(config{Bind: bind})
		if err == nil {
			t.Errorf("bind %q without key must be refused", bind)
		} else if !strings.Contains(err.Error(), "MLAI_API_KEY") {
			t.Errorf("bind %q: error should name the fix: %v", bind, err)
		}
		if err := validateConfig(config{Bind: bind, APIKey: testKey}); err != nil {
			t.Errorf("bind %q with key: %v", bind, err)
		}
	}
}

// ---------- 2. full chain with an API key ----------

func TestAPIKeyGate(t *testing.T) {
	setupStub(t)
	h := testChain(newTestServer(t), testKey)
	bearer := func(v string) map[string]string { return map[string]string{"Authorization": v} }
	cases := []struct {
		name   string
		method string
		target string
		hdr    map[string]string
		want   int
	}{
		{"missing key", "GET", "/v1/models", nil, 401},
		{"wrong bearer", "GET", "/v1/models", bearer("Bearer wrong"), 401},
		{"empty bearer", "GET", "/v1/models", bearer("Bearer "), 401},
		{"key prefix only", "GET", "/v1/models", bearer("Bearer " + testKey[:8]), 401},
		{"key plus suffix", "GET", "/v1/models", bearer("Bearer " + testKey + "x"), 401},
		{"basic scheme", "GET", "/v1/models", bearer("Basic " + testKey), 401},
		{"wrong x-api-key", "GET", "/v1/models", map[string]string{"x-api-key": "wrong"}, 401},
		{"wrong api-key", "GET", "/v1/models", map[string]string{"api-key": "wrong"}, 401},
		{"bearer", "GET", "/v1/models", bearer("Bearer " + testKey), 200},
		{"x-api-key", "GET", "/v1/models", map[string]string{"x-api-key": testKey}, 200},
		{"api-key", "GET", "/v1/models", map[string]string{"api-key": testKey}, 200},
		{"key in query", "GET", "/v1/models?key=" + testKey, nil, 401},
		{"api_key in query", "GET", "/v1/models?api_key=" + testKey, nil, 401},
		{"healthz is open", "GET", "/healthz", nil, 200},
		{"health is open", "GET", "/health", nil, 200},
		{"metrics is open", "GET", "/metrics", nil, 200},
		{"models stays closed", "GET", "/v1/models", nil, 401},
		{"preflight", "OPTIONS", "/v1/chat/completions", nil, 204},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := serve(h, c.method, c.target, "", c.hdr, nil)
			if rec.Code != c.want {
				t.Fatalf("status = %d, want %d (%s)", rec.Code, c.want, rec.Body.String())
			}
			if c.want == 401 {
				if rec.Header().Get("WWW-Authenticate") == "" {
					t.Error("401 without WWW-Authenticate")
				}
				if strings.Contains(rec.Body.String(), "pixis") {
					t.Error("401 body leaks the model list")
				}
			}
		})
	}
}

func TestAPIKeyGatePathTricks(t *testing.T) {
	setupStub(t)
	h := testChain(newTestServer(t), testKey)
	targets := []string{
		"/healthz/../v1/models",
		"//v1/models",
		"/healthz%2f..%2fv1/models",
		"/metrics/../v1/models",
		"/v1/models/",
		"/v1//models",
	}
	auth := map[string]string{"Authorization": "Bearer " + testKey}
	for _, target := range targets {
		rec := serve(h, "GET", target, "", nil, nil)
		if rec.Code == 200 || strings.Contains(rec.Body.String(), "pixis") {
			t.Errorf("GET %s without key = %d: %s", target, rec.Code, rec.Body.String())
		}
		// Even with a valid key the odd path must not serve the model list.
		rec = serve(h, "GET", target, "", auth, nil)
		if rec.Code == 200 || strings.Contains(rec.Body.String(), "pixis") {
			t.Errorf("GET %s with key = %d: %s", target, rec.Code, rec.Body.String())
		}
	}
}

func TestErrorShapes(t *testing.T) {
	setupStub(t)
	s := newTestServer(t)
	jsonType := map[string]string{"Content-Type": "application/json"}
	authJSON := map[string]string{"Content-Type": "application/json", "Authorization": "Bearer " + testKey}
	textType := map[string]string{"Content-Type": "text/plain"}
	const messages, chat = "/v1/messages", "/v1/chat/completions"
	cases := []struct {
		name      string
		key       string
		host      string
		hdr       map[string]string
		path      string
		body      string
		status    int
		anthropic bool
		code      string
	}{
		{"anthropic missing key", testKey, "", jsonType, messages, anthropicBody, 401, true, "authentication_error"},
		{"openai missing key", testKey, "", jsonType, chat, chatBody, 401, false, "invalid_api_key"},
		{"anthropic bad json", testKey, "", authJSON, messages, "{", 400, true, "invalid_request_error"},
		{"openai bad json", testKey, "", authJSON, chat, "{", 400, false, "invalid_request"},
		{"anthropic foreign host", "", "evil.example", jsonType, messages, anthropicBody, 403, true, "permission_error"},
		{"openai foreign host", "", "evil.example", jsonType, chat, chatBody, 403, false, "forbidden_host"},
		{"anthropic wrong media type", "", "", textType, messages, anthropicBody, 415, true, "invalid_request_error"},
		{"openai wrong media type", "", "", textType, chat, chatBody, 415, false, "unsupported_media_type"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			h := testChain(s, c.key)
			rec := serve(h, "POST", c.path, c.host, c.hdr, strings.NewReader(c.body))
			if c.anthropic {
				assertAnthropicError(t, rec, c.status, c.code)
			} else {
				assertOpenAIError(t, rec, c.status, c.code)
			}
		})
	}
}

// ---------- 3. loopback-only mode (no key) ----------

func TestNoKeyRejectsForeignHost(t *testing.T) {
	env := setupStub(t)
	h := testChain(newTestServer(t), "")
	hosts := []string{
		"evil.example",
		"evil.example:8197",
		"localhost.evil.example",
		"127.0.0.1.evil.example:8197",
		"10.0.0.5:8197",
		"0.0.0.0:8197",
		"[::]:8197",
	}
	for _, host := range hosts {
		for _, target := range []string{"/v1/models", "/healthz", "/metrics"} {
			rec := serve(h, "GET", target, host, nil, nil)
			if rec.Code != 403 {
				t.Errorf("GET %s with Host %q = %d, want 403", target, host, rec.Code)
			}
			assertNoCORS(t, rec)
		}
		rec := serve(h, "POST", "/v1/chat/completions", host, map[string]string{"Content-Type": "application/json"}, strings.NewReader(chatBody))
		if rec.Code != 403 {
			t.Errorf("POST with Host %q = %d, want 403", host, rec.Code)
		}
	}
	if n := env.count(t, "whoami"); n != 0 {
		t.Errorf("a rejected /healthz still ran the CLI %d times", n)
	}
}

func TestNoKeyAllowsLoopbackHost(t *testing.T) {
	setupStub(t)
	h := testChain(newTestServer(t), "")
	for _, host := range []string{"localhost:8197", "127.0.0.1:8197", "[::1]:8197", "localhost", "127.0.0.1", "[::1]", "LOCALHOST:8197"} {
		rec := serve(h, "GET", "/v1/models", host, nil, nil)
		if rec.Code != 200 {
			t.Errorf("Host %q = %d, want 200", host, rec.Code)
		}
		assertNoCORS(t, rec)
	}
}

func TestNoKeyContentTypeCheck(t *testing.T) {
	setupStub(t)
	h := testChain(newTestServer(t), "")
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	cases := []struct {
		name string
		ct   string
		want int
	}{
		{"text/plain", "text/plain", 415},
		{"no content type", "", 415},
		{"form", "application/x-www-form-urlencoded", 415},
		{"multipart", "multipart/form-data; boundary=x", 415},
		{"json-like subtype", "application/json-patch+json", 415},
		{"json-like prefix", "application/jsonx", 415},
		{"json", "application/json", 200},
		{"json with charset", "application/json; charset=utf-8", 200},
		{"json upper case", "Application/JSON", 200},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var hdr map[string]string
			if c.ct != "" {
				hdr = map[string]string{"Content-Type": c.ct}
			}
			rec := serve(h, "POST", "/v1/messages/count_tokens", "", hdr, strings.NewReader(body))
			if rec.Code != c.want {
				t.Errorf("status = %d, want %d (%s)", rec.Code, c.want, rec.Body.String())
			}
			assertNoCORS(t, rec)
		})
	}
}

func TestNoKeySendsNoCORSHeaders(t *testing.T) {
	setupStub(t)
	h := testChain(newTestServer(t), "")
	origin := map[string]string{"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
	for _, method := range []string{"GET", "OPTIONS"} {
		for _, target := range []string{"/v1/models", "/v1/chat/completions", "/metrics"} {
			assertNoCORS(t, serve(h, method, target, "", origin, nil))
		}
	}
}

// ---------- 4. oversized bodies ----------

func TestOversizedBodiesAreRejected(t *testing.T) {
	setupStub(t)
	s := newTestServer(t)
	big := []byte(`{"messages":[{"role":"user","content":"` + strings.Repeat("a", 17<<20) + `"}]}`)
	small := `{"messages":[{"role":"user","content":"hi"}]}`
	for _, key := range []string{"", testKey} {
		h := testChain(s, key)
		hdr := map[string]string{"Content-Type": "application/json"}
		if key != "" {
			hdr["Authorization"] = "Bearer " + key
		}
		if rec := serve(h, "POST", "/v1/messages/count_tokens", "", hdr, strings.NewReader(small)); rec.Code != 200 {
			t.Fatalf("key %q: control request = %d: %s", key, rec.Code, rec.Body.String())
		}
		for _, method := range []string{"POST", "PUT", "GET"} {
			rec := serve(h, method, "/v1/messages/count_tokens", "", hdr, bytes.NewReader(big))
			if rec.Code == 200 || rec.Code < 400 {
				t.Errorf("key %q: %s with a 17 MiB body = %d, want an error", key, method, rec.Code)
			}
		}
	}
}

// ---------- 5. logging ----------

func TestLogLineCannotBeForged(t *testing.T) {
	setupStub(t)
	var buf bytes.Buffer
	logger = log.New(&buf, "", 0)
	h := testChain(newTestServer(t), "")

	rec := serve(h, "GET", "/v1/x%0Afake%20log%20line", "", nil, nil)
	if rec.Code != 404 {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	out := strings.TrimSuffix(buf.String(), "\n")
	if strings.Contains(out, "\n") {
		t.Fatalf("one request produced several log lines: %q", buf.String())
	}
	if !strings.Contains(out, `\n`) {
		t.Errorf("newline in the path should be escaped in the log: %q", out)
	}
}

func TestRequestIDHandling(t *testing.T) {
	setupStub(t)
	var buf bytes.Buffer
	logger = log.New(&buf, "", 0)
	h := testChain(newTestServer(t), "")
	cases := []struct {
		name string
		id   string
		keep bool
	}{
		{"valid", "abc-123_X.y", true},
		{"64 chars", strings.Repeat("a", 64), true},
		{"65 chars", strings.Repeat("b", 65), false},
		{"with spaces", "has some spaces", false},
		{"with semicolon", "a;b", false},
		{"with quote", `a"b`, false},
		{"non ascii", "caf\u00e9", false},
		{"empty", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			buf.Reset()
			hdr := map[string]string{"X-Request-Id": c.id}
			rec := serve(h, "GET", "/v1/models", "", hdr, nil)
			got := rec.Header().Get("X-Request-Id")
			if c.keep {
				if got != c.id {
					t.Errorf("X-Request-Id = %q, want it echoed", got)
				}
				return
			}
			if got == c.id || !strings.HasPrefix(got, "req_") || !requestIDPattern.MatchString(got) {
				t.Errorf("X-Request-Id = %q, want a generated id", got)
			}
			if c.id != "" && strings.Contains(buf.String(), c.id) {
				t.Errorf("rejected id reached the log: %q", buf.String())
			}
		})
	}
}

// ---------- 6. /healthz ----------

func TestHealthzIsMinimalAndRunsWhoamiOnce(t *testing.T) {
	env := setupStub(t)
	h := testChain(newTestServer(t), "")
	const callers = 20
	recs := make([]*httptest.ResponseRecorder, callers)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range recs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			recs[i] = serve(h, "GET", "/healthz", "", nil, nil)
		}()
	}
	close(start)
	wg.Wait()
	recs = append(recs, serve(h, "GET", "/health", "", nil, nil))

	allowed := []string{"status", "version", "uptime_s", "signed_in"}
	for _, rec := range recs {
		if rec.Code != 200 {
			t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
		}
		body := rec.Body.String()
		for _, leak := range []string{"whoami", "workdir", "stub@", cfg.WorkDir, env.home} {
			if strings.Contains(body, leak) {
				t.Errorf("/healthz body leaks %q: %s", leak, body)
			}
		}
		doc, _ := parseBody(t, rec).(map[string]any)
		for k := range doc {
			if !slices.Contains(allowed, k) {
				t.Errorf("unexpected /healthz field %q", k)
			}
		}
		if dig(doc, "signed_in") != true {
			t.Errorf("signed_in = %v, want true", dig(doc, "signed_in"))
		}
	}
	if n := env.count(t, "whoami"); n != 1 {
		t.Errorf("whoami ran %d times for %d requests, want 1", n, len(recs))
	}
}

// ---------- 7. runner ----------

func runBounded(t *testing.T, ctx context.Context, r *Runner, prompt string, limit time.Duration) RunResult {
	t.Helper()
	ch := make(chan RunResult, 1)
	go func() { ch <- r.Run(ctx, prompt, "test-model", "") }()
	select {
	case res := <-ch:
		return res
	case <-time.After(limit):
		t.Fatalf("Run did not return within %s", limit)
		return RunResult{}
	}
}

func TestRunnerPassesPromptOnStdinOnly(t *testing.T) {
	env := setupStub(t)
	r := &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: 30}
	prompts := []string{"--version", "-y", "--effort max", "--dangerously-skip-permissions"}
	for _, prompt := range prompts {
		res := runBounded(t, context.Background(), r, prompt, 20*time.Second)
		if res.Err != nil || res.Text != "stub:"+prompt {
			t.Errorf("prompt %q: text %q, err %v", prompt, res.Text, res.Err)
		}
	}
	calls := env.calls(t)
	if len(calls) != len(prompts) {
		t.Fatalf("stub ran %d times, want %d", len(calls), len(prompts))
	}
	for _, c := range calls {
		if c.Args[0] != "-p" {
			t.Errorf("unexpected argv %v", c.Args)
		}
		for _, arg := range c.Args {
			if slices.Contains(prompts, arg) {
				t.Errorf("prompt reached argv: %v", c.Args)
			}
		}
	}
}

func TestChildEnvDropsAPIKey(t *testing.T) {
	t.Setenv("MLAI_API_KEY", stubSecret)
	t.Setenv("mlai_api_key", stubSecret)
	env := childEnv()
	for _, kv := range env {
		name, _, _ := strings.Cut(kv, "=")
		if strings.EqualFold(name, "MLAI_API_KEY") || strings.Contains(kv, stubSecret) {
			t.Errorf("child env contains %q", name)
		}
	}
	if !slices.Contains(env, "NO_COLOR=1") {
		t.Error("child env lost NO_COLOR=1")
	}
}

func TestRunnerEmptyPromptStartsNothing(t *testing.T) {
	env := setupStub(t)
	r := &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: 30}
	for _, prompt := range []string{"", "   ", "\n\t "} {
		res := runBounded(t, context.Background(), r, prompt, 5*time.Second)
		if res.Err == nil || !strings.Contains(res.Err.Error(), "empty prompt") {
			t.Errorf("prompt %q: err = %v, want an empty prompt error", prompt, res.Err)
		}
	}
	if n := len(env.calls(t)); n != 0 {
		t.Errorf("an empty prompt started %d processes", n)
	}
}

func TestRunnerSurvivesHugeOutput(t *testing.T) {
	env := setupStub(t)
	r := &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: 30}
	t.Run("8 MiB stdout line", func(t *testing.T) {
		t.Setenv("STUB_MODE", "bigout")
		runBounded(t, context.Background(), r, "hello", 20*time.Second)
	})
	t.Run("1 MiB stderr line", func(t *testing.T) {
		t.Setenv("STUB_MODE", "bigerr")
		res := runBounded(t, context.Background(), r, "hello", 20*time.Second)
		if res.Err != nil || res.Text != "stub:hello" {
			t.Errorf("text %q, err %v", res.Text, res.Err)
		}
	})
	env.calls(t)
}

func TestRunnerKillsHungChild(t *testing.T) {
	env := setupStub(t)
	t.Setenv("STUB_MODE", "hang")
	r := &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: 30}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	start := time.Now()
	res := runBounded(t, ctx, r, "hello", 20*time.Second)
	if elapsed := time.Since(start); elapsed > 10*time.Second {
		t.Errorf("Run took %s after a 1s deadline", elapsed)
	}
	if res.Err == nil {
		t.Fatal("expected an error from a hung CLI")
	}
	if got := classify(res.Err.Error()).Status; got != 504 {
		t.Errorf("classify(%q) = %d, want 504", res.Err, got)
	}
	env.calls(t)
}

// ---------- 8. keeper ----------

func TestKeeperStartsOneRunForCancelledCallers(t *testing.T) {
	env := setupStub(t)
	keeperCooldown = time.Hour
	k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })

	const callers = 5
	errs := make(chan error, callers)
	var wg sync.WaitGroup
	for range callers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			_, err := k.Token(ctx)
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil && !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("Token returned %v, want a deadline error", err)
		}
	}

	// The refresh is not tied to any caller's context and finishes on its own.
	waitFor(t, 30*time.Second, "the keeper token", func() bool {
		tok, _, busy := keeperState(k)
		return !busy && tok != ""
	})
	if n := env.count(t, "--bg"); n != 1 {
		t.Errorf("--bg ran %d times for %d callers, want 1", n, callers)
	}
	tok, fails, _ := keeperState(k)
	if !strings.HasPrefix(tok, "mlrun_") || fails != 0 {
		t.Errorf("token %q, fails %d", tok, fails)
	}

	// A second refresh inside the cooldown is refused without a spawn.
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := k.Refresh(ctx, tok); err == nil || !strings.Contains(err.Error(), "cooling down") {
		t.Errorf("second refresh: err = %v, want a cooldown error", err)
	}
	if n := env.count(t, "--bg"); n != 1 {
		t.Errorf("--bg ran %d times after the cooldown refusal, want 1", n)
	}
}

func TestKeeperRefreshReturnsNewerToken(t *testing.T) {
	env := setupStub(t)
	k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })
	k.mu.Lock()
	k.token, k.at = "mlrun_new", time.Now()
	k.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	got, err := k.Refresh(ctx, "mlrun_old")
	if err != nil || got != "mlrun_new" {
		t.Errorf("Refresh(old) = %q, %v; want the newer token", got, err)
	}
	if got, err := k.Token(ctx); err != nil || got != "mlrun_new" {
		t.Errorf("Token = %q, %v; want the fresh token", got, err)
	}
	if n := len(env.calls(t)); n != 0 {
		t.Errorf("%d processes were spawned, want none", n)
	}
}

func TestKeeperClosedRefusesToSpawn(t *testing.T) {
	env := setupStub(t)
	k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })
	k.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := k.Token(ctx); err == nil {
		t.Error("a closed keeper returned a token")
	}
	if n := len(env.calls(t)); n != 0 {
		t.Errorf("a closed keeper spawned %d processes", n)
	}
}

func TestTokenFromNewestSessionIgnoresOtherRoots(t *testing.T) {
	env := setupStub(t)
	dir, other := t.TempDir(), t.TempDir()
	sessions := filepath.Join(env.home, ".mlai", "sessions")
	if got := sessionsDir(); got != sessions {
		t.Fatalf("sessionsDir() = %q, want %q", got, sessions)
	}
	add := func(id, sessionJSON, token string, age time.Duration) {
		t.Helper()
		base := filepath.Join(sessions, id)
		if err := os.MkdirAll(filepath.Join(base, "engine"), 0o700); err != nil {
			t.Fatal(err)
		}
		mcp := fmt.Sprintf(`{"mcpServers":{"x":{"env":{"MODEL_API_KEY":%q}}}}`, token)
		mcpPath := filepath.Join(base, "engine", "mcp.json")
		if err := os.WriteFile(filepath.Join(base, "session.json"), []byte(sessionJSON), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(mcpPath, []byte(mcp), 0o600); err != nil {
			t.Fatal(err)
		}
		mod := time.Now().Add(-age)
		if err := os.Chtimes(mcpPath, mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	rootJSON := func(root string) string {
		b, _ := json.Marshal(map[string]string{"root": root})
		return string(b)
	}
	expect := func(want, wantID string) {
		t.Helper()
		tok, id, ok := tokenFromNewestSession(time.Time{}, dir, nil)
		if want == "" {
			if ok {
				t.Errorf("got token %q from session %q, want none", tok, id)
			}
			return
		}
		if !ok || tok != want || id != wantID {
			t.Errorf("got (%q, %q, %v), want (%q, %q)", tok, id, ok, want, wantID)
		}
	}

	add("other-root", rootJSON(other), "mlrun_other", 0)
	expect("", "")
	add("no-root", `{}`, "mlrun_noroot", 0)
	add("broken-json", `not json`, "mlrun_broken", 0)
	add("wrong-prefix", rootJSON(dir), "plain_token", time.Minute)
	expect("", "")
	add("mine-old", rootJSON(dir), "mlrun_mine_old", 10*time.Minute)
	expect("mlrun_mine_old", "mine-old")
	add("mine-new", rootJSON(dir), "mlrun_mine_new", 5*time.Minute)
	expect("mlrun_mine_new", "mine-new")

	if _, _, ok := tokenFromNewestSession(time.Now().Add(-time.Minute), dir, nil); ok {
		t.Error("a session older than `since` was accepted")
	}
}

func TestKeeperBackoff(t *testing.T) {
	setupStub(t)
	keeperCooldown = time.Second
	missing := filepath.Join(t.TempDir(), "no-such-mlai")
	k := newRunKeeper(missing, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	// age sets how long ago the last spawn was and how many failures preceded it.
	age := func(fails int, ago time.Duration) {
		k.mu.Lock()
		k.fails, k.lastSpawn = fails, time.Now().Add(-ago)
		k.mu.Unlock()
	}
	refresh := func(wantCooldown bool, wantFails int) {
		t.Helper()
		_, err := k.Refresh(ctx, "")
		switch {
		case err == nil:
			t.Fatal("refresh with a missing binary succeeded")
		case wantCooldown && !strings.Contains(err.Error(), "cooling down"):
			t.Fatalf("err = %v, want a cooldown refusal", err)
		case !wantCooldown && !strings.Contains(err.Error(), "start keeper"):
			t.Fatalf("err = %v, want a spawn failure", err)
		}
		if _, fails, _ := keeperState(k); fails != wantFails {
			t.Fatalf("fails = %d, want %d", fails, wantFails)
		}
	}

	refresh(false, 1)
	// One failure doubles the wait to 2s: 1.5s after the last spawn is too soon.
	age(1, 1500*time.Millisecond)
	refresh(true, 1)
	age(1, 2500*time.Millisecond)
	refresh(false, 2)
	// The factor stops at 32x: 31s is too soon, 33s is enough, however many failures.
	age(100, 31*time.Second)
	refresh(true, 100)
	age(100, 33*time.Second)
	refresh(false, 101)
}

func TestKeeperSuccessResetsBackoff(t *testing.T) {
	env := setupStub(t)
	k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })
	k.mu.Lock()
	k.fails = 3
	k.mu.Unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tok, err := k.Refresh(ctx, "")
	if err != nil || !strings.HasPrefix(tok, "mlrun_") {
		t.Fatalf("Refresh = %q, %v", tok, err)
	}
	if _, fails, _ := keeperState(k); fails != 0 {
		t.Errorf("fails = %d after a successful spawn, want 0", fails)
	}
	if n := env.count(t, "--bg"); n != 1 {
		t.Errorf("--bg ran %d times, want 1", n)
	}
}

func TestKeeperStopsOnlyRejectedRuns(t *testing.T) {
	for _, tc := range []struct {
		name string
		age  time.Duration
		want int
	}{{"rejected", time.Minute, 1}, {"expired", tokenTTL + time.Minute, 0}} {
		t.Run(tc.name, func(t *testing.T) {
			env := setupStub(t)
			k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
			t.Cleanup(func() { waitKeeperIdle(t, k) })
			k.token, k.session, k.at = "mlrun_old", "old-session", time.Now().Add(-tc.age)
			if tok, err := k.Refresh(context.Background(), "mlrun_old"); err != nil || tok == "mlrun_old" {
				t.Fatalf("Refresh = %q, %v", tok, err)
			}
			if n := env.count(t, "agents"); n != tc.want {
				t.Errorf("agents stop ran %d times, want %d", n, tc.want)
			}
		})
	}
}

func TestSessionRootInsideGitRepo(t *testing.T) {
	repo := t.TempDir()
	dir := filepath.Join(repo, "workspace", ".keeper")
	session := t.TempDir()
	if err := errors.Join(os.Mkdir(filepath.Join(repo, ".git"), 0o700), os.MkdirAll(dir, 0o700)); err != nil {
		t.Fatal(err)
	}
	for root, want := range map[string]bool{repo: true, dir: true, filepath.Dir(dir): false, filepath.Dir(repo): false} {
		raw, _ := json.Marshal(map[string]string{"root": root})
		if err := os.WriteFile(filepath.Join(session, "session.json"), raw, 0o600); err != nil {
			t.Fatal(err)
		}
		if got := sessionRunsIn(session, dir); got != want {
			t.Errorf("root %q: got %v, want %v", root, got, want)
		}
	}
}

func TestKeeperIgnoresExistingSessions(t *testing.T) {
	setupStub(t)
	k := newRunKeeper(cfg.MLAI, cfg.WorkDir)
	t.Cleanup(func() { waitKeeperIdle(t, k) })
	other := filepath.Join(sessionsDir(), "already-there")
	root, _ := json.Marshal(map[string]string{"root": k.dir})
	if err := errors.Join(
		os.MkdirAll(filepath.Join(other, "engine"), 0o700),
		os.WriteFile(filepath.Join(other, "session.json"), root, 0o600),
		os.WriteFile(filepath.Join(other, "engine", "mcp.json"), []byte(`{"mcpServers":{"x":{"env":{"MODEL_API_KEY":"mlrun_foreign"}}}}`), 0o600),
	); err != nil {
		t.Fatal(err)
	}
	if tok, err := k.Refresh(context.Background(), ""); err != nil || tok == "mlrun_foreign" {
		t.Errorf("Refresh = %q, %v", tok, err)
	}
}

// ---------- direct mode ----------

var jsonHeader = map[string]string{"Content-Type": "application/json"}

func directServer(t *testing.T) (*server, http.Handler) {
	s := newTestServer(t)
	s.direct = true
	seedToken(s, "mlrun_seed")
	return s, testChain(s, "")
}

func TestDirectFailureFallsBackToCLI(t *testing.T) {
	for _, status := range []int{302, 400, 404, 429, 500, 503} {
		for path, body := range map[string]string{"/v1/chat/completions": chatBody, "/v1/messages": anthropicBody} {
			t.Run(fmt.Sprintf("%d%s", status, path), func(t *testing.T) {
				setupStub(t)
				_, h := directServer(t)
				hits := 0
				setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
					hits++
					resp := fakeResp(r, status, "gateway-detail")
					resp.Header.Set("Location", "https://elsewhere.example/")
					return resp, nil
				}))
				rec := serve(h, "POST", path, "", jsonHeader, strings.NewReader(body))
				if rec.Code != 200 || !strings.Contains(rec.Body.String(), "stub:hello") {
					t.Fatalf("got %d %s", rec.Code, rec.Body)
				}
				if hits != 1 {
					t.Errorf("gateway hit %d times, want 1 (no redirect)", hits)
				}
			})
		}
	}
}

func TestDirectRefreshesOnceOn401(t *testing.T) {
	env := setupStub(t)
	_, h := directServer(t)
	var tokens []string
	setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		tokens = append(tokens, r.Header.Get("Authorization"))
		if len(tokens) == 1 {
			return fakeResp(r, 401, ""), nil
		}
		return fakeResp(r, 200, gatewayOK), nil
	}))
	rec := serve(h, "POST", "/v1/chat/completions", "", jsonHeader, strings.NewReader(chatBody))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "direct says hi") {
		t.Fatalf("got %d %s", rec.Code, rec.Body)
	}
	if len(tokens) != 2 || tokens[0] == tokens[1] {
		t.Errorf("gateway calls = %d, want 2 with different tokens", len(tokens))
	}
	if n := env.count(t, "--bg"); n != 1 {
		t.Errorf("--bg ran %d times, want 1", n)
	}
}

func TestDirectConcurrencyLimit(t *testing.T) {
	setupStub(t)
	cfg.DirectConcurrency, cfg.QueueWait = 1, 100*time.Millisecond
	s, h := directServer(t)
	entered, release := make(chan struct{}), make(chan struct{})
	setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		entered <- struct{}{}
		<-release
		return fakeResp(r, 200, gatewayOK), nil
	}))
	first := make(chan int, 1)
	go func() {
		first <- serve(h, "POST", "/v1/chat/completions", "", jsonHeader, strings.NewReader(chatBody)).Code
	}()
	<-entered
	if rec := serve(h, "POST", "/v1/chat/completions", "", jsonHeader, strings.NewReader(chatBody)); rec.Code != 429 {
		t.Errorf("second request = %d, want 429", rec.Code)
	}
	close(release)
	if code := <-first; code != 200 {
		t.Errorf("first request = %d, want 200", code)
	}
	if len(s.directSem) != 0 {
		t.Error("direct slot was not released")
	}
}

func TestDirectTimeout(t *testing.T) {
	setupStub(t)
	cfg.Timeout = 200 * time.Millisecond
	s, _ := directServer(t)
	setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	}))
	t0 := time.Now()
	done, aerr := s.serveDirect(context.Background(), nil, false, func(*http.Response) bool { return true })
	if done || aerr != nil || time.Since(t0) > 5*time.Second {
		t.Errorf("done=%v err=%v after %s", done, aerr, time.Since(t0))
	}
	if len(s.directSem) != 0 {
		t.Error("direct slot was not released")
	}
}

func TestErrorsHideUpstreamText(t *testing.T) {
	if msg := classify("/home/someone/.mlai: credentials for a@b.example").Message; msg != errAuth.Message {
		t.Errorf("message = %q", msg)
	}
}

// ---------- tool calls ----------

func TestOnlyDeclaredToolsAreCalled(t *testing.T) {
	const text = `{"tool_calls":[{"name":"known","arguments":{}},{"name":"delete_all","arguments":{}}]}`
	for declared, want := range map[string]int{"known": 1, "other": 0} {
		t.Run(declared, func(t *testing.T) {
			setupStub(t)
			t.Setenv("STUB_MODE", "text")
			t.Setenv("STUB_TEXT", text)
			s := newTestServer(t)
			msgs := []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}}
			res := s.execute(context.Background(), "m", "", msgs, []toolDef{{Name: declared}}, "", 0)
			if res.err != nil {
				t.Fatal(res.err)
			}
			if len(res.answer.Tools) != want {
				t.Fatalf("tools = %v, want %d", res.answer.Tools, want)
			}
			if want == 1 && res.answer.Tools[0].Name != "known" {
				t.Errorf("tool = %q", res.answer.Tools[0].Name)
			}
			if want == 0 && res.answer.Text != text {
				t.Errorf("text = %q, want it unchanged", res.answer.Text)
			}
		})
	}
}

func TestFlagLikeModelIsRejected(t *testing.T) {
	env := setupStub(t)
	s := newTestServer(t)
	msgs := []chatMessage{{Role: "user", Content: json.RawMessage(`"hi"`)}}
	res := s.execute(context.Background(), "--mode", "", msgs, nil, "", 0)
	if res.err == nil || res.err.Status != 404 {
		t.Errorf("err = %v, want 404", res.err)
	}
	if n := len(env.calls(t)); n != 0 {
		t.Errorf("CLI ran %d times, want 0", n)
	}
}

// ---------- slow bodies ----------

func TestSlowBodyIsCutOff(t *testing.T) {
	saveGlobals(t)
	bodyReadTimeout = 300 * time.Millisecond
	srv := httptest.NewServer(testChain(&server{}, ""))
	defer srv.Close()
	conn, err := net.Dial("tcp", srv.Listener.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	fmt.Fprint(conn, "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{\"mess")
	conn.SetReadDeadline(time.Now().Add(3 * time.Second))
	raw, _ := io.ReadAll(conn)
	if !bytes.HasPrefix(raw, []byte("HTTP/1.1 400")) {
		t.Errorf("got %q, want a 400 within the deadline", raw)
	}
}

func TestLongRequestSurvivesBodyDeadline(t *testing.T) {
	saveGlobals(t)
	bodyReadTimeout = 300 * time.Millisecond
	srv := httptest.NewServer(withBodyDeadline(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var v any
		if err := decodeBody(w, r, &v); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		select {
		case <-r.Context().Done():
			http.Error(w, "cancelled", 500)
		case <-time.After(900 * time.Millisecond):
		}
	})))
	defer srv.Close()
	resp, err := http.Post(srv.URL, "application/json", strings.NewReader("{}"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
}
