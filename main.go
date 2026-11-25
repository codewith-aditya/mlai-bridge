package main

// main.go — server wiring, config, health, metrics.

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const version = "1.0.0"

type config struct {
	Port              int
	MLAI              string
	WorkDir           string
	DefaultModel      string
	Concurrency       int
	DirectConcurrency int
	QueueWait         time.Duration
	Timeout           time.Duration
	CacheTTL          time.Duration
	CacheSize         int
	Effort            string
	Direct            bool
	Bind              string
	APIKey            string
}

var (
	cfg    = config{}
	logger = log.New(os.Stdout, "", log.LstdFlags|log.LUTC)

	requestsTotal  atomic.Int64
	requestsFailed atomic.Int64
	activeRequests atomic.Int64
	startedAt      = time.Now()
)

func initFlags() {
	flag.IntVar(&cfg.Port, "port", envInt("MLAI_PORT", 8197), "listen port (127.0.0.1)")
	flag.StringVar(&cfg.MLAI, "mlai", envStr("MLAI_BIN", "mlai"), "path to the mlai CLI")
	flag.StringVar(&cfg.WorkDir, "workdir", envStr("MLAI_WORKDIR", defaultWorkDir()), "folder the agent runs in")
	flag.StringVar(&cfg.DefaultModel, "model", envStr("MLAI_MODEL", "pixis-opus"), "default model id")
	flag.IntVar(&cfg.Concurrency, "concurrency", envInt("MLAI_CONCURRENCY", 1), "max parallel CLI runs")
	flag.IntVar(&cfg.DirectConcurrency, "direct-concurrency", envInt("MLAI_DIRECT_CONCURRENCY", 8), "max parallel gateway requests")
	flag.DurationVar(&cfg.QueueWait, "queue-wait", 30*time.Second, "how long a request waits for a free slot")
	flag.DurationVar(&cfg.Timeout, "timeout", 10*time.Minute, "per-request wall clock limit")
	flag.DurationVar(&cfg.CacheTTL, "cache-ttl", 10*time.Minute, "identical-request cache TTL (0 disables)")
	flag.IntVar(&cfg.CacheSize, "cache-size", 512, "cache entries")
	flag.StringVar(&cfg.Effort, "effort", envStr("MLAI_EFFORT", ""), "reasoning effort: low|medium|high|max")
	flag.BoolVar(&cfg.Direct, "direct", envBool("MLAI_DIRECT", true), "talk to the gateway directly (real token streaming) with CLI fallback")
	flag.StringVar(&cfg.Bind, "bind", envStr("MLAI_BIND", "127.0.0.1"), "listen address (0.0.0.0 to expose; requires an API key)")
	// The env default is applied after Parse so `-h` never prints the key.
	flag.StringVar(&cfg.APIKey, "api-key", "", "require this bearer key on /v1/* (env MLAI_API_KEY; empty = open, loopback only)")
	flag.Parse()
	cfg.Bind = strings.Trim(cfg.Bind, "[]")
	cfg.APIKey = strings.TrimSpace(cfg.APIKey)
	if cfg.APIKey == "" {
		cfg.APIKey = strings.TrimSpace(os.Getenv("MLAI_API_KEY"))
	}
}

// validateConfig refuses to listen beyond loopback without an API key.
func validateConfig(c config) error {
	if c.APIKey == "" && !isLoopbackHost(c.Bind) {
		return fmt.Errorf("bind address %q is not loopback: set MLAI_API_KEY (or -api-key) to expose the bridge", c.Bind)
	}
	return nil
}

func defaultWorkDir() string {
	if h, err := os.UserHomeDir(); err == nil {
		return filepath.Join(h, "mlai-bridge", "workspace")
	}
	return "workspace"
}

func envBool(k string, def bool) bool {
	if v := os.Getenv(k); v != "" {
		return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
	}
	return def
}

func envStr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envInt(k string, d int) int {
	if v := os.Getenv(k); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil {
			return n
		}
	}
	return d
}

// childEnv is the environment for every child process: the bridge's own key
// stays out of it.
func childEnv() []string {
	env := os.Environ()
	out := make([]string, 0, len(env)+2)
	for _, kv := range env {
		if name, _, _ := strings.Cut(kv, "="); strings.EqualFold(name, "MLAI_API_KEY") {
			continue
		}
		out = append(out, kv)
	}
	return append(out, "NO_COLOR=1", "CI=1")
}

type server struct {
	runner    *Runner
	cache     *LRU
	sem       chan struct{}
	directSem chan struct{}
	keeper    *runKeeper
	direct    bool

	healthMu sync.Mutex
	healthAt time.Time
	healthOK bool
}

func newServer() *server {
	os.MkdirAll(cfg.WorkDir, 0o700)
	var c *LRU
	if cfg.CacheTTL > 0 {
		c = NewLRU(cfg.CacheSize, cfg.CacheTTL)
	}
	return &server{
		runner:    &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: int(cfg.Timeout.Seconds())},
		cache:     c,
		sem:       make(chan struct{}, maxInt(1, cfg.Concurrency)),
		directSem: make(chan struct{}, maxInt(1, cfg.DirectConcurrency)),
		keeper:    newRunKeeper(cfg.MLAI, cfg.WorkDir),
		direct:    cfg.Direct,
	}
}

func (s *server) acquire(ctx context.Context, sem chan struct{}) (func(), *apiError) {
	wait := time.NewTimer(cfg.QueueWait)
	defer wait.Stop()
	select {
	case sem <- struct{}{}:
		return func() { <-sem }, nil
	case <-wait.C:
		e := errBusy
		return nil, &e
	case <-ctx.Done():
		e := errBusy
		e.Message = "client closed the request"
		return nil, &e
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	// The lock is held during the exec so concurrent calls start at most one process.
	s.healthMu.Lock()
	if s.healthAt.IsZero() || time.Since(s.healthAt) >= 60*time.Second {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		cmd := exec.CommandContext(ctx, cfg.MLAI, "whoami")
		cmd.Env = childEnv()
		cmd.WaitDelay = 5 * time.Second
		out, _ := cmd.CombinedOutput()
		cancel()
		s.healthOK = strings.HasPrefix(strings.TrimSpace(string(out)), "Signed in")
		s.healthAt = time.Now()
	}
	signedIn := s.healthOK
	s.healthMu.Unlock()

	status := "ok"
	if !signedIn {
		status = "degraded"
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":    status,
		"version":   version,
		"uptime_s":  int(time.Since(startedAt).Seconds()),
		"signed_in": signedIn,
	})
}

func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	hits, miss := int64(0), int64(0)
	if s.cache != nil {
		hits, miss = s.cache.Stats()
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE mlai_requests_total counter\nmlai_requests_total %d\n", requestsTotal.Load())
	fmt.Fprintf(w, "# TYPE mlai_requests_failed_total counter\nmlai_requests_failed_total %d\n", requestsFailed.Load())
	fmt.Fprintf(w, "# TYPE mlai_cache_hits_total counter\nmlai_cache_hits_total %d\nmlai_cache_misses_total %d\n", hits, miss)
	fmt.Fprintf(w, "# TYPE mlai_active_requests gauge\nmlai_active_requests %d\n", activeRequests.Load())
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

var requestIDPattern = regexp.MustCompile(`^[A-Za-z0-9._-]{1,64}$`)

func withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if !requestIDPattern.MatchString(id) {
			id = fmt.Sprintf("req_%d", time.Now().UnixNano())
		}
		w.Header().Set("X-Request-Id", id)
		r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		t0 := time.Now()
		next.ServeHTTP(w, r)
		logger.Printf("%s %q %q %s", id, r.Method, r.URL.Path, time.Since(t0).Round(time.Millisecond))
	})
}

// bodyReadTimeout bounds how long a client may take to send a request body.
// http.Server.ReadTimeout would also cancel long streaming responses, so the
// deadline is set per request and cleared by decodeBody once the body is in.
var bodyReadTimeout = 30 * time.Second

func withBodyDeadline(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NewResponseController(w).SetReadDeadline(time.Now().Add(bodyReadTimeout))
		next.ServeHTTP(w, r)
	})
}

// decodeBody reads the whole request body, clears the read deadline, then decodes JSON.
func decodeBody(w http.ResponseWriter, r *http.Request, v any) error {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			return errors.New("request body too large")
		}
		return errors.New("could not read the request body")
	}
	// Only on success: after a failed read the server still drains the body.
	http.NewResponseController(w).SetReadDeadline(time.Time{})
	return json.Unmarshal(raw, v)
}

func main() {
	initFlags()
	if err := validateConfig(cfg); err != nil {
		logger.Fatal(err)
	}
	s := newServer()

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages/count_tokens", s.handleCountTokens)

	addr := net.JoinHostPort(cfg.Bind, strconv.Itoa(cfg.Port))
	base, cancelBase := context.WithCancel(context.Background())
	defer cancelBase()
	srv := &http.Server{
		BaseContext:       func(net.Listener) context.Context { return base },
		Addr:              addr,
		Handler:           withBodyDeadline(withMiddleware(apiKeyGate(mux, cfg.APIKey))),
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
		MaxHeaderBytes:    64 << 10,
	}

	go func() {
		auth := "open"
		if cfg.APIKey != "" {
			auth = "bearer key required"
		}
		logger.Printf("mlai-bridge %s on http://%s  (mlai=%s model=%s concurrency=%d cache=%s auth=%s)",
			version, addr, cfg.MLAI, cfg.DefaultModel, cap(s.sem), cfg.CacheTTL, auth)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Fatalf("listen: %v", err)
		}
	}()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	logger.Print("shutting down")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if srv.Shutdown(ctx) != nil {
		cancelBase() // kill CLI runs that outlived the grace period, then let handlers return
		ctx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel2()
		srv.Shutdown(ctx2)
	}
	s.keeper.Close()
}
