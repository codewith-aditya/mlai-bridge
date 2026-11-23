package main

// main.go — server wiring, config, health, metrics.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"
)

const version = "1.0.0"

type config struct {
	Port         int
	MLAI         string
	WorkDir      string
	DefaultModel string
	Concurrency  int
	QueueWait    time.Duration
	Timeout      time.Duration
	CacheTTL     time.Duration
	CacheSize    int
	Effort       string
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
	flag.StringVar(&cfg.WorkDir, "workdir", envStr("MLAI_WORKDIR", "/root/mlai-bridge/workspace"), "folder the agent runs in")
	flag.StringVar(&cfg.DefaultModel, "model", envStr("MLAI_MODEL", "pixis-opus"), "default model id")
	flag.IntVar(&cfg.Concurrency, "concurrency", envInt("MLAI_CONCURRENCY", 1), "max parallel CLI runs")
	flag.DurationVar(&cfg.QueueWait, "queue-wait", 30*time.Second, "how long a request waits for a free slot")
	flag.DurationVar(&cfg.Timeout, "timeout", 10*time.Minute, "per-request wall clock limit")
	flag.DurationVar(&cfg.CacheTTL, "cache-ttl", 10*time.Minute, "identical-request cache TTL (0 disables)")
	flag.IntVar(&cfg.CacheSize, "cache-size", 512, "cache entries")
	flag.StringVar(&cfg.Effort, "effort", envStr("MLAI_EFFORT", ""), "reasoning effort: low|medium|high|max")
	flag.Parse()
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

type server struct {
	runner *Runner
	cache  *LRU
	sem    chan struct{}
}

func newServer() *server {
	os.MkdirAll(cfg.WorkDir, 0o755)
	var c *LRU
	if cfg.CacheTTL > 0 {
		c = NewLRU(cfg.CacheSize, cfg.CacheTTL)
	}
	return &server{
		runner: &Runner{Bin: cfg.MLAI, WorkDir: cfg.WorkDir, Timeout: int(cfg.Timeout.Seconds())},
		cache:  c,
		sem:    make(chan struct{}, maxInt(1, cfg.Concurrency)),
	}
}

func (s *server) acquire(ctx context.Context) (func(), *apiError) {
	wait := time.NewTimer(cfg.QueueWait)
	defer wait.Stop()
	select {
	case s.sem <- struct{}{}:
		return func() { <-s.sem }, nil
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
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
	defer cancel()
	out, _ := exec.CommandContext(ctx, cfg.MLAI, "whoami").CombinedOutput()
	who := strings.TrimSpace(string(out))
	signedIn := strings.HasPrefix(who, "Signed in")
	status := "ok"
	if !signedIn {
		status = "degraded"
	}
	hits, miss := int64(0), int64(0)
	if s.cache != nil {
		hits, miss = s.cache.Stats()
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":      status,
		"version":     version,
		"uptime_s":    int(time.Since(startedAt).Seconds()),
		"signed_in":   signedIn,
		"whoami":      who,
		"workdir":     filepath.Clean(cfg.WorkDir),
		"concurrency": cap(s.sem),
		"cache":       map[string]any{"hits": hits, "miss": miss, "ttl_s": int(cfg.CacheTTL.Seconds())},
		"requests":    map[string]any{"total": requestsTotal.Load(), "failed": requestsFailed.Load(), "active": activeRequests.Load()},
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

func withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-Id")
		if id == "" {
			id = fmt.Sprintf("req_%d", time.Now().UnixNano())
		}
		w.Header().Set("X-Request-Id", id)
		if r.Method == http.MethodPost {
			r.Body = http.MaxBytesReader(w, r.Body, 2<<20)
		}
		t0 := time.Now()
		next.ServeHTTP(w, r)
		logger.Printf("%s %s %s %s", id, r.Method, r.URL.Path, time.Since(t0).Round(time.Millisecond))
	})
}

func main() {
	initFlags()
	s := newServer()

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleOpenAIChat)
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/messages/count_tokens", s.handleCountTokens)

	addr := fmt.Sprintf("127.0.0.1:%d", cfg.Port)
	srv := &http.Server{
		Addr:              addr,
		Handler:           withMiddleware(mux),
		ReadHeaderTimeout: 15 * time.Second,
		IdleTimeout:       120 * time.Second,
	}

	go func() {
		logger.Printf("mlai-bridge %s on http://%s  (mlai=%s model=%s concurrency=%d cache=%s)",
			version, addr, cfg.MLAI, cfg.DefaultModel, cap(s.sem), cfg.CacheTTL)
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
	srv.Shutdown(ctx)
}
