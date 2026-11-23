package main

// auth.go — optional bearer-key auth + CORS for exposure beyond loopback.

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// apiKeyGate wraps the mux and rejects requests without the configured key.
// /healthz and /metrics stay open so monitors work.
func apiKeyGate(next http.Handler, key string) http.Handler {
	if key == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			writeCORS(w)
			w.WriteHeader(http.StatusNoContent)
			return
		}
		switch r.URL.Path {
		case "/healthz", "/health", "/metrics":
			next.ServeHTTP(w, r)
			return
		}
		token := ""
		if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
			token = strings.TrimSpace(strings.TrimPrefix(h, "Bearer "))
		} else if h := r.Header.Get("x-api-key"); h != "" {
			token = strings.TrimSpace(h)
		} else if h := r.Header.Get("api-key"); h != "" {
			token = strings.TrimSpace(h)
		} else if q := r.URL.Query().Get("key"); q != "" {
			token = q
		}
		if subtle.ConstantTimeCompare([]byte(token), []byte(key)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="mlai-bridge"`)
			writeJSONAuthError(w, r)
			return
		}
		writeCORS(w)
		next.ServeHTTP(w, r)
	})
}

func writeCORS(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, anthropic-version")
	h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

// writeJSONAuthError picks the error shape the caller expects.
func writeJSONAuthError(w http.ResponseWriter, r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		writeAnthropicError(w, errBadKey, w.Header().Get("X-Request-Id"))
		return
	}
	writeOpenAIError(w, errBadKey, w.Header().Get("X-Request-Id"))
}
