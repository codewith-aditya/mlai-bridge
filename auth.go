package main

// auth.go — optional bearer-key auth + CORS for exposure beyond loopback.

import (
	"crypto/subtle"
	"mime"
	"net"
	"net/http"
	"strings"
)

// isLoopbackHost reports whether host (no port) is "localhost" or a loopback IP.
func isLoopbackHost(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// apiKeyGate wraps the mux and rejects requests without the configured key.
// /healthz and /metrics stay open so monitors work. Without a key the bridge is
// loopback-only: it rejects foreign Host headers (DNS rebinding) and non-JSON
// POSTs (cross-site form posts), and sends no CORS headers.
func apiKeyGate(next http.Handler, key string) http.Handler {
	if key == "" {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			host := r.Host
			if h, _, err := net.SplitHostPort(host); err == nil {
				host = h
			} else {
				host = strings.Trim(host, "[]")
			}
			if !isLoopbackHost(host) {
				writeJSONAuthError(w, r, errForbiddenHost)
				return
			}
			if r.Method == http.MethodPost {
				mt, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
				if err != nil || mt != "application/json" {
					writeJSONAuthError(w, r, errMediaType)
					return
				}
			}
			next.ServeHTTP(w, r)
		})
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
		}
		if subtle.ConstantTimeCompare([]byte(token), []byte(key)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="mlai-bridge"`)
			writeJSONAuthError(w, r, errBadKey)
			return
		}
		writeCORS(w)
		next.ServeHTTP(w, r)
	})
}

func writeCORS(w http.ResponseWriter) {
	h := w.Header()
	h.Set("Access-Control-Allow-Origin", "*")
	h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, x-api-key, api-key, anthropic-version")
	h.Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

// writeJSONAuthError picks the error shape the caller expects.
func writeJSONAuthError(w http.ResponseWriter, r *http.Request, e apiError) {
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		writeAnthropicError(w, e, w.Header().Get("X-Request-Id"))
		return
	}
	writeOpenAIError(w, e, w.Header().Get("X-Request-Id"))
}
