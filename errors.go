package main

// errors.go — one place that turns CLI/server failures into API error responses
// with stable codes (OpenAI and Anthropic shapes).

import (
	"encoding/json"
	"net/http"
	"strings"
)

type apiError struct {
	Status  int
	Type    string // openai error.type
	Code    string // openai error.code
	AType   string // anthropic error.type
	Message string
}

func (e apiError) Error() string { return e.Message }

var (
	errAuth        = apiError{401, "authentication_error", "not_signed_in", "authentication_error", "mlai CLI is not signed in — run `mlai login` on the bridge host"}
	errRate        = apiError{429, "rate_limit_error", "rate_limited", "rate_limit_error", "upstream session limit reached — retry shortly"}
	errBudget      = apiError{429, "rate_limit_error", "daily_cap", "rate_limit_error", "account allowance or credits exhausted"}
	errUpgrade     = apiError{400, "invalid_request_error", "upgrade_required", "invalid_request_error", "mlai CLI is out of date — run `mlai update`"}
	errBusy        = apiError{429, "rate_limit_error", "concurrent", "rate_limit_error", "another request is in flight — retry shortly"}
	errBadRequest  = apiError{400, "invalid_request_error", "invalid_request", "invalid_request_error", "bad request"}
	errModel       = apiError{404, "invalid_request_error", "model_not_found", "not_found_error", "unknown model"}
	errTimeout     = apiError{504, "server_error", "timeout", "timeout_error", "the agent took too long and was stopped"}
	errUpstream    = apiError{502, "server_error", "upstream_error", "api_error", "the agent CLI failed"}
	errUnavailable = apiError{503, "server_error", "unavailable", "overloaded_error", "bridge is not ready"}
)

// classify maps a CLI error string to a stable apiError.
func classify(msg string) apiError {
	m := strings.ToLower(msg)
	switch {
	case strings.Contains(m, "not signed in"), strings.Contains(m, "unauthorized"), strings.Contains(m, "credentials"):
		e := errAuth
		e.Message = msg
		return e
	case strings.Contains(m, "daily budget"), strings.Contains(m, "credits"), strings.Contains(m, "allowance"):
		e := errBudget
		e.Message = msg
		return e
	case strings.Contains(m, "too many requests"), strings.Contains(m, "hourly ceiling"), strings.Contains(m, "retry-after"), strings.Contains(m, "capacity"):
		e := errRate
		e.Message = msg
		return e
	case strings.Contains(m, "out of date"), strings.Contains(m, "upgrade_required"):
		e := errUpgrade
		e.Message = msg
		return e
	case strings.Contains(m, "another free-pi session"), strings.Contains(m, "concurrent"):
		e := errBusy
		e.Message = msg
		return e
	case strings.Contains(m, "deadline exceeded"), strings.Contains(m, "killed"), strings.Contains(m, "timeout"):
		e := errTimeout
		e.Message = msg
		return e
	case strings.Contains(m, "ubuntu"), strings.Contains(m, "no such file"), strings.Contains(m, "exec:"):
		e := errUnavailable
		e.Message = msg
		return e
	}
	e := errUpstream
	e.Message = msg
	return e
}

func writeOpenAIError(w http.ResponseWriter, e apiError, requestID string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-Id", requestID)
	w.Header().Set("Retry-After", retryAfter(e))
	w.WriteHeader(e.Status)
	json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": e.Message,
			"type":    e.Type,
			"code":    e.Code,
			"param":   nil,
		},
	})
}

func writeAnthropicError(w http.ResponseWriter, e apiError, requestID string) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("request-id", requestID)
	if r := retryAfter(e); r != "" {
		w.Header().Set("retry-after", r)
	}
	w.WriteHeader(e.Status)
	json.NewEncoder(w).Encode(map[string]any{
		"type":  "error",
		"error": map[string]string{"type": e.AType, "message": e.Message},
	})
}

func retryAfter(e apiError) string {
	if e.Status == 429 {
		return "5"
	}
	return ""
}
