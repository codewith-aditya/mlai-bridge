package main

// api_direct.go — handlers that use the gateway directly (real streaming),
// falling back to the CLI path when the gateway is unavailable.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// openAIBody builds the gateway request from an OpenAI-shaped request.
func openAIBodyFromOpenAI(req openAIRequest, model string) []byte {
	body := map[string]any{
		"model":    model,
		"messages": req.Messages,
		"stream":   req.Stream,
	}
	if req.Stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if len(req.Tools) > 0 {
		body["tools"] = req.Tools
	}
	if cfg.Effort != "" {
		body["reasoning_effort"] = cfg.Effort
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	raw, _ := json.Marshal(body)
	return raw
}

// openAIBodyFromAnthropic converts an Anthropic Messages request to OpenAI shape.
func openAIBodyFromAnthropic(req anthropicRequest, model string) []byte {
	msgs := make([]map[string]any, 0, len(req.Messages)+1)
	if len(req.System) > 0 {
		var s string
		if json.Unmarshal(req.System, &s) != nil {
			s = string(req.System)
		}
		if strings.TrimSpace(s) != "" {
			msgs = append(msgs, map[string]any{"role": "system", "content": s})
		}
	}
	for _, m := range req.Messages {
		msgs = append(msgs, map[string]any{"role": m.Role, "content": m.text()})
	}
	body := map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   req.Stream,
	}
	if req.Stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if cfg.Effort != "" {
		body["reasoning_effort"] = cfg.Effort
	}
	raw, _ := json.Marshal(body)
	return raw
}

// copySSE streams the gateway's SSE straight through (OpenAI clients).
func copySSE(w http.ResponseWriter, resp *http.Response) {
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 8192)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// aggregateGateway reads a non-stream gateway response and returns text + usage.
func aggregateGateway(resp *http.Response) (text, model string, usage map[string]any, err error) {
	defer resp.Body.Close()
	var doc struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
		} `json:"choices"`
		Usage map[string]any `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		return "", "", nil, err
	}
	if len(doc.Choices) == 0 {
		return "", doc.Model, doc.Usage, fmt.Errorf("gateway returned no choices")
	}
	return doc.Choices[0].Message.Content, doc.Model, doc.Usage, nil
}

// streamToAnthropic translates the gateway's OpenAI SSE into Anthropic events.
func streamToAnthropic(w http.ResponseWriter, resp *http.Response, model string) {
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(200)
	flusher, _ := w.(http.Flusher)
	send := func(event string, data any) {
		b, _ := json.Marshal(data)
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
		if flusher != nil {
			flusher.Flush()
		}
	}
	id := fmt.Sprintf("msg_%d", time.Now().UnixNano())
	started := false
	usage := map[string]any{}
	start := func() {
		if started {
			return
		}
		started = true
		send("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": model,
				"content": []any{}, "stop_reason": nil,
				"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
			},
		})
		send("content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 4<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var ev struct {
			Model   string `json:"model"`
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
				} `json:"delta"`
				FinishReason any `json:"finish_reason"`
			} `json:"choices"`
			Usage map[string]any `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &ev) != nil {
			continue
		}
		if ev.Model != "" {
			model = ev.Model
		}
		if ev.Usage != nil {
			usage = ev.Usage
		}
		text := ""
		for _, ch := range ev.Choices {
			if ch.Delta.Content != "" {
				text += ch.Delta.Content
			}
		}
		if text == "" {
			continue
		}
		start()
		send("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "text_delta", "text": text},
		})
	}
	start()
	send("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	in := toInt(usage["prompt_tokens"])
	out := toInt(usage["completion_tokens"])
	cached := 0
	if d, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		cached = toInt(d["cached_tokens"])
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": "end_turn", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": out},
	})
	send("message_stop", map[string]any{"type": "message_stop"})
	_ = in
	_ = cached
}

func toInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	}
	return 0
}

// anthropicFromGateway handles a non-stream Anthropic request via the gateway.
func anthropicFromGateway(w http.ResponseWriter, resp *http.Response, model string) {
	text, gwModel, usage, err := aggregateGateway(resp)
	if err != nil {
		writeAnthropicError(w, errUpstream, w.Header().Get("X-Request-Id"))
		return
	}
	if gwModel != "" {
		model = gwModel
	}
	in := toInt(usage["prompt_tokens"])
	out := toInt(usage["completion_tokens"])
	cached := 0
	if d, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		cached = toInt(d["cached_tokens"])
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": fmt.Sprintf("msg_%d", time.Now().UnixNano()), "type": "message",
		"role": "assistant", "model": model,
		"content":     []map[string]any{{"type": "text", "text": text}},
		"stop_reason": "end_turn", "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": in, "output_tokens": out, "cache_read_input_tokens": cached},
	})
}

var _ = io.Discard
