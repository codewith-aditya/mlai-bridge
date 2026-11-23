package main

// handlers.go — OpenAI /v1/chat/completions and Anthropic /v1/messages on top of
// the mlai CLI, including tool-call emulation, streaming and caching.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ---------- shared types ----------

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// text flattens string content or [{type:text,text:...}] parts.
func (m chatMessage) text() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(m.Content, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Text != "" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return string(m.Content)
}

type toolDef struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
	// anthropic shape
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func (t toolDef) toolName() string {
	if t.Function.Name != "" {
		return t.Function.Name
	}
	return t.Name
}

func (t toolDef) toolDesc() string {
	if t.Function.Description != "" {
		return t.Function.Description
	}
	return t.Description
}

func (t toolDef) toolSchema() json.RawMessage {
	if len(t.Function.Parameters) > 0 {
		return t.Function.Parameters
	}
	return t.InputSchema
}

type cachedAnswer struct {
	Text    string        `json:"text"`
	Model   string        `json:"model"`
	Usage   Usage         `json:"usage"`
	Tools   []toolCallOut `json:"tools,omitempty"`
	Created int64         `json:"created"`
}

type toolCallOut struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ---------- prompt building ----------

func toolPreamble(tools []toolDef) string {
	var b strings.Builder
	b.WriteString("You can call tools. Tools available:\n")
	for _, t := range tools {
		schema := string(t.toolSchema())
		if len(schema) > 1500 {
			schema = schema[:1500] + "…"
		}
		b.WriteString(fmt.Sprintf("- %s: %s\n  input schema: %s\n", t.toolName(), t.toolDesc(), schema))
	}
	b.WriteString("\nIf a tool is needed, reply with ONLY this JSON and nothing else:\n")
	b.WriteString(`{"tool_calls":[{"name":"<tool name>","arguments":{...}}]}`)
	b.WriteString("\nIf no tool is needed, reply normally in plain text.\n\n")
	return b.String()
}

// buildPrompt turns a message list into a single prompt for `mlai -p`.
func buildPrompt(system string, msgs []chatMessage, tools []toolDef) string {
	var b strings.Builder
	if system != "" {
		b.WriteString(system)
		b.WriteString("\n\n")
	}
	if len(tools) > 0 {
		b.WriteString(toolPreamble(tools))
	}
	if len(msgs) > 1 {
		b.WriteString("Conversation so far:\n\n")
		for _, m := range msgs {
			t := strings.TrimSpace(m.text())
			if t == "" {
				continue
			}
			switch m.Role {
			case "assistant":
				b.WriteString("Assistant: " + t + "\n\n")
			case "tool":
				b.WriteString("Tool result: " + t + "\n\n")
			default:
				b.WriteString("User: " + t + "\n\n")
			}
		}
		b.WriteString("Reply to the last User message.")
	} else if len(msgs) == 1 {
		b.WriteString(msgs[0].text())
	}
	return b.String()
}

// parseToolCalls finds the emulated tool-call JSON in an answer.
func parseToolCalls(text string) ([]toolCallOut, string) {
	t := strings.TrimSpace(text)
	start := strings.Index(t, `{"tool_calls"`)
	if start < 0 {
		return nil, text
	}
	// trim to the last closing brace of that object
	frag := t[start:]
	var probe struct {
		ToolCalls []struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"tool_calls"`
	}
	if err := json.Unmarshal([]byte(frag), &probe); err != nil {
		// try to cut at the last }
		if end := strings.LastIndex(frag, "}"); end > 0 {
			if err2 := json.Unmarshal([]byte(frag[:end+1]), &probe); err2 != nil {
				return nil, text
			}
		} else {
			return nil, text
		}
	}
	if len(probe.ToolCalls) == 0 {
		return nil, text
	}
	out := make([]toolCallOut, 0, len(probe.ToolCalls))
	for _, tc := range probe.ToolCalls {
		out = append(out, toolCallOut{Name: tc.Name, Arguments: tc.Arguments})
	}
	// text before the JSON is kept as a preamble
	return out, strings.TrimSpace(t[:start])
}

// ---------- shared execution ----------

type execResult struct {
	answer   cachedAnswer
	err      *apiError
	cacheHit bool
}

func (s *server) execute(ctx context.Context, model, system string, msgs []chatMessage, tools []toolDef, effort string, maxTokens int) execResult {
	prompt := buildPrompt(system, msgs, tools)
	noCache := s.cache == nil || len(tools) > 0
	var key string
	if !noCache {
		key = cacheKey(map[string]any{"m": model, "s": system, "msgs": msgs, "e": effort})
		if raw, ok := s.cache.Get(key); ok {
			var ca cachedAnswer
			if json.Unmarshal(raw, &ca) == nil {
				return execResult{answer: ca, cacheHit: true}
			}
		}
	}

	release, aerr := s.acquire(ctx)
	if aerr != nil {
		return execResult{err: aerr}
	}
	defer release()

	requestsTotal.Add(1)
	activeRequests.Add(1)
	defer activeRequests.Add(-1)

	runCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()

	res := s.runner.Run(runCtx, prompt, model, effort)
	if res.Err != nil {
		requestsFailed.Add(1)
		e := classify(res.Err.Error())
		return execResult{err: &e}
	}

	ca := cachedAnswer{Text: res.Text, Model: res.Model, Usage: res.Usage, Created: time.Now().Unix()}
	if ca.Model == "" {
		ca.Model = model
	}
	if len(tools) > 0 {
		if calls, preamble := parseToolCalls(ca.Text); calls != nil {
			ca.Tools = calls
			ca.Text = preamble
		}
	}
	ca.Usage.TotalTokens = ca.Usage.PromptTokens + ca.Usage.CompletionTokens
	if !noCache {
		if raw, err := json.Marshal(ca); err == nil {
			s.cache.Put(key, raw)
		}
	}
	return execResult{answer: ca}
}

// ---------- OpenAI ----------

type openAIRequest struct {
	Model         string        `json:"model"`
	Messages      []chatMessage `json:"messages"`
	Stream        bool          `json:"stream"`
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage"`
	} `json:"stream_options"`
	Tools       []toolDef       `json:"tools"`
	ToolChoice  json.RawMessage `json:"tool_choice"`
	Temperature *float64        `json:"temperature"`
	MaxTokens   int             `json:"max_tokens"`
	User        string          `json:"user"`
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	// ids as the CLI reports them (`mlai -p … -m <bad>` prints the authoritative list)
	ids := []string{
		"pixis",             // fast, coding-tuned
		"pixis-opus",        // Opus 5.5R — strongest, default
		"pixis-sol",         // GPT 6.1R Sol — fast, large context
		"ml-ai-forge",       // ML.ai Forge (text only)
		"gpt-6-astra",       // most capable GPT (credits)
		"grok-4.7",          // xAI reasoning (credits)
		"mlai-coder",        // ML.ai Coder (credits)
		"glm-5.3",           // GLM 5.3 (credits, text only)
		"claude-sonnet-5.5", // Anthropic Sonnet (credits)
		"claude-fable-5.2",  // Anthropic Fable (credits)
	}
	data := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		data = append(data, map[string]any{
			"id": id, "object": "model", "created": 0, "owned_by": "ml.ai",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
}

func (s *server) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	reqID := w.Header().Get("X-Request-Id")
	if r.Method != http.MethodPost {
		writeOpenAIError(w, errBadRequest, reqID)
		return
	}
	var req openAIRequest
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(&req); err != nil {
		e := errBadRequest
		e.Message = "invalid JSON: " + err.Error()
		writeOpenAIError(w, e, reqID)
		return
	}
	if len(req.Messages) == 0 {
		e := errBadRequest
		e.Message = "messages is required"
		writeOpenAIError(w, e, reqID)
		return
	}
	model := req.Model
	if model == "" {
		model = cfg.DefaultModel
	}

	var system string
	var convo []chatMessage
	for _, m := range req.Messages {
		if m.Role == "system" {
			if system != "" {
				system += "\n"
			}
			system += m.text()
			continue
		}
		convo = append(convo, m)
	}

	// Fast path: talk to the gateway directly — real token streaming, no CLI overhead.
	if s.direct {
		body := openAIBodyFromOpenAI(req, model)
		if resp, derr := s.directRequest(r.Context(), body, req.Stream); derr == nil {
			if req.Stream {
				copySSE(w, resp)
				return
			}
			text, gwModel, usage, aerr := aggregateGateway(resp)
			if aerr == nil {
				if gwModel == "" {
					gwModel = model
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]any{
					"id": fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()), "object": "chat.completion",
					"created": time.Now().Unix(), "model": gwModel,
					"choices": []map[string]any{{
						"index": 0, "finish_reason": "stop",
						"message": map[string]any{"role": "assistant", "content": text},
					}},
					"usage": usage,
				})
				return
			}
			logger.Printf("direct aggregate failed: %v", aerr)
		} else {
			logger.Printf("direct path failed (%v) — falling back to CLI", derr)
		}
	}

	res := s.execute(r.Context(), model, system, convo, req.Tools, cfg.Effort, req.MaxTokens)
	if res.err != nil {
		writeOpenAIError(w, *res.err, reqID)
		return
	}
	answer := res.answer
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Cache", cacheFlag(res.cacheHit))
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		chunk := func(delta map[string]any, finish any) {
			payload := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": answer.Created,
				"model":   answer.Model,
				"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
			}
			b, _ := json.Marshal(payload)
			fmt.Fprintf(w, "data: %s\n\n", b)
			if flusher != nil {
				flusher.Flush()
			}
		}
		chunk(map[string]any{"role": "assistant"}, nil)
		if len(answer.Tools) > 0 {
			calls := make([]map[string]any, 0, len(answer.Tools))
			for i, tc := range answer.Tools {
				calls = append(calls, map[string]any{
					"index": i, "id": fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i),
					"type":     "function",
					"function": map[string]any{"name": tc.Name, "arguments": string(tc.Arguments)},
				})
			}
			chunk(map[string]any{"tool_calls": calls}, nil)
		} else {
			for _, piece := range splitText(answer.Text, 48) {
				chunk(map[string]any{"content": piece}, nil)
			}
		}
		chunk(map[string]any{}, "stop")
		// usage chunk (OpenAI shape): empty choices + usage, always sent so clients
		// that ignore stream_options still see token accounting.
		{
			payload := map[string]any{
				"id": id, "object": "chat.completion.chunk", "created": answer.Created,
				"model": answer.Model, "choices": []any{},
				"usage": map[string]any{
					"prompt_tokens":     answer.Usage.PromptTokens,
					"completion_tokens": answer.Usage.CompletionTokens,
					"total_tokens":      answer.Usage.TotalTokens,
					"prompt_tokens_details": map[string]any{
						"cached_tokens": answer.Usage.CachedTokens,
					},
				},
			}
			b, _ := json.Marshal(payload)
			fmt.Fprintf(w, "data: %s\n\n", b)
			if flusher != nil {
				flusher.Flush()
			}
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	msg := map[string]any{"role": "assistant", "content": answer.Text}
	finish := "stop"
	if len(answer.Tools) > 0 {
		finish = "tool_calls"
		calls := make([]map[string]any, 0, len(answer.Tools))
		for i, tc := range answer.Tools {
			calls = append(calls, map[string]any{
				"id": fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i), "type": "function",
				"function": map[string]any{"name": tc.Name, "arguments": string(tc.Arguments)},
			})
		}
		msg["tool_calls"] = calls
		if strings.TrimSpace(answer.Text) == "" {
			msg["content"] = nil
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", cacheFlag(res.cacheHit))
	json.NewEncoder(w).Encode(map[string]any{
		"id": id, "object": "chat.completion", "created": answer.Created, "model": answer.Model,
		"choices": []map[string]any{{"index": 0, "message": msg, "finish_reason": finish}},
		"usage": map[string]any{
			"prompt_tokens": answer.Usage.PromptTokens, "completion_tokens": answer.Usage.CompletionTokens,
			"total_tokens":          answer.Usage.TotalTokens,
			"prompt_tokens_details": map[string]any{"cached_tokens": answer.Usage.CachedTokens},
		},
	})
}

// ---------- Anthropic ----------

type anthropicRequest struct {
	Model     string          `json:"model"`
	System    json.RawMessage `json:"system"`
	Messages  []chatMessage   `json:"messages"`
	MaxTokens int             `json:"max_tokens"`
	Stream    bool            `json:"stream"`
	Tools     []toolDef       `json:"tools"`
	Metadata  map[string]any  `json:"metadata"`
}

func (s *server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	reqID := w.Header().Get("X-Request-Id")
	if r.Method != http.MethodPost {
		writeAnthropicError(w, errBadRequest, reqID)
		return
	}
	var req anthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		e := errBadRequest
		e.Message = "invalid JSON: " + err.Error()
		writeAnthropicError(w, e, reqID)
		return
	}
	if len(req.Messages) == 0 {
		e := errBadRequest
		e.Message = "messages is required"
		writeAnthropicError(w, e, reqID)
		return
	}
	model := req.Model
	if model == "" {
		model = cfg.DefaultModel
	}
	system := ""
	if len(req.System) > 0 {
		var sv string
		if json.Unmarshal(req.System, &sv) == nil {
			system = sv
		} else {
			var parts []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(req.System, &parts) == nil {
				var b strings.Builder
				for _, p := range parts {
					if p.Text != "" {
						b.WriteString(p.Text)
						b.WriteString("\n")
					}
				}
				system = strings.TrimSpace(b.String())
			}
		}
	}

	// Fast path: gateway direct (real streaming), CLI as fallback.
	if s.direct {
		body := openAIBodyFromAnthropic(req, model)
		if resp, derr := s.directRequest(r.Context(), body, req.Stream); derr == nil {
			if req.Stream {
				streamToAnthropic(w, resp, model)
				return
			}
			anthropicFromGateway(w, resp, model)
			return
		} else {
			logger.Printf("direct path failed (%v) — falling back to CLI", derr)
		}
	}

	res := s.execute(r.Context(), model, system, req.Messages, req.Tools, cfg.Effort, req.MaxTokens)
	if res.err != nil {
		writeAnthropicError(w, *res.err, reqID)
		return
	}
	answer := res.answer
	id := fmt.Sprintf("msg_%d", time.Now().UnixNano())

	stopReason := "end_turn"
	if len(answer.Tools) > 0 {
		stopReason = "tool_use"
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("X-Cache", cacheFlag(res.cacheHit))
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		send := func(event string, data any) {
			b, _ := json.Marshal(data)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, b)
			if flusher != nil {
				flusher.Flush()
			}
		}
		send("message_start", map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id": id, "type": "message", "role": "assistant", "model": answer.Model,
				"content": []any{}, "stop_reason": nil,
				"usage": map[string]any{"input_tokens": answer.Usage.PromptTokens, "output_tokens": 0},
			},
		})
		if len(answer.Tools) > 0 {
			for i, tc := range answer.Tools {
				send("content_block_start", map[string]any{
					"type": "content_block_start", "index": i,
					"content_block": map[string]any{"type": "tool_use", "id": fmt.Sprintf("toolu_%d_%d", time.Now().UnixNano(), i), "name": tc.Name, "input": map[string]any{}},
				})
				send("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": i,
					"delta": map[string]any{"type": "input_json_delta", "partial_json": string(tc.Arguments)},
				})
				send("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
			}
		} else {
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": 0,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
			for _, piece := range splitText(answer.Text, 48) {
				send("content_block_delta", map[string]any{
					"type": "content_block_delta", "index": 0,
					"delta": map[string]any{"type": "text_delta", "text": piece},
				})
			}
			send("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		}
		send("message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": answer.Usage.CompletionTokens},
		})
		send("message_stop", map[string]any{"type": "message_stop"})
		return
	}

	content := make([]map[string]any, 0, 2)
	if strings.TrimSpace(answer.Text) != "" {
		content = append(content, map[string]any{"type": "text", "text": answer.Text})
	}
	for i, tc := range answer.Tools {
		var input map[string]any
		if json.Unmarshal(tc.Arguments, &input) != nil {
			input = map[string]any{}
		}
		content = append(content, map[string]any{
			"type": "tool_use", "id": fmt.Sprintf("toolu_%d_%d", time.Now().UnixNano(), i),
			"name": tc.Name, "input": input,
		})
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", cacheFlag(res.cacheHit))
	json.NewEncoder(w).Encode(map[string]any{
		"id": id, "type": "message", "role": "assistant", "model": answer.Model,
		"content": content, "stop_reason": stopReason, "stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens": answer.Usage.PromptTokens, "output_tokens": answer.Usage.CompletionTokens,
			"cache_read_input_tokens": answer.Usage.CachedTokens,
		},
	})
}

func (s *server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	var req anthropicRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAnthropicError(w, errBadRequest, w.Header().Get("X-Request-Id"))
		return
	}
	n := 0
	for _, m := range req.Messages {
		n += len(m.text()) / 4
	}
	if len(req.System) > 0 {
		n += len(req.System) / 4
	}
	if n == 0 {
		n = 1
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"input_tokens": n})
}

// ---------- helpers ----------

func cacheFlag(hit bool) string {
	if hit {
		return "HIT"
	}
	return "MISS"
}

func splitText(s string, n int) []string {
	if s == "" {
		return nil
	}
	var out []string
	for len(s) > n {
		out = append(out, s[:n])
		s = s[n:]
	}
	if s != "" {
		out = append(out, s)
	}
	return out
}
