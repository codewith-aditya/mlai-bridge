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
	if tools := cleanTools(req.Tools, false); len(tools) > 0 {
		body["tools"] = tools
		if tc := strings.TrimSpace(string(req.ToolChoice)); tc != "" && tc != "null" {
			body["tool_choice"] = req.ToolChoice
		}
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
		if s := anthropicText(req.System); strings.TrimSpace(s) != "" {
			msgs = append(msgs, map[string]any{"role": "system", "content": s})
		}
	}
	msgs = append(msgs, anthropicMessages(req.Messages)...)
	body := map[string]any{
		"model":    model,
		"messages": msgs,
		"stream":   req.Stream,
	}
	if req.Stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if tools := cleanTools(req.Tools, true); len(tools) > 0 {
		body["tools"] = tools
		if tc := anthropicToolChoice(req.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
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

// aggregateGateway reads a non-stream gateway response and returns text, tool
// calls, finish reason and usage.
func aggregateGateway(resp *http.Response) (text, model string, toolCalls []gwToolCall, finish string, usage map[string]any, err error) {
	defer resp.Body.Close()
	var doc gwFullDoc
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		return "", "", nil, "", nil, err
	}
	if len(doc.Choices) == 0 {
		return "", doc.Model, nil, "", doc.Usage, fmt.Errorf("gateway returned no choices")
	}
	ch := doc.Choices[0]
	for i, tc := range ch.Message.ToolCalls {
		callID := tc.ID
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), i)
		}
		toolCalls = append(toolCalls, gwToolCall{ID: callID, Name: tc.Function.Name, Arguments: toolArgs(tc.Function.Arguments)})
	}
	return ch.Message.Content, doc.Model, toolCalls, ch.FinishReason, doc.Usage, nil
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
	finish := ""
	// next: index of the next content block; textIdx: open text block or -1;
	// tools: tool calls in first-seen order; byIdx: latest tool call per upstream
	// index; cur: tool call whose block is open; pending: text that arrived once a
	// tool call had begun.
	next := 0
	textIdx := -1
	produced := false
	var tools []*streamTool
	byIdx := map[int]*streamTool{}
	var cur *streamTool
	var pending strings.Builder
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
	}
	openBlock := func(block map[string]any) int {
		start()
		i := next
		next++
		produced = true
		send("content_block_start", map[string]any{"type": "content_block_start", "index": i, "content_block": block})
		return i
	}
	closeBlock := func(i int) {
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": i})
	}
	delta := func(i int, d map[string]any) {
		send("content_block_delta", map[string]any{"type": "content_block_delta", "index": i, "delta": d})
	}
	writeText := func(s string) {
		if textIdx < 0 {
			textIdx = openBlock(map[string]any{"type": "text", "text": ""})
		}
		delta(textIdx, map[string]any{"type": "text_delta", "text": s})
	}
	closeText := func() {
		if textIdx >= 0 {
			closeBlock(textIdx)
			textIdx = -1
		}
	}
	// pump emits tool_use blocks one at a time. The open tool call ends once its
	// arguments form valid JSON (or at end of stream); later tool calls stay
	// buffered until then.
	pump := func(final bool) {
		for {
			if cur != nil {
				if args := cur.args.String(); cur.sent < len(args) {
					delta(cur.index, map[string]any{"type": "input_json_delta", "partial_json": args[cur.sent:]})
					cur.sent = len(args)
				}
				if args := cur.args.String(); !final && (!strings.HasSuffix(args, "}") || !json.Valid([]byte(args))) {
					return
				}
				if cur.sent == 0 {
					delta(cur.index, map[string]any{"type": "input_json_delta", "partial_json": "{}"})
				}
				closeBlock(cur.index)
				cur = nil
			}
			var nxt *streamTool
			for _, t := range tools {
				if !t.opened {
					nxt = t
					break
				}
			}
			if nxt == nil || (!final && nxt.name == "") {
				return
			}
			closeText()
			if nxt.id == "" {
				nxt.id = fmt.Sprintf("toolu_%d_%d", time.Now().UnixNano(), next)
			}
			nxt.opened = true
			nxt.index = openBlock(map[string]any{"type": "tool_use", "id": nxt.id, "name": nxt.name, "input": map[string]any{}})
			cur = nxt
		}
	}
	fail := func(msg string) {
		send("error", map[string]any{"type": "error", "error": map[string]any{"type": "api_error", "message": msg}})
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
		var ev gwChunk
		if json.Unmarshal([]byte(payload), &ev) != nil {
			continue
		}
		if len(ev.Error) > 0 && string(ev.Error) != "null" {
			fail(gwErrorMessage(ev.Error))
			return
		}
		if ev.Model != "" {
			model = ev.Model
		}
		if ev.Usage != nil {
			usage = ev.Usage
		}
		text := ""
		for _, ch := range ev.Choices {
			text += ch.Delta.Content
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			for _, tc := range ch.Delta.ToolCalls {
				t := byIdx[tc.Index]
				if t == nil || (tc.ID != "" && t.id != "" && tc.ID != t.id) {
					t = &streamTool{}
					byIdx[tc.Index] = t
					tools = append(tools, t)
				}
				if tc.ID != "" {
					t.id = tc.ID
				}
				if t.name == "" {
					t.name = tc.Function.Name
				}
				t.args.WriteString(rawString(tc.Function.Arguments))
			}
		}
		if text != "" {
			if len(tools) > 0 {
				pending.WriteString(text)
			} else {
				writeText(text)
			}
		}
		pump(false)
	}
	if err := sc.Err(); err != nil {
		fail(err.Error())
		return
	}
	if finish == "" { // cut off: a half-written tool call must not look complete
		fail("the gateway stream ended early")
		return
	}
	pump(true)
	if pending.Len() > 0 {
		writeText(pending.String())
	}
	closeText()
	if !produced {
		closeBlock(openBlock(map[string]any{"type": "text", "text": ""}))
	}
	in := toInt(usage["prompt_tokens"])
	out := toInt(usage["completion_tokens"])
	cached := 0
	if d, ok := usage["prompt_tokens_details"].(map[string]any); ok {
		cached = toInt(d["cached_tokens"])
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": anthropicStopReason(finish, len(tools) > 0), "stop_sequence": nil},
		"usage": map[string]any{"input_tokens": in, "output_tokens": out, "cache_read_input_tokens": cached},
	})
	send("message_stop", map[string]any{"type": "message_stop"})
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
	text, gwModel, calls, finish, usage, err := aggregateGateway(resp)
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
	content := make([]map[string]any, 0, len(calls)+1)
	if text != "" || len(calls) == 0 {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	for _, c := range calls {
		var input map[string]any
		if json.Unmarshal([]byte(c.Arguments), &input) != nil || input == nil {
			input = map[string]any{}
		}
		content = append(content, map[string]any{"type": "tool_use", "id": c.ID, "name": c.Name, "input": input})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": fmt.Sprintf("msg_%d", time.Now().UnixNano()), "type": "message",
		"role": "assistant", "model": model,
		"content":     content,
		"stop_reason": anthropicStopReason(finish, len(calls) > 0), "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": in, "output_tokens": out, "cache_read_input_tokens": cached},
	})
}

var _ = io.Discard
