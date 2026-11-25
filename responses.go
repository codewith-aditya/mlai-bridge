package main

// responses.go — OpenAI Responses API (POST /v1/responses) on top of the gateway
// chat-completions path, with the CLI as fallback. Built for the Codex CLI.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// ---------- request types ----------

type respReasoning struct {
	Effort string `json:"effort"`
}

type respFormat struct {
	Type       string `json:"type"`
	Syntax     string `json:"syntax"`
	Definition string `json:"definition"`
}

type respTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Format      *respFormat     `json:"format"`
	Tools       []respTool      `json:"tools"`
}

type respInputItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Namespace string          `json:"namespace"`
	Arguments json.RawMessage `json:"arguments"`
	Input     json.RawMessage `json:"input"`
	Output    json.RawMessage `json:"output"`
}

type respPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	ImageURL json.RawMessage `json:"image_url"`
}

type responsesRequest struct {
	Model              string          `json:"model"`
	Instructions       string          `json:"instructions"`
	Input              json.RawMessage `json:"input"`
	Tools              []respTool      `json:"tools"`
	ToolChoice         json.RawMessage `json:"tool_choice"`
	Stream             bool            `json:"stream"`
	MaxOutputTokens    int             `json:"max_output_tokens"`
	Temperature        *float64        `json:"temperature"`
	Reasoning          *respReasoning  `json:"reasoning"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls"`
	PreviousResponseID string          `json:"previous_response_id"`
}

// respToolRef maps a flattened chat function name back to the Responses tool.
type respToolRef struct {
	ns, name string
	custom   bool
}

// ---------- request conversion ----------

func rawJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// respImageURL reads an input_image image_url (string or {url}).
func respImageURL(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var o struct {
		URL string `json:"url"`
	}
	if json.Unmarshal(raw, &o) == nil {
		return o.URL
	}
	return ""
}

// respParts flattens string or part-array content into text, plus chat content
// parts when images are kept.
func respParts(raw json.RawMessage, images bool) (string, []map[string]any, bool) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil, false
	}
	var ps []respPart
	if json.Unmarshal(raw, &ps) != nil {
		if t := strings.TrimSpace(string(raw)); t != "" && t != "null" {
			return string(raw), nil, false
		}
		return "", nil, false
	}
	var texts []string
	var parts []map[string]any
	hasImage := false
	for _, p := range ps {
		switch p.Type {
		case "input_text", "output_text", "text":
			if p.Text != "" {
				texts = append(texts, p.Text)
				parts = append(parts, map[string]any{"type": "text", "text": p.Text})
			}
		case "input_image":
			if u := respImageURL(p.ImageURL); u != "" && images {
				hasImage = true
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": u}})
			}
		}
	}
	return strings.Join(texts, "\n"), parts, hasImage
}

// responsesMessages converts instructions and input into chat messages.
func responsesMessages(instructions string, input json.RawMessage) []chatMessage {
	var msgs []chatMessage
	if strings.TrimSpace(instructions) != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: rawJSON(instructions)})
	}
	var text string
	if json.Unmarshal(input, &text) == nil {
		if text != "" {
			msgs = append(msgs, chatMessage{Role: "user", Content: rawJSON(text)})
		}
		return msgs
	}
	var items []respInputItem
	if json.Unmarshal(input, &items) != nil {
		return msgs
	}
	callsAt := -1
	var pending []map[string]any
	for _, it := range items {
		switch it.Type {
		case "message", "":
			if it.Role == "" {
				continue
			}
			callsAt, pending = -1, nil
			role := it.Role
			if role == "developer" {
				role = "system"
			}
			text, parts, hasImage := respParts(it.Content, role == "user")
			if hasImage {
				msgs = append(msgs, chatMessage{Role: role, Content: rawJSON(parts)})
			} else if text != "" {
				msgs = append(msgs, chatMessage{Role: role, Content: rawJSON(text)})
				if role == "assistant" { // its tool calls belong to the same message
					callsAt = len(msgs) - 1
				}
			}
		case "function_call", "custom_tool_call":
			name := it.Name
			if it.Namespace != "" {
				name = it.Namespace + "__" + name
			}
			var args string
			if it.Type == "custom_tool_call" {
				args = string(rawJSON(map[string]string{"input": rawString(it.Input)}))
			} else {
				args = toolArgs(it.Arguments)
			}
			if callsAt < 0 {
				pending = nil
				msgs = append(msgs, chatMessage{Role: "assistant"})
				callsAt = len(msgs) - 1
			}
			pending = append(pending, map[string]any{"id": it.CallID, "type": "function", "function": map[string]any{"name": name, "arguments": args}})
			msgs[callsAt].ToolCalls = rawJSON(pending)
		case "function_call_output", "custom_tool_call_output":
			callsAt = -1
			out, _, _ := respParts(it.Output, false)
			msgs = append(msgs, chatMessage{Role: "tool", Content: rawJSON(out), ToolCallID: it.CallID})
		}
	}
	return msgs
}

const respCustomParams = `{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}`

// responsesTools flattens Responses tools into chat tool definitions. Namespace
// tools become "<namespace>__<tool>"; unsupported tool types are dropped.
func responsesTools(tools []respTool, ns string, defs []toolDef, refs map[string]respToolRef) []toolDef {
	for _, t := range tools {
		switch t.Type {
		case "function", "custom":
			if t.Name == "" {
				continue
			}
			full := t.Name
			if ns != "" {
				full = ns + "__" + t.Name
			}
			def := toolDef{Type: "function"}
			def.Function.Name = full
			def.Function.Description = t.Description
			if t.Type == "custom" {
				if t.Format != nil && strings.TrimSpace(t.Format.Definition) != "" {
					def.Function.Description = strings.TrimSpace(t.Description + "\n\n" + t.Format.Definition)
				}
				def.Function.Parameters = json.RawMessage(respCustomParams)
			} else if len(t.Parameters) > 0 && string(t.Parameters) != "null" {
				def.Function.Parameters = t.Parameters
			}
			refs[full] = respToolRef{ns: ns, name: t.Name, custom: t.Type == "custom"}
			defs = append(defs, def)
		case "namespace":
			if ns == "" && t.Name != "" {
				defs = responsesTools(t.Tools, t.Name, defs, refs)
			}
		}
	}
	return defs
}

// responsesToolChoice maps a Responses tool_choice to the chat one (nil if unset).
func responsesToolChoice(raw json.RawMessage) any {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "auto", "none", "required":
			return s
		}
		return nil
	}
	var o struct {
		Type      string `json:"type"`
		Name      string `json:"name"`
		Namespace string `json:"namespace"`
	}
	if json.Unmarshal(raw, &o) != nil || o.Name == "" || (o.Type != "function" && o.Type != "custom") {
		return nil
	}
	name := o.Name
	if o.Namespace != "" {
		name = o.Namespace + "__" + name
	}
	return map[string]any{"type": "function", "function": map[string]any{"name": name}}
}

// ---------- event emitter ----------

// respEmitter builds the Responses output once and either streams it as SSE or,
// for non-stream requests, writes the final response as one JSON document.
type respEmitter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	reqID   string
	stream  bool
	refs    map[string]respToolRef

	id      string
	model   string
	created int64
	seq     int
	n       int
	started bool
	output  []any

	textOpen bool
	textIdx  int
	msgID    string
	text     strings.Builder
}

func (e *respEmitter) newID(prefix string) string {
	e.n++
	return fmt.Sprintf("%s_%d_%d", prefix, time.Now().UnixNano(), e.n)
}

func (e *respEmitter) send(typ string, data map[string]any) {
	data["type"] = typ
	data["sequence_number"] = e.seq
	e.seq++
	if !e.stream {
		return
	}
	b, _ := json.Marshal(data)
	fmt.Fprintf(e.w, "event: %s\ndata: %s\n\n", typ, b)
	if e.flusher != nil {
		e.flusher.Flush()
	}
}

func (e *respEmitter) response(status string) map[string]any {
	out := e.output
	if out == nil {
		out = []any{}
	}
	return map[string]any{"id": e.id, "object": "response", "created_at": e.created, "status": status, "model": e.model, "output": out}
}

func (e *respEmitter) begin() {
	if e.started {
		return
	}
	e.started = true
	if e.stream {
		e.w.Header().Set("Content-Type", "text/event-stream")
		e.w.Header().Set("Cache-Control", "no-cache")
		e.w.WriteHeader(200)
	}
	e.send("response.created", map[string]any{"response": e.response("in_progress")})
	e.send("response.in_progress", map[string]any{"response": e.response("in_progress")})
}

func (e *respEmitter) openText() {
	if e.textOpen {
		return
	}
	e.textOpen = true
	e.msgID = e.newID("msg")
	e.textIdx = len(e.output)
	item := map[string]any{"id": e.msgID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{}}
	e.send("response.output_item.added", map[string]any{"output_index": e.textIdx, "item": item})
	part := map[string]any{"type": "output_text", "text": "", "annotations": []any{}}
	e.send("response.content_part.added", map[string]any{"item_id": e.msgID, "output_index": e.textIdx, "content_index": 0, "part": part})
}

func (e *respEmitter) writeText(delta string) {
	if delta == "" {
		return
	}
	e.openText()
	e.text.WriteString(delta)
	e.send("response.output_text.delta", map[string]any{"item_id": e.msgID, "output_index": e.textIdx, "content_index": 0, "delta": delta, "logprobs": []any{}})
}

func (e *respEmitter) closeText() {
	if !e.textOpen {
		return
	}
	e.textOpen = false
	text := e.text.String()
	e.text.Reset()
	e.send("response.output_text.done", map[string]any{"item_id": e.msgID, "output_index": e.textIdx, "content_index": 0, "text": text, "logprobs": []any{}})
	part := map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
	e.send("response.content_part.done", map[string]any{"item_id": e.msgID, "output_index": e.textIdx, "content_index": 0, "part": part})
	item := map[string]any{"id": e.msgID, "type": "message", "status": "completed", "role": "assistant", "content": []any{part}}
	e.send("response.output_item.done", map[string]any{"output_index": e.textIdx, "item": item})
	e.output = append(e.output, item)
}

// tool emits one complete tool call. args is the chat function arguments JSON.
func (e *respEmitter) tool(callID, name, args string) {
	ref, ok := e.refs[name]
	if !ok {
		ref = respToolRef{name: name}
	}
	if callID == "" {
		callID = e.newID("call")
	}
	args = strings.TrimSpace(args)
	if args == "" {
		args = "{}"
	}
	idx := len(e.output)
	mk := func(id, typ string) map[string]any {
		item := map[string]any{"id": id, "type": typ, "call_id": callID, "name": ref.name}
		if ref.ns != "" {
			item["namespace"] = ref.ns
		}
		return item
	}
	if ref.custom {
		input := args
		var v struct {
			Input *string `json:"input"`
		}
		if json.Unmarshal([]byte(args), &v) == nil && v.Input != nil {
			input = *v.Input
		}
		id := e.newID("ctc")
		added := mk(id, "custom_tool_call")
		added["input"] = ""
		e.send("response.output_item.added", map[string]any{"output_index": idx, "item": added})
		if input != "" {
			e.send("response.custom_tool_call_input.delta", map[string]any{"item_id": id, "output_index": idx, "delta": input})
		}
		e.send("response.custom_tool_call_input.done", map[string]any{"item_id": id, "output_index": idx, "input": input})
		item := mk(id, "custom_tool_call")
		item["input"] = input
		e.send("response.output_item.done", map[string]any{"output_index": idx, "item": item})
		e.output = append(e.output, item)
		return
	}
	id := e.newID("fc")
	added := mk(id, "function_call")
	added["arguments"] = ""
	added["status"] = "in_progress"
	e.send("response.output_item.added", map[string]any{"output_index": idx, "item": added})
	e.send("response.function_call_arguments.delta", map[string]any{"item_id": id, "output_index": idx, "delta": args})
	e.send("response.function_call_arguments.done", map[string]any{"item_id": id, "output_index": idx, "arguments": args})
	item := mk(id, "function_call")
	item["arguments"] = args
	item["status"] = "completed"
	e.send("response.output_item.done", map[string]any{"output_index": idx, "item": item})
	e.output = append(e.output, item)
}

func respInt(v any) int {
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

// respUsage converts chat usage (prompt_tokens, ...) to Responses usage.
func respUsage(u map[string]any) map[string]any {
	in := respInt(u["prompt_tokens"])
	out := respInt(u["completion_tokens"])
	total := respInt(u["total_tokens"])
	if total == 0 {
		total = in + out
	}
	cached, reasoning := 0, 0
	if d, ok := u["prompt_tokens_details"].(map[string]any); ok {
		cached = respInt(d["cached_tokens"])
	}
	if d, ok := u["completion_tokens_details"].(map[string]any); ok {
		reasoning = respInt(d["reasoning_tokens"])
	}
	return map[string]any{
		"input_tokens":          in,
		"input_tokens_details":  map[string]any{"cached_tokens": cached},
		"output_tokens":         out,
		"output_tokens_details": map[string]any{"reasoning_tokens": reasoning},
		"total_tokens":          total,
	}
}

// finish closes the response: completed, or incomplete when finish is "length".
func (e *respEmitter) finish(finish string, usage map[string]any) {
	e.closeText()
	if len(e.output) == 0 {
		e.openText()
		e.closeText()
	}
	resp := e.response("completed")
	evt := "response.completed"
	if finish == "length" {
		resp["status"] = "incomplete"
		resp["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
		evt = "response.incomplete"
	}
	resp["usage"] = respUsage(usage)
	if !e.stream {
		e.w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(e.w).Encode(resp)
		return
	}
	e.send(evt, map[string]any{"response": resp})
}

// fail ends the response with an error: response.failed when streaming, an
// OpenAI error body otherwise.
func (e *respEmitter) fail(code, msg string) {
	if !e.stream {
		writeOpenAIError(e.w, apiError{502, "server_error", code, "api_error", msg}, e.reqID)
		return
	}
	resp := e.response("failed")
	resp["error"] = map[string]any{"code": code, "message": msg}
	e.send("response.failed", map[string]any{"response": resp})
}

// respStreamTool accumulates one streamed chat tool call.
type respStreamTool struct {
	id, name string
	args     strings.Builder
}

// fromGateway reads the gateway's chat-completions SSE and emits the response.
func (e *respEmitter) fromGateway(resp *http.Response) {
	defer resp.Body.Close()
	e.begin()
	finish := ""
	usage := map[string]any{}
	var tools []*respStreamTool
	byIdx := map[int]*respStreamTool{}
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
			e.fail("upstream_error", gwErrorMessage(ev.Error))
			return
		}
		if ev.Model != "" {
			e.model = ev.Model
		}
		if ev.Usage != nil {
			usage = ev.Usage
		}
		for _, ch := range ev.Choices {
			e.writeText(ch.Delta.Content)
			if ch.FinishReason != "" {
				finish = ch.FinishReason
			}
			for _, tc := range ch.Delta.ToolCalls {
				t := byIdx[tc.Index]
				if t == nil || (tc.ID != "" && t.id != "" && tc.ID != t.id) {
					t = &respStreamTool{}
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
	}
	if err := sc.Err(); err != nil {
		e.fail("upstream_error", err.Error())
		return
	}
	if finish == "" {
		e.fail("upstream_error", "upstream stream ended without a finish reason")
		return
	}
	e.closeText()
	for _, t := range tools {
		if t.name != "" {
			e.tool(t.id, t.name, t.args.String())
		}
	}
	e.finish(finish, usage)
}

// ---------- handler ----------

func (s *server) handleResponses(w http.ResponseWriter, r *http.Request) {
	reqID := w.Header().Get("X-Request-Id")
	if r.Method != http.MethodPost {
		e := errBadRequest
		e.Message = "use POST"
		writeOpenAIError(w, e, reqID)
		return
	}
	var req responsesRequest
	if err := decodeBody(w, r, &req); err != nil {
		e := errBadRequest
		e.Message = "invalid JSON: " + err.Error()
		writeOpenAIError(w, e, reqID)
		return
	}
	if req.PreviousResponseID != "" {
		e := errBadRequest
		e.Message = "previous_response_id is not supported: this bridge is stateless, send the full input"
		writeOpenAIError(w, e, reqID)
		return
	}
	model := req.Model
	if model == "" {
		model = cfg.DefaultModel
	}
	msgs := responsesMessages(req.Instructions, req.Input)
	hasConvo := false
	for _, m := range msgs {
		if m.Role != "system" {
			hasConvo = true
		}
	}
	if !hasConvo {
		e := errBadRequest
		e.Message = "input is required"
		writeOpenAIError(w, e, reqID)
		return
	}
	refs := map[string]respToolRef{}
	defs := responsesTools(req.Tools, "", nil, refs)
	effort := cfg.Effort
	if req.Reasoning != nil && req.Reasoning.Effort != "" {
		effort = req.Reasoning.Effort
	}
	flusher, _ := w.(http.Flusher)
	em := &respEmitter{
		w: w, flusher: flusher, reqID: reqID, stream: req.Stream, refs: refs,
		id: fmt.Sprintf("resp_%d", time.Now().UnixNano()), model: model, created: time.Now().Unix(),
	}

	// Fast path: gateway direct. It is always streamed; non-stream clients get
	// the aggregated response.
	if s.direct {
		body := map[string]any{
			"model":          model,
			"messages":       msgs,
			"stream":         true,
			"stream_options": map[string]any{"include_usage": true},
		}
		if tools := cleanTools(defs, false); len(tools) > 0 {
			body["tools"] = tools
			if tc := responsesToolChoice(req.ToolChoice); tc != nil {
				body["tool_choice"] = tc
			}
			if req.ParallelToolCalls != nil {
				body["parallel_tool_calls"] = *req.ParallelToolCalls
			}
		}
		if effort != "" {
			body["reasoning_effort"] = effort
		}
		if req.Temperature != nil {
			body["temperature"] = *req.Temperature
		}
		if req.MaxOutputTokens > 0 {
			body["max_tokens"] = req.MaxOutputTokens
		}
		raw, _ := json.Marshal(body)
		done, derr := s.serveDirect(r.Context(), raw, true, func(resp *http.Response) bool {
			em.fromGateway(resp)
			return true
		})
		if derr != nil {
			writeOpenAIError(w, *derr, reqID)
			return
		}
		if done {
			return
		}
	}

	var system string
	var convo []chatMessage
	for _, m := range msgs {
		if m.Role == "system" {
			if system != "" {
				system += "\n"
			}
			system += m.text()
			continue
		}
		convo = append(convo, m)
	}
	res := s.execute(r.Context(), model, system, convo, defs, effort, req.MaxOutputTokens)
	if res.err != nil {
		writeOpenAIError(w, *res.err, reqID)
		return
	}
	a := res.answer
	if a.Model != "" {
		em.model = a.Model
	}
	em.begin()
	em.writeText(a.Text)
	em.closeText()
	for _, tc := range a.Tools {
		em.tool("", tc.Name, toolArgs(tc.Arguments))
	}
	em.finish("stop", map[string]any{
		"prompt_tokens":         a.Usage.PromptTokens,
		"completion_tokens":     a.Usage.CompletionTokens,
		"total_tokens":          a.Usage.TotalTokens,
		"prompt_tokens_details": map[string]any{"cached_tokens": a.Usage.CachedTokens},
	})
}
