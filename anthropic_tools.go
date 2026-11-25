package main

// anthropic_tools.go — Anthropic <-> OpenAI message and tool translation for the
// gateway direct path.

import (
	"encoding/json"
	"strings"
)

// anthropicBlock is one Anthropic content block (text, image, tool_use, tool_result, ...).
type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	Source    json.RawMessage `json:"source"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

// gwToolCall is a complete tool call from a non-stream gateway reply.
type gwToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// gateway wire shapes (OpenAI chat completions).
type gwFunction struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type gwCallDelta struct {
	Index    int        `json:"index"`
	ID       string     `json:"id"`
	Function gwFunction `json:"function"`
}

type gwChoiceDelta struct {
	Content          string        `json:"content"`
	ReasoningContent string        `json:"reasoning_content"`
	ToolCalls        []gwCallDelta `json:"tool_calls"`
}

type gwChunkChoice struct {
	Delta        gwChoiceDelta `json:"delta"`
	FinishReason string        `json:"finish_reason"`
}

type gwChunk struct {
	Model   string          `json:"model"`
	Error   json.RawMessage `json:"error"`
	Choices []gwChunkChoice `json:"choices"`
	Usage   map[string]any  `json:"usage"`
}

type gwFullChoice struct {
	Message      gwChoiceDelta `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type gwFullDoc struct {
	Model   string         `json:"model"`
	Choices []gwFullChoice `json:"choices"`
	Usage   map[string]any `json:"usage"`
}

// streamTool accumulates one streamed tool call.
type streamTool struct {
	id, name string
	args     strings.Builder
	index    int
	sent     int
	opened   bool
}

// rawString returns a JSON string's value, or the raw text for anything else.
func rawString(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	return string(raw)
}

// toolArgs returns tool arguments as a JSON text, "{}" when empty.
func toolArgs(raw json.RawMessage) string {
	s := strings.TrimSpace(rawString(raw))
	if s == "" || s == "null" {
		return "{}"
	}
	return s
}

// anthropicText flattens a string or an array of text blocks.
func anthropicText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []anthropicBlock
	if json.Unmarshal(raw, &blocks) == nil {
		texts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if b.Text != "" {
				texts = append(texts, b.Text)
			}
		}
		return strings.Join(texts, "\n")
	}
	return string(raw)
}

// anthropicImageURL turns an Anthropic image source into an image URL.
func anthropicImageURL(raw json.RawMessage) string {
	var src struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	}
	if json.Unmarshal(raw, &src) != nil {
		return ""
	}
	switch src.Type {
	case "base64":
		if src.Data == "" {
			return ""
		}
		mt := src.MediaType
		if mt == "" {
			mt = "image/png"
		}
		return "data:" + mt + ";base64," + src.Data
	case "url":
		return src.URL
	}
	return ""
}

// anthropicMessages converts Anthropic messages to OpenAI chat messages.
func anthropicMessages(msgs []chatMessage) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		var blocks []anthropicBlock
		if len(m.Content) == 0 || json.Unmarshal(m.Content, &blocks) != nil {
			// plain string content
			out = append(out, map[string]any{"role": m.Role, "content": m.text()})
			continue
		}
		switch m.Role {
		case "system":
			out = append(out, map[string]any{"role": "system", "content": anthropicText(m.Content)})
		case "assistant":
			var texts []string
			var calls []map[string]any
			for _, b := range blocks {
				switch b.Type {
				case "text":
					if b.Text != "" {
						texts = append(texts, b.Text)
					}
				case "tool_use":
					calls = append(calls, map[string]any{
						"id": b.ID, "type": "function",
						"function": map[string]any{"name": b.Name, "arguments": toolArgs(b.Input)},
					})
				}
			}
			if len(texts) == 0 && len(calls) == 0 {
				continue
			}
			msg := map[string]any{"role": "assistant", "content": nil}
			if len(texts) > 0 {
				msg["content"] = strings.Join(texts, "\n")
			}
			if len(calls) > 0 {
				msg["tool_calls"] = calls
			}
			out = append(out, msg)
		default:
			var toolMsgs []map[string]any
			var parts []map[string]any
			var texts []string
			hasImage := false
			for _, b := range blocks {
				switch b.Type {
				case "text":
					if b.Text != "" {
						texts = append(texts, b.Text)
						parts = append(parts, map[string]any{"type": "text", "text": b.Text})
					}
				case "image":
					if u := anthropicImageURL(b.Source); u != "" {
						hasImage = true
						parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": u}})
					}
				case "tool_result":
					c := anthropicText(b.Content)
					if b.IsError {
						c = "Error: " + c
					}
					toolMsgs = append(toolMsgs, map[string]any{"role": "tool", "tool_call_id": b.ToolUseID, "content": c})
				}
			}
			out = append(out, toolMsgs...)
			if len(parts) == 0 {
				continue
			}
			if hasImage {
				out = append(out, map[string]any{"role": "user", "content": parts})
			} else {
				out = append(out, map[string]any{"role": "user", "content": strings.Join(texts, "\n")})
			}
		}
	}
	return out
}

// cleanTools builds OpenAI {type,function{name,description,parameters}} tools from
// either request shape. With anthropic set, server tools (a "type" other than
// "" or "custom") are skipped.
func cleanTools(tools []toolDef, anthropic bool) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if anthropic && t.Type != "" && t.Type != "custom" {
			continue
		}
		name := t.toolName()
		if name == "" {
			continue
		}
		params := t.toolSchema()
		if len(params) == 0 || string(params) == "null" {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name, "description": t.toolDesc(), "parameters": params},
		})
	}
	return out
}

// anthropicToolChoice maps an Anthropic tool_choice to the OpenAI one (nil if unset).
func anthropicToolChoice(raw json.RawMessage) any {
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &tc) != nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "none":
		return "none"
	case "tool":
		if tc.Name != "" {
			return map[string]any{"type": "function", "function": map[string]any{"name": tc.Name}}
		}
	}
	return nil
}

// anthropicStopReason maps an OpenAI finish_reason to an Anthropic stop_reason.
func anthropicStopReason(finish string, hasTools bool) string {
	switch finish {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	}
	if hasTools && (finish == "stop" || finish == "") {
		return "tool_use"
	}
	return "end_turn"
}

// gwErrorMessage extracts a message from an upstream error object or string.
func gwErrorMessage(raw json.RawMessage) string {
	var e struct {
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &e) == nil && e.Message != "" {
		return e.Message
	}
	if s := rawString(raw); s != "" {
		return s
	}
	return "upstream error"
}
