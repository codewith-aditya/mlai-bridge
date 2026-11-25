package main

// agents_test.go: what Claude Code and Codex need from the bridge (tool use on
// /v1/messages and /v1/responses, direct mode).

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

const toolStream = `data: {"choices":[{"delta":{"content":"hi "}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"run","arguments":"{\"a\":"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","function":{"name":"read","arguments":"{}"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}}]}}]}

`

const toolStreamEnd = `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}

data: [DONE]

`

// gatewayStream answers every gateway call with body and records the request.
func gatewayStream(t *testing.T, body string) (http.Handler, *map[string]any) {
	t.Helper()
	setupStub(t)
	s, _ := directServer(t)
	sent := map[string]any{}
	setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
			t.Error(err)
		}
		return fakeResp(r, 200, body), nil
	}))
	return newHTTPServer(s).Handler, &sent
}

func asJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

const anthropicToolBody = `{"model":"m","stream":true,"max_tokens":9,"system":[{"type":"text","text":"sys"}],
"tools":[{"name":"run","description":"d","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}],
"tool_choice":{"type":"any"},
"messages":[{"role":"user","content":"go"},
{"role":"assistant","content":[{"type":"thinking","thinking":"x"},{"type":"tool_use","id":"t1","name":"run","input":{"a":1}}]},
{"role":"user","content":[{"type":"text","text":"next"},{"type":"tool_result","tool_use_id":"t1","is_error":true,"content":[{"type":"text","text":"boom"}]}]}]}`

func TestAnthropicToolsDirect(t *testing.T) {
	h, sent := gatewayStream(t, toolStream+toolStreamEnd)
	out := serve(h, "POST", "/v1/messages", "", jsonHeader, strings.NewReader(anthropicToolBody)).Body.String()

	msgs := asJSON((*sent)["messages"])
	for _, want := range []string{`"content":"sys"`, `"arguments":"{\"a\":1}"`, `"tool_call_id":"t1"`, `Error: boom`} {
		if !strings.Contains(msgs, want) {
			t.Errorf("gateway messages lack %s: %s", want, msgs)
		}
	}
	if strings.Index(msgs, `"tool_call_id"`) > strings.Index(msgs, `"next"`) {
		t.Error("the tool result must come before the new user text")
	}
	if n := len((*sent)["tools"].([]any)); n != 1 {
		t.Errorf("gateway got %d tools, want 1", n)
	}
	if (*sent)["tool_choice"] != "required" {
		t.Errorf("tool_choice = %v", (*sent)["tool_choice"])
	}

	// Rebuild the blocks the way a client does.
	open := -1
	name := map[int]string{}
	args := map[int]string{}
	for _, line := range strings.Split(out, "\n") {
		var ev struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
			Block struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta struct {
				PartialJSON string `json:"partial_json"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
		}
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || json.Unmarshal([]byte(data), &ev) != nil {
			continue
		}
		switch ev.Type {
		case "content_block_start":
			if open != -1 {
				t.Error("two blocks open at once")
			}
			open = ev.Index
			name[open] = ev.Block.Type + ev.Block.Name
		case "content_block_delta":
			args[ev.Index] += ev.Delta.PartialJSON
		case "content_block_stop":
			open = -1
		case "message_delta":
			if ev.Delta.StopReason != "tool_use" {
				t.Errorf("stop_reason = %q", ev.Delta.StopReason)
			}
		}
	}
	if got := fmt.Sprint(name); got != "map[0:text 1:tool_userun 2:tool_useread]" {
		t.Errorf("blocks = %s", got)
	}
	if args[1] != `{"a":1}` || args[2] != `{}` {
		t.Errorf("arguments = %v", args)
	}
	if !strings.Contains(out, "event: message_stop") {
		t.Error("no message_stop")
	}
}

func TestCutStreamIsAnError(t *testing.T) {
	h, _ := gatewayStream(t, toolStream)
	out := serve(h, "POST", "/v1/messages", "", jsonHeader, strings.NewReader(anthropicToolBody)).Body.String()
	if !strings.Contains(out, "event: error") || strings.Contains(out, "message_stop") {
		t.Errorf("anthropic: %s", out)
	}
	out = serve(h, "POST", "/v1/responses", "", jsonHeader, strings.NewReader(responsesBody)).Body.String()
	if !strings.Contains(out, "event: response.failed") || strings.Contains(out, "function_call") {
		t.Errorf("responses: %s", out)
	}
}

const responsesBody = `{"model":"m","stream":true,"instructions":"sys",
"tools":[{"type":"function","name":"run","parameters":{"type":"object"}},{"type":"custom","name":"read"},{"type":"web_search"}],
"input":[{"type":"message","role":"developer","content":[{"type":"input_text","text":"dev"}]},
{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]},
{"type":"function_call","call_id":"f1","name":"run","arguments":"{}"},
{"type":"function_call_output","call_id":"f1","output":"result"}]}`

func TestResponsesDirect(t *testing.T) {
	h, sent := gatewayStream(t, toolStream+toolStreamEnd)
	out := serve(h, "POST", "/v1/responses", "", jsonHeader, strings.NewReader(responsesBody)).Body.String()

	var roles []string
	for _, m := range (*sent)["messages"].([]any) {
		roles = append(roles, m.(map[string]any)["role"].(string))
	}
	if got := strings.Join(roles, " "); got != "system system assistant tool" {
		t.Errorf("roles = %s", got)
	}
	if n := len((*sent)["tools"].([]any)); n != 2 {
		t.Errorf("gateway got %d tools, want 2", n)
	}
	for _, want := range []string{"event: response.created", "event: response.output_text.delta", `"arguments":"{\"a\":1}"`,
		`"type":"custom_tool_call"`, "event: response.completed", `"input_tokens":7`} {
		if !strings.Contains(out, want) {
			t.Errorf("stream lacks %s:\n%s", want, out)
		}
	}
}

func TestResponsesFallsBackToCLI(t *testing.T) {
	setupStub(t)
	s, _ := directServer(t)
	setTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) { return fakeResp(r, 503, ""), nil }))
	rec := serve(newHTTPServer(s).Handler, "POST", "/v1/responses", "", jsonHeader, strings.NewReader(`{"model":"m","input":"hello"}`))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "stub:hello") {
		t.Errorf("got %d %s", rec.Code, rec.Body)
	}
}

func TestNoProxyCoversTheBridge(t *testing.T) {
	env := planLaunch("claude", "http://127.0.0.1:1", "k", "m").environ([]string{"NO_PROXY=corp.example"})
	if got := strings.Join(env, "\n"); !strings.Contains(got, "NO_PROXY=corp.example,127.0.0.1,localhost,::1") {
		t.Error(got)
	}
}
