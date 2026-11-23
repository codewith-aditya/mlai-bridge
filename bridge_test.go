package main

import (
	"encoding/json"
	"testing"
	"time"
)

func TestParseToolCalls(t *testing.T) {
	in := `{"tool_calls":[{"name":"get_weather","arguments":{"city":"Pune"}}]}`
	calls, preamble := parseToolCalls(in)
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	if calls[0].Name != "get_weather" {
		t.Fatalf("name = %q", calls[0].Name)
	}
	var args map[string]string
	if err := json.Unmarshal(calls[0].Arguments, &args); err != nil {
		t.Fatalf("arguments: %v", err)
	}
	if args["city"] != "Pune" {
		t.Fatalf("city = %q", args["city"])
	}
	if preamble != "" {
		t.Fatalf("preamble = %q", preamble)
	}
}

func TestParseToolCallsWithPreamble(t *testing.T) {
	in := "Let me check.\n{\"tool_calls\":[{\"name\":\"x\",\"arguments\":{}}]}"
	calls, preamble := parseToolCalls(in)
	if len(calls) != 1 || preamble != "Let me check." {
		t.Fatalf("calls=%d preamble=%q", len(calls), preamble)
	}
}

func TestParseToolCallsPlainText(t *testing.T) {
	calls, preamble := parseToolCalls("just an answer")
	if calls != nil {
		t.Fatalf("expected nil calls")
	}
	if preamble != "just an answer" {
		t.Fatalf("preamble = %q", preamble)
	}
}

func TestCacheTTL(t *testing.T) {
	c := NewLRU(2, 50*time.Millisecond)
	c.Put("a", []byte("1"))
	if v, ok := c.Get("a"); !ok || string(v) != "1" {
		t.Fatalf("expected hit")
	}
	time.Sleep(80 * time.Millisecond)
	if _, ok := c.Get("a"); ok {
		t.Fatalf("expected expiry")
	}
}

func TestCacheLRUEviction(t *testing.T) {
	c := NewLRU(2, time.Minute)
	c.Put("a", []byte("1"))
	c.Put("b", []byte("2"))
	c.Put("c", []byte("3")) // evicts a
	if _, ok := c.Get("a"); ok {
		t.Fatalf("a should be evicted")
	}
	if _, ok := c.Get("c"); !ok {
		t.Fatalf("c should be present")
	}
}

func TestClassify(t *testing.T) {
	cases := map[string]int{
		"error: Not signed in. Run `mlai login`.":   401,
		"free-pi is over its daily budget":          429,
		"too many requests this hour":               429,
		"your free-pi-cli is out of date":           400,
		"exec: \"mlai\": executable file not found": 503,
		"deadline exceeded":                         504,
	}
	for msg, want := range cases {
		if got := classify(msg).Status; got != want {
			t.Errorf("classify(%q) = %d, want %d", msg, got, want)
		}
	}
}

func TestSplitText(t *testing.T) {
	got := splitText("abcdefghij", 4)
	if len(got) != 3 || got[0] != "abcd" || got[2] != "ij" {
		t.Fatalf("got %#v", got)
	}
	if splitText("", 4) != nil {
		t.Fatalf("empty should be nil")
	}
}

func TestChatMessageText(t *testing.T) {
	var m chatMessage
	json.Unmarshal([]byte(`{"role":"user","content":"hi"}`), &m)
	if m.text() != "hi" {
		t.Fatalf("string content: %q", m.text())
	}
	var m2 chatMessage
	json.Unmarshal([]byte(`{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`), &m2)
	if m2.text() != "ab" {
		t.Fatalf("parts: %q", m2.text())
	}
}
