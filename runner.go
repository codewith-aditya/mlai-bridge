package main

// runner.go — executes the mlai CLI and parses its stream-json output.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Usage mirrors the CLI's token accounting.
type Usage struct {
	PromptTokens     int     `json:"promptTokens"`
	CompletionTokens int     `json:"completionTokens"`
	CachedTokens     int     `json:"cachedTokens"`
	TotalTokens      int     `json:"totalTokens"`
	Credits          float64 `json:"credits"`
	Calls            int     `json:"calls"`
	Engine           string  `json:"engine"`
}

// RunResult is what one CLI invocation produced.
type RunResult struct {
	Text      string
	Model     string
	SessionID string
	Usage     Usage
	Err       error
}

// streamEvent is the subset of CLI events we care about.
type streamEvent struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Text    string `json:"text"`
	Model   string `json:"model"`
	ID      string `json:"id"`
	Usage   struct {
		PromptTokens     int `json:"promptTokens"`
		CompletionTokens int `json:"completionTokens"`
		CachedTokens     int `json:"cachedTokens"`
		TotalTokens      int `json:"totalTokens"`
	} `json:"usage"`
	Credits float64 `json:"credits"`
	Engine  string  `json:"engine"`
	// Error is {"code","message"} or, from mlai 0.3.x, a plain string.
	Error json.RawMessage `json:"error"`
}

// cliError returns the message of a stream-json error field, or "".
func cliError(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var o struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
	if json.Unmarshal(raw, &o) != nil || o.Message == "" {
		return ""
	}
	if o.Code == "" {
		return o.Message
	}
	return o.Code + ": " + o.Message
}

// cappedBuffer is a goroutine-safe writer that keeps the first max bytes and
// discards the rest.
type cappedBuffer struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if room := b.max - len(b.buf); room > 0 {
		b.buf = append(b.buf, p[:min(room, len(p))]...)
	}
	return len(p), nil
}

func (b *cappedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(b.buf)
}

// Runner shells out to `mlai`.
type Runner struct {
	Bin     string
	WorkDir string
	Timeout int                      // seconds
	OnEvent func(raw map[string]any) // optional tap for streaming
}

// Run executes one prompt and returns the answer. `cancel` kills the process.
func (r *Runner) Run(ctx context.Context, prompt, model, effort string) RunResult {
	if strings.TrimSpace(prompt) == "" {
		return RunResult{Err: fmt.Errorf("empty prompt")}
	}
	// `-p` without a value reads the task from stdin, so the prompt can never be
	// parsed as a flag and does not show up in the process list.
	args := []string{"-p", "--output-format", "stream-json", "--no-tui"}
	if model != "" {
		args = append(args, "-m", model)
	}
	if effort != "" {
		args = append(args, "--effort", effort)
	}
	// `ask` keeps the agent from editing files: this is a chat bridge.
	args = append(args, "--mode", "ask")

	cmd := exec.CommandContext(ctx, r.Bin, args...)
	cmd.Dir = r.WorkDir
	cmd.Env = append(childEnv(), "TERM=dumb")
	cmd.Stdin = strings.NewReader(prompt)
	setProcGroup(cmd)
	cmd.Cancel = func() error { return killTree(cmd) }
	cmd.WaitDelay = 5 * time.Second

	pr, pw := io.Pipe()
	cmd.Stdout = pw
	stderr := &cappedBuffer{max: 16 << 10}
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		return RunResult{Err: fmt.Errorf("start mlai: %w", err)}
	}
	waitCh := make(chan error, 1)
	go func() {
		err := cmd.Wait()
		pw.Close()
		waitCh <- err
	}()

	var res RunResult
	sc := bufio.NewScanner(pr)
	sc.Buffer(make([]byte, 1<<20), 4<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var ev streamEvent
		if json.Unmarshal([]byte(line), &ev) != nil {
			continue
		}
		if r.OnEvent != nil {
			var raw map[string]any
			if json.Unmarshal([]byte(line), &raw) == nil {
				r.OnEvent(raw)
			}
		}
		switch ev.Type {
		case "answer":
			if ev.Content != "" {
				res.Text = ev.Content
			}
		case "model":
			if ev.ID != "" {
				res.Model = ev.ID
			}
		case "result":
			if ev.Text != "" {
				res.Text = ev.Text
			}
			res.SessionID = ev.ID
			res.Usage = Usage{
				PromptTokens:     ev.Usage.PromptTokens,
				CompletionTokens: ev.Usage.CompletionTokens,
				CachedTokens:     ev.Usage.CachedTokens,
				TotalTokens:      ev.Usage.TotalTokens,
				Credits:          ev.Credits,
			}
			if ev.Model != "" {
				res.Model = ev.Model
			}
			if msg := cliError(ev.Error); msg != "" {
				res.Err = fmt.Errorf("%s", msg)
			}
		case "error":
			if ev.Content != "" {
				res.Err = fmt.Errorf("%s", ev.Content)
			}
		}
	}
	scanErr := sc.Err()
	// Keep reading so the child never blocks on a full pipe (scanner stopped early).
	io.Copy(io.Discard, pr)
	waitErr := <-waitCh
	if errors.Is(waitErr, exec.ErrWaitDelay) {
		waitErr = nil // the process exited; only an inherited pipe held on
	}

	var errLines []string
	for _, l := range strings.Split(stderr.String(), "\n") {
		if l = strings.TrimSpace(l); l != "" && len(errLines) < 20 {
			errLines = append(errLines, l)
		}
	}
	if scanErr != nil && res.Err == nil && res.Text == "" {
		res.Err = scanErr
	}
	if waitErr != nil && res.Err == nil {
		if cerr := ctx.Err(); cerr != nil {
			res.Err = cerr // "context deadline exceeded" keeps classify at 504
		} else {
			msg := strings.Join(errLines, " | ")
			if msg == "" {
				msg = waitErr.Error()
			}
			res.Err = fmt.Errorf("%s", msg)
		}
	}
	if res.Err == nil && strings.TrimSpace(res.Text) == "" {
		msg := strings.Join(errLines, " | ")
		if msg == "" {
			msg = "cli produced no answer"
		}
		res.Err = fmt.Errorf("%s", msg)
	}
	return res
}
