package main

// runner.go — executes the mlai CLI and parses its stream-json output.

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
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
	Error   *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
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
	args := []string{"-p", prompt, "--output-format", "stream-json", "--no-tui"}
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
	cmd.Env = append(cmd.Environ(), "NO_COLOR=1", "CI=1", "TERM=dumb")
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return RunResult{Err: fmt.Errorf("stdout pipe: %w", err)}
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return RunResult{Err: fmt.Errorf("stderr pipe: %w", err)}
	}
	if err := cmd.Start(); err != nil {
		return RunResult{Err: fmt.Errorf("start mlai: %w", err)}
	}

	var (
		res      RunResult
		mu       sync.Mutex
		errLines []string
	)
	go func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			mu.Lock()
			if len(errLines) < 20 {
				errLines = append(errLines, line)
			}
			mu.Unlock()
		}
	}()

	sc := bufio.NewScanner(stdout)
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
			if ev.Error != nil && ev.Error.Message != "" {
				res.Err = fmt.Errorf("%s: %s", ev.Error.Code, ev.Error.Message)
			}
		case "error":
			if ev.Content != "" {
				res.Err = fmt.Errorf("%s", ev.Content)
			}
		}
	}

	waitErr := cmd.Wait()
	if waitErr != nil && res.Err == nil {
		msg := strings.Join(errLines, " | ")
		if msg == "" {
			msg = waitErr.Error()
		}
		res.Err = fmt.Errorf("%s", msg)
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

// RunText is a convenience for one-shot prompts (used by /healthz checks).
func (r *Runner) RunText(ctx context.Context, prompt string) (string, error) {
	out, err := exec.CommandContext(ctx, r.Bin, "-p", prompt, "--mode", "ask", "--output-format", "text", "--no-tui").CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("%s", strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out)), nil
}

var _ = io.Discard
