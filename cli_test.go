package main

// cli_test.go: subcommands and the launcher. The client is replaced by this
// test binary re-executed with WRAP_STUB=1 (see init), so no real claude or
// codex is needed.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

const cliKey = "k-test-secret-value"

// init turns the test binary into a stub client when WRAP_STUB=1. It records
// its arguments and environment in WRAP_STUB_OUT and exits with WRAP_STUB_EXIT,
// or dies from SIGTERM when WRAP_STUB_SIGNAL=1. TestMain in security_test.go
// stays the only TestMain.
func init() {
	if os.Getenv("WRAP_STUB") != "1" {
		return
	}
	if path := os.Getenv("WRAP_STUB_OUT"); path != "" {
		rec, _ := json.Marshal(stubDump{Args: os.Args[1:], Env: os.Environ()})
		os.WriteFile(path, rec, 0o600)
	}
	if os.Getenv("WRAP_STUB_SIGNAL") == "1" {
		if p, err := os.FindProcess(os.Getpid()); err == nil {
			p.Signal(syscall.SIGTERM)
		}
		time.Sleep(10 * time.Second)
	}
	code, _ := strconv.Atoi(os.Getenv("WRAP_STUB_EXIT"))
	os.Exit(code)
}

type stubDump struct {
	Args []string `json:"args"`
	Env  []string `json:"env"`
}

// readDump returns what the stub client recorded, with env keys upper-cased.
func readDump(t *testing.T, path string) (stubDump, map[string]string) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("the stub client did not run: %v", err)
	}
	var d stubDump
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	env := map[string]string{}
	for _, kv := range d.Env {
		name, val, _ := strings.Cut(kv, "=")
		env[strings.ToUpper(name)] = val
	}
	return d, env
}

// ---------- helpers ----------

// cliSandbox isolates globals, the home directory and the bridge-related env.
func cliSandbox(t *testing.T) string {
	t.Helper()
	saveGlobals(t)
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	for _, k := range []string{"MLAI_BRIDGE_URL", "MLAI_MODEL", "MLAI_API_KEY", "MLAI_BIND", "MLAI_WORKDIR", "WRAP_STUB", "WRAP_STUB_OUT", "WRAP_STUB_EXIT", "WRAP_STUB_SIGNAL"} {
		t.Setenv(k, "")
	}
	return home
}

func setLookPath(t *testing.T, f func(string) (string, error)) {
	t.Helper()
	old := lookPath
	lookPath = f
	t.Cleanup(func() { lookPath = old })
}

// useStubClient makes every client lookup resolve to this test binary.
func useStubClient(t *testing.T) {
	t.Helper()
	setLookPath(t, func(string) (string, error) { return os.Args[0], nil })
}

func noClient(t *testing.T) {
	t.Helper()
	setLookPath(t, func(string) (string, error) { return "", exec.ErrNotFound })
}

// fakeBridge answers /healthz and /v1/models like a running bridge.
func fakeBridge(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			fmt.Fprint(w, `{"status":"ok","version":"9.9.9","uptime_s":5,"signed_in":true}`)
		case "/v1/models":
			if r.Header.Get("Authorization") != "Bearer "+cliKey {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			fmt.Fprint(w, `{"object":"list","data":[{"id":"a-model"},{"id":"b-model"}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func wrapLaunch(kind string, args ...string) (code int, stdout, stderr string) {
	var out, errb bytes.Buffer
	code = runLaunch(kind, args, &out, &errb)
	return code, out.String(), errb.String()
}

// ---------- argument parsing ----------

func TestParseLaunchArgs(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want launchOpts
		fail bool
	}{
		{"none", nil, launchOpts{}, false},
		{"own flags then client args", []string{"-m", "x", "--url", "http://h", "--dry-run", "fix", "-m", "y"},
			launchOpts{model: "x", url: "http://h", dryRun: true, rest: []string{"fix", "-m", "y"}}, false},
		{"double dash ends own flags", []string{"--", "-m", "x"}, launchOpts{rest: []string{"-m", "x"}}, false},
		{"equals form", []string{"--model=a", "--url=http://h", "--", "--dry-run"},
			launchOpts{model: "a", url: "http://h", rest: []string{"--dry-run"}}, false},
		{"unknown flag belongs to the client", []string{"-p", "hi"}, launchOpts{rest: []string{"-p", "hi"}}, false},
		{"own flag after the first client arg is the client's", []string{"resume", "--dry-run"},
			launchOpts{rest: []string{"resume", "--dry-run"}}, false},
		{"missing value", []string{"-m"}, launchOpts{}, true},
		{"flag as value", []string{"--model", "--dry-run"}, launchOpts{}, true},
		{"empty value", []string{"--url="}, launchOpts{}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := parseLaunchArgs(c.args)
			if (err != nil) != c.fail {
				t.Fatalf("err = %v, want failure %v", err, c.fail)
			}
			if c.fail {
				return
			}
			if got.model != c.want.model || got.url != c.want.url || got.dryRun != c.want.dryRun || !slices.Equal(got.rest, c.want.rest) {
				t.Errorf("got %+v, want %+v", got, c.want)
			}
		})
	}
}

func TestBinaryNameAndHelpFlag(t *testing.T) {
	for _, p := range []string{"ml-wrap", "ml-wrap.exe", "ML-Wrap.EXE"} {
		if got := binaryName(filepath.Join("some", "dir", p)); got != "ml-wrap" {
			t.Errorf("binaryName(%q) = %q", p, got)
		}
	}
	for _, a := range []string{"-h", "--help", "--version"} {
		if !isHelpFlag(a) {
			t.Errorf("%q should be a help flag", a)
		}
	}
	for _, a := range []string{"-port", "--model", "-version", "serve"} {
		if isHelpFlag(a) {
			t.Errorf("%q should not be a help flag", a)
		}
	}
}

func TestRunCLIBasics(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCLI([]string{"bogus"}, &out, &errb); code != 2 || out.Len() != 0 || !strings.Contains(errb.String(), `unknown command "bogus"`) {
		t.Errorf("unknown command: code %d, stdout %q, stderr %q", code, out.String(), errb.String())
	}
	for _, a := range []string{"help", "-h", "--help"} {
		out.Reset()
		if code := runCLI([]string{a}, &out, &errb); code != 0 || !strings.Contains(out.String(), "ml-wrap") {
			t.Errorf("%s: code %d, stdout %q", a, code, out.String())
		}
	}
	for _, a := range []string{"version", "--version"} {
		out.Reset()
		if code := runCLI([]string{a}, &out, &errb); code != 0 || strings.TrimSpace(out.String()) != "mlai-bridge "+version {
			t.Errorf("%s: code %d, stdout %q", a, code, out.String())
		}
	}
}

// ---------- environment and arguments ----------

func TestPlanClaudeEnvironment(t *testing.T) {
	plan := planLaunch("claude", "http://127.0.0.1:5000", cliKey, "m-1")
	parent := []string{
		"PATH=/usr/bin",
		"anthropic_api_key=leak1",
		"ANTHROPIC_BETAS=b",
		"Anthropic_Custom_Headers=h",
		"CLAUDE_CODE_OAUTH_TOKEN=o",
		"CLAUDE_CODE_USE_BEDROCK=1",
		"claude_code_use_vertex=1",
		"CLAUDE_CODE_USE_FOUNDRY=1",
		"CLAUDE_CODE_USE_MANTLE=1",
		"MLAI_API_KEY=leak2",
		"CLAUDECODE=1",
		"ANTHROPIC_BASE_URL=http://old.example",
		"ANTHROPIC_MODEL=old",
	}
	env := plan.environ(parent)
	got := map[string]string{}
	for _, kv := range env {
		name, val, _ := strings.Cut(kv, "=")
		if _, dup := got[strings.ToUpper(name)]; dup && name != "no_proxy" {
			t.Errorf("variable %s appears twice", name)
		}
		got[strings.ToUpper(name)] = val
	}
	for _, name := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_BETAS", "ANTHROPIC_CUSTOM_HEADERS", "CLAUDE_CODE_OAUTH_TOKEN",
		"CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_MANTLE", "MLAI_API_KEY", "CLAUDECODE"} {
		if _, ok := got[name]; ok {
			t.Errorf("%s is still in the child environment", name)
		}
	}
	expect := func(name, want string) {
		t.Helper()
		if got[name] != want {
			t.Errorf("%s = %q, want %q", name, got[name], want)
		}
	}
	expect("PATH", "/usr/bin")
	expect("ANTHROPIC_BASE_URL", "http://127.0.0.1:5000")
	expect("ANTHROPIC_AUTH_TOKEN", cliKey)
	for _, name := range []string{"ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_FABLE_MODEL"} {
		expect(name, "m-1")
	}
	expect("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
	expect("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1")
	if len(plan.Args) != 0 {
		t.Errorf("claude gets no extra arguments, got %v", plan.Args)
	}
}

func TestPlanCodexArgsAndEnvironment(t *testing.T) {
	plan := planLaunch("codex", "http://127.0.0.1:5000", cliKey, "m-2")
	joined := strings.Join(plan.Args, "\n")
	if strings.Contains(joined, cliKey) {
		t.Fatalf("the key is in argv: %v", plan.Args)
	}
	if strings.Contains(joined, `"`) {
		t.Errorf("argv contains a double quote (cmd.exe mangles it): %v", plan.Args)
	}
	wantArgs := []string{
		"-c", "model_provider='mlai'",
		"-c", "model_providers.mlai.name='mlai-bridge'",
		"-c", "model_providers.mlai.base_url='http://127.0.0.1:5000/v1'",
		"-c", "model_providers.mlai.env_key='MLAI_BRIDGE_KEY'",
		"-c", "model_providers.mlai.wire_api='responses'",
		"-c", "model_providers.mlai.supports_websockets=false",
		"-m", "m-2",
	}
	if !slices.Equal(plan.Args, wantArgs) {
		t.Errorf("args = %v\nwant   %v", plan.Args, wantArgs)
	}
	env := plan.environ([]string{"openai_api_key=x", "OPENAI_BASE_URL=y", "MLAI_API_KEY=z", "MLAI_BRIDGE_KEY=old", "HOME=/h"})
	slices.Sort(env)
	want := []string{"HOME=/h", "MLAI_BRIDGE_KEY=" + cliKey, "NO_PROXY=127.0.0.1,localhost,::1"}
	if runtime.GOOS != "windows" {
		want = append(want, "no_proxy=127.0.0.1,localhost,::1")
	}
	if !slices.Equal(env, want) {
		t.Errorf("env = %v, want %v", env, want)
	}
	if empty := planLaunch("codex", "http://h", "", "m"); empty.Set[0] != "MLAI_BRIDGE_KEY=none" {
		t.Errorf("empty key: %v", empty.Set)
	}
}

func TestTomlStr(t *testing.T) {
	for in, want := range map[string]string{"mlai": `'mlai'`, "it's": `"it's"`, "a\nb": `"a\nb"`} {
		if got := tomlStr(in); got != want {
			t.Errorf("tomlStr(%q) = %s, want %s", in, got, want)
		}
	}
}

// ---------- dry run ----------

func TestDryRunPrintsNamesAndHidesKey(t *testing.T) {
	home := cliSandbox(t)
	noClient(t)
	t.Setenv("MLAI_API_KEY", cliKey)
	for _, kind := range []string{"claude", "codex"} {
		for _, args := range [][]string{
			{"--dry-run", "--url", "http://127.0.0.1:1", "-m", "m-3", "--", "exec", "hello world"},
			{"--dry-run", "exec", "hello world"},
		} {
			code, out, errs := wrapLaunch(kind, args...)
			if code != 0 || errs != "" {
				t.Fatalf("%s %v: code %d, stderr %q", kind, args, code, errs)
			}
			if strings.Contains(out, cliKey) {
				t.Errorf("%s: the key is in the dry-run output:\n%s", kind, out)
			}
			for _, want := range []string{"command: " + kind, `"hello world"`, "env set:", "env removed:", "MLAI_API_KEY"} {
				if !strings.Contains(out, want) {
					t.Errorf("%s %v: output lacks %q:\n%s", kind, args, want, out)
				}
			}
			keyVar := map[string]string{"claude": "ANTHROPIC_AUTH_TOKEN", "codex": "MLAI_BRIDGE_KEY"}[kind]
			if !strings.Contains(out, keyVar+"=***") {
				t.Errorf("%s: output lacks %s=***:\n%s", kind, keyVar, out)
			}
		}
	}
	if _, err := os.Stat(filepath.Join(home, "mlai-bridge")); err == nil {
		t.Error("a dry run created the bridge directory (it must start nothing)")
	}
}

// ---------- remote bridge ----------

func TestBridgeURL(t *testing.T) {
	cases := []struct {
		in   string
		want string
		fail bool
	}{
		{"http://127.0.0.1:8197", "http://127.0.0.1:8197", false},
		{"http://localhost:1/v1/", "http://localhost:1", false},
		{"http://[::1]:8197/", "http://[::1]:8197", false},
		{"https://bridge.example/base/v1", "https://bridge.example/base", false},
		{"https://bridge.example:8443?x=1#f", "https://bridge.example:8443", false},
		{"http://bridge.example:8197", "", true},
		{"http://127.0.0.1.evil.example", "", true},
		{"http://10.0.0.5:8197", "", true},
		{"ftp://127.0.0.1", "", true},
		{"127.0.0.1:8197", "", true},
		{"http://user:pw@127.0.0.1:8197", "", true},
		{"", "", true},
	}
	for _, c := range cases {
		got, err := bridgeURL(c.in)
		if (err != nil) != c.fail || got != c.want {
			t.Errorf("bridgeURL(%q) = %q, %v; want %q (fail %v)", c.in, got, err, c.want, c.fail)
		}
	}
	if _, err := bridgeURL("http://bridge.example"); err == nil || !strings.Contains(err.Error(), "clear") {
		t.Errorf("refusal should say the key would be sent in clear: %v", err)
	}
}

func TestRemotePlainHTTPIsRefused(t *testing.T) {
	cliSandbox(t)
	useStubClient(t)
	dump := filepath.Join(t.TempDir(), "dump.json")
	t.Setenv("WRAP_STUB", "1")
	t.Setenv("WRAP_STUB_OUT", dump)
	t.Setenv("MLAI_API_KEY", cliKey)
	for _, kind := range []string{"claude", "codex"} {
		for _, args := range [][]string{
			{"--url", "http://bridge.example:8197"},
			{"--dry-run", "--url", "http://bridge.example:8197"},
		} {
			code, out, errs := wrapLaunch(kind, args...)
			if code != 2 || out != "" || !strings.Contains(errs, "refusing plain http") {
				t.Errorf("%s %v: code %d, stdout %q, stderr %q", kind, args, code, out, errs)
			}
		}
	}
	t.Setenv("MLAI_BRIDGE_URL", "http://bridge.example:8197")
	if code, _, errs := wrapLaunch("claude"); code != 2 || !strings.Contains(errs, "refusing plain http") {
		t.Errorf("MLAI_BRIDGE_URL: code %d, stderr %q", code, errs)
	}
	if _, err := os.Stat(dump); err == nil {
		t.Error("the client was started although the URL was refused")
	}
	if code, _, errs := wrapLaunch("claude", "--dry-run", "--url", "https://bridge.example"); code != 0 {
		t.Errorf("https should be accepted: code %d, stderr %q", code, errs)
	}
}

func TestRemoteHealthCheckGatesTheClient(t *testing.T) {
	cliSandbox(t)
	useStubClient(t)
	dump := filepath.Join(t.TempDir(), "dump.json")
	t.Setenv("WRAP_STUB", "1")
	t.Setenv("WRAP_STUB_OUT", dump)
	notBridge := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			fmt.Fprint(w, `{"hello":"world"}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer notBridge.Close()
	empty := httptest.NewServer(http.NotFoundHandler())
	defer empty.Close()
	closed := httptest.NewServer(http.NotFoundHandler())
	closedURL := closed.URL
	closed.Close()
	for name, url := range map[string]string{"not a bridge": notBridge.URL, "404": empty.URL, "closed port": closedURL} {
		code, _, errs := wrapLaunch("codex", "--url", url)
		if code != 1 || !strings.Contains(errs, "no healthy bridge") && !strings.Contains(errs, "does not look like") {
			t.Errorf("%s: code %d, stderr %q", name, code, errs)
		}
	}
	if _, err := os.Stat(dump); err == nil {
		t.Error("the client was started although the health check failed")
	}
}

// ---------- running the client ----------

func TestExitCodePropagationAndChildEnvironment(t *testing.T) {
	cliSandbox(t)
	useStubClient(t)
	bridge := fakeBridge(t)
	dump := filepath.Join(t.TempDir(), "dump.json")
	t.Setenv("WRAP_STUB", "1")
	t.Setenv("WRAP_STUB_OUT", dump)
	t.Setenv("MLAI_API_KEY", cliKey)
	t.Setenv("ANTHROPIC_API_KEY", "leak-anthropic")
	t.Setenv("OPENAI_API_KEY", "leak-openai")
	t.Setenv("CLAUDECODE", "1")

	for _, want := range []int{0, 3, 42} {
		t.Setenv("WRAP_STUB_EXIT", strconv.Itoa(want))
		code, _, errs := wrapLaunch("claude", "--url", bridge.URL, "--", "-m", "keep", "--dry-run", "a b")
		if code != want {
			t.Fatalf("exit code = %d, want %d (stderr %q)", code, want, errs)
		}
	}
	d, env := readDump(t, dump)
	if want := []string{"-m", "keep", "--dry-run", "a b"}; !slices.Equal(d.Args, want) {
		t.Errorf("client args = %v, want %v", d.Args, want)
	}
	if env["ANTHROPIC_AUTH_TOKEN"] != cliKey || env["ANTHROPIC_BASE_URL"] != bridge.URL || env["ANTHROPIC_MODEL"] != "pixis-opus" {
		t.Errorf("claude env: token %q, base %q, model %q", env["ANTHROPIC_AUTH_TOKEN"], env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_MODEL"])
	}
	for _, name := range []string{"ANTHROPIC_API_KEY", "MLAI_API_KEY", "CLAUDECODE"} {
		if v, ok := env[name]; ok {
			t.Errorf("%s=%q reached the client", name, v)
		}
	}

	code, _, errs := wrapLaunch("codex", "--url", bridge.URL, "-m", "m-4", "exec", "do it")
	if code != 42 {
		t.Fatalf("codex exit code = %d, stderr %q", code, errs)
	}
	d, env = readDump(t, dump)
	if !slices.Contains(d.Args, "m-4") || !slices.Equal(d.Args[len(d.Args)-2:], []string{"exec", "do it"}) {
		t.Errorf("codex args = %v", d.Args)
	}
	if !slices.Contains(d.Args, "model_providers.mlai.base_url='"+bridge.URL+"/v1'") {
		t.Errorf("codex args lack the base url: %v", d.Args)
	}
	for _, a := range d.Args {
		if strings.Contains(a, cliKey) {
			t.Errorf("the key is in the client's argv: %q", a)
		}
	}
	if env["MLAI_BRIDGE_KEY"] != cliKey {
		t.Errorf("MLAI_BRIDGE_KEY = %q", env["MLAI_BRIDGE_KEY"])
	}
	for _, name := range []string{"MLAI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"} {
		if v, ok := env[name]; ok {
			t.Errorf("%s=%q reached codex", name, v)
		}
	}
}

func TestSignalDeathExitsWith128PlusSignal(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("no signal exit status on Windows")
	}
	cliSandbox(t)
	useStubClient(t)
	bridge := fakeBridge(t)
	t.Setenv("WRAP_STUB", "1")
	t.Setenv("WRAP_STUB_SIGNAL", "1")
	if code, _, errs := wrapLaunch("claude", "--url", bridge.URL); code != 128+int(syscall.SIGTERM) {
		t.Errorf("exit code = %d, want %d (stderr %q)", code, 128+int(syscall.SIGTERM), errs)
	}
}

func TestMissingClientExits127(t *testing.T) {
	cliSandbox(t)
	noClient(t)
	bridge := fakeBridge(t)
	for _, kind := range []string{"claude", "codex"} {
		code, _, errs := wrapLaunch(kind, "--url", bridge.URL)
		if code != 127 || !strings.Contains(errs, kind+" not found") {
			t.Errorf("%s: code %d, stderr %q", kind, code, errs)
		}
	}
}

func TestPrivateBridgeForTheClient(t *testing.T) {
	home := cliSandbox(t)
	useStubClient(t)
	dump := filepath.Join(t.TempDir(), "dump.json")
	t.Setenv("WRAP_STUB", "1")
	t.Setenv("WRAP_STUB_OUT", dump)
	t.Setenv("WRAP_STUB_EXIT", "5")
	t.Setenv("MLAI_API_KEY", cliKey)
	t.Setenv("MLAI_MODEL", "env-model")

	code, _, errs := wrapLaunch("claude", "--", "hello")
	if code != 5 {
		t.Fatalf("exit code = %d, want 5 (stderr %q)", code, errs)
	}
	d, env := readDump(t, dump)
	if !slices.Equal(d.Args, []string{"hello"}) {
		t.Errorf("client args = %v", d.Args)
	}
	if !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(env["ANTHROPIC_AUTH_TOKEN"]) || env["ANTHROPIC_AUTH_TOKEN"] == cliKey {
		t.Errorf("token %q is not a fresh 32-byte hex key", env["ANTHROPIC_AUTH_TOKEN"])
	}
	if !strings.HasPrefix(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:") || env["ANTHROPIC_MODEL"] != "env-model" {
		t.Errorf("base %q, model %q", env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_MODEL"])
	}
	if _, ok := env["MLAI_API_KEY"]; ok {
		t.Error("MLAI_API_KEY reached the client")
	}

	logPath := filepath.Join(home, "mlai-bridge", "wrap.log")
	st, err := os.Stat(logPath)
	if err != nil {
		t.Fatalf("wrap.log: %v", err)
	}
	if runtime.GOOS != "windows" {
		if st.Mode().Perm() != 0o600 {
			t.Errorf("wrap.log mode = %v, want 0600", st.Mode().Perm())
		}
		if dir, err := os.Stat(filepath.Dir(logPath)); err != nil || dir.Mode().Perm() != 0o700 {
			t.Errorf("log directory mode: %v, %v", dir, err)
		}
	}
	// A second launch gets its own key.
	first := env["ANTHROPIC_AUTH_TOKEN"]
	if code, _, _ := wrapLaunch("claude"); code != 5 {
		t.Fatalf("second launch exit code = %d", code)
	}
	if _, env = readDump(t, dump); env["ANTHROPIC_AUTH_TOKEN"] == first {
		t.Error("two launches used the same key")
	}
}

// ---------- Windows .cmd shims ----------

func TestPrepareClientCmdShim(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) string {
		t.Helper()
		p := filepath.Join(dir, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}

	// A plain .cmd: safe arguments pass, every cmd.exe metacharacter is refused.
	plain := write("plain/tool.cmd", "@echo off\r\n")
	safe := []string{"-c", "model_provider='mlai'", "fix the bug"}
	if p, a, err := prepareClient(plain, safe); err != nil || p != plain || !slices.Equal(a, safe) {
		t.Errorf("safe args: %q %v %v", p, a, err)
	}
	for _, c := range strings.Split(cmdUnsafe, "") {
		if _, _, err := prepareClient(plain, []string{"a" + c + "b"}); err == nil {
			t.Errorf("argument with %q was not refused", c)
		}
	}

	// A real .exe next to the shim wins, and then any argument is fine.
	shim := write("withexe/tool.cmd", "@echo off\r\n")
	exe := write("withexe/tool.exe", "")
	if p, a, err := prepareClient(shim, []string{"a&b"}); err != nil || p != exe || !slices.Equal(a, []string{"a&b"}) {
		t.Errorf("sibling exe: %q %v %v", p, a, err)
	}

	// An npm shim is replaced by node plus its script.
	npm := write("npm/claude.cmd", "@ECHO off\r\n\"%_prog%\"  \"%dp0%\\node_modules\\@scope\\pkg\\cli.js\" %*\r\n")
	script := write("npm/node_modules/@scope/pkg/cli.js", "")
	node := write("npm/node.exe", "")
	p, a, err := prepareClient(npm, []string{"a&b"})
	if err != nil || p != node || !slices.Equal(a, []string{script, "a&b"}) {
		t.Errorf("npm shim: %q %v %v", p, a, err)
	}

	// Anything that is not a .cmd/.bat is left alone.
	if p, a, err := prepareClient("tool.exe", []string{"a&b"}); err != nil || p != "tool.exe" || len(a) != 1 {
		t.Errorf("exe: %q %v %v", p, a, err)
	}
}

// ---------- env, models, status, doctor ----------

func TestEnvCommand(t *testing.T) {
	cliSandbox(t)
	t.Setenv("MLAI_API_KEY", "must-not-be-printed")
	run := func(args ...string) (int, string, string) {
		var out, errb bytes.Buffer
		code := runCLI(append([]string{"env"}, args...), &out, &errb)
		return code, out.String(), errb.String()
	}

	code, out, errs := run("claude", "--url", "http://127.0.0.1:9000", "-m", "m-5")
	if code != 0 || errs != "" {
		t.Fatalf("code %d, stderr %q", code, errs)
	}
	for _, want := range []string{
		"export ANTHROPIC_BASE_URL='http://127.0.0.1:9000'\n",
		"export ANTHROPIC_MODEL='m-5'\n",
		`export ANTHROPIC_AUTH_TOKEN="${MLAI_API_KEY:-none}"` + "\n",
		"export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1'\n",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("output lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "must-not-be-printed") {
		t.Error("the key value was printed")
	}

	// Flags may come before or after the client name.
	code, out, _ = run("--powershell", "codex", "--url", "http://localhost:9000")
	if code != 0 || !strings.Contains(out, "$env:MLAI_BRIDGE_KEY = if ($env:MLAI_API_KEY)") || !strings.Contains(out, "# then run: codex -c") ||
		strings.Contains(out, "must-not-be-printed") {
		t.Errorf("powershell codex: code %d\n%s", code, out)
	}
	if code, out, _ = run("claude", "--powershell", "--url", "http://127.0.0.1:9000"); code != 0 || !strings.Contains(out, "$env:ANTHROPIC_BASE_URL = 'http://127.0.0.1:9000'\n") {
		t.Errorf("powershell claude: code %d\n%s", code, out)
	}

	for _, args := range [][]string{{}, {"vim"}, {"claude", "codex"}, {"claude", "--url", "http://example.com"}, {"claude", "--nope"}} {
		if code, out, _ := run(args...); code != 2 || out != "" {
			t.Errorf("env %v: code %d, stdout %q, want 2 and nothing", args, code, out)
		}
	}
}

func TestModelsAndStatusCommands(t *testing.T) {
	cliSandbox(t)
	bridge := fakeBridge(t)
	run := func(args ...string) (int, string, string) {
		var out, errb bytes.Buffer
		code := runCLI(args, &out, &errb)
		return code, out.String(), errb.String()
	}

	t.Setenv("MLAI_API_KEY", cliKey)
	if code, out, errs := run("models", "--url", bridge.URL); code != 0 || out != "a-model\nb-model\n" {
		t.Errorf("models: code %d, stdout %q, stderr %q", code, out, errs)
	}
	t.Setenv("MLAI_API_KEY", "wrong")
	if code, out, errs := run("models", "--url", bridge.URL); code != 1 || out != "" || !strings.Contains(errs, "401") {
		t.Errorf("models with a wrong key: code %d, stdout %q, stderr %q", code, out, errs)
	}

	t.Setenv("MLAI_BRIDGE_URL", bridge.URL)
	if code, out, _ := run("status"); code != 0 || !strings.Contains(out, "9.9.9") || !strings.Contains(out, "ok") || !strings.Contains(out, "signed in: yes") {
		t.Errorf("status: code %d, stdout %q", code, out)
	}
	bridge.Close()
	if code, out, errs := run("status"); code != 1 || out != "" || !strings.Contains(errs, "down") {
		t.Errorf("status of a closed bridge: code %d, stdout %q, stderr %q", code, out, errs)
	}
}

func TestDoctorReportsYesNoOnly(t *testing.T) {
	cliSandbox(t)
	t.Setenv("MLAI_STUB", "1") // the re-executed test binary answers `whoami` (see TestMain)
	t.Setenv("MLAI_API_KEY", cliKey)
	var out bytes.Buffer

	setLookPath(t, func(name string) (string, error) {
		if name == "codex" {
			return "", exec.ErrNotFound
		}
		return os.Args[0], nil
	})
	if code := runDoctor(&out); code != 0 {
		t.Errorf("code = %d, want 0\n%s", code, out.String())
	}
	for _, want := range []string{"mlai in PATH:", "claude in PATH:", "codex in PATH:", "mlai signed in:"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output lacks %q:\n%s", want, out.String())
		}
	}
	for _, line := range strings.Split(strings.TrimSpace(out.String()), "\n") {
		f := strings.Fields(line)
		if last := f[len(f)-1]; last != "yes" && last != "no" {
			t.Errorf("line %q does not end in yes or no", line)
		}
	}
	if strings.Contains(out.String(), "stub@example.test") || strings.Contains(out.String(), cliKey) {
		t.Errorf("doctor printed account details or the key:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "codex in PATH:") || !regexp.MustCompile(`codex in PATH:\s+no`).MatchString(out.String()) {
		t.Errorf("codex should be reported missing:\n%s", out.String())
	}

	out.Reset()
	noClient(t)
	if code := runDoctor(&out); code != 1 || !regexp.MustCompile(`mlai in PATH:\s+no`).MatchString(out.String()) {
		t.Errorf("without mlai: code %d\n%s", code, out.String())
	}
}
