package main

// cli.go — subcommands and the launcher. `claude` and `codex` start the client
// pointed at a bridge (their own, or a running one) through environment
// variables and arguments only; no file of the client is written.

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const defaultBridgeURL = "http://127.0.0.1:8197"

const helpText = `mlai-bridge ` + version + ` — OpenAI/Anthropic compatible API on top of the mlai CLI

Usage:
  mlai-bridge [flags]              run the bridge (same as "serve")
  mlai-bridge serve [flags]        run the bridge
  mlai-bridge claude [opts] [--] [claude args...]
  mlai-bridge codex  [opts] [--] [codex args...]
                                   start the client against a bridge
                                   opts: -m/--model ID, --url URL, --dry-run
  mlai-bridge env <claude|codex> [--url URL] [--powershell] [-m ID]
                                   print variables for a running bridge
  mlai-bridge models [--url URL]   list the model ids of a running bridge
  mlai-bridge status [--url URL]   health of a running bridge (exit 1 when down)
  mlai-bridge doctor               check mlai, claude and codex
  mlai-bridge version
  mlai-bridge help

claude and codex use the bridge at --url or MLAI_BRIDGE_URL (key from
MLAI_API_KEY). Without either, they run a private bridge on 127.0.0.1 with a
random key for the lifetime of the client; its log is ~/mlai-bridge/wrap.log.
The same binary also works as "ml-wrap".
`

// lookPath is replaced in tests.
var lookPath = exec.LookPath

// binaryName is the lower-case program name without a .exe suffix.
func binaryName(arg0 string) string {
	return strings.TrimSuffix(strings.ToLower(filepath.Base(arg0)), ".exe")
}

func isHelpFlag(a string) bool {
	return a == "-h" || a == "--help" || a == "--version"
}

func printHelp(w io.Writer) {
	io.WriteString(w, helpText)
}

// runCLI runs every subcommand except plain serving and returns the exit code.
func runCLI(args []string, stdout, stderr io.Writer) int {
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "serve":
		serveMain(rest)
		return 0
	case "claude", "codex":
		return runLaunch(cmd, rest, stdout, stderr)
	case "env":
		return runEnv(rest, stdout, stderr)
	case "models":
		return runModels(rest, stdout, stderr)
	case "status":
		return runStatus(rest, stdout, stderr)
	case "doctor":
		return runDoctor(stdout)
	case "version", "--version":
		fmt.Fprintf(stdout, "mlai-bridge %s\n", version)
		return 0
	case "help", "-h", "--help":
		printHelp(stdout)
		return 0
	}
	fmt.Fprintf(stderr, "mlai-bridge: unknown command %q\n\n", cmd)
	printHelp(stderr)
	return 2
}

// ---------- launcher ----------

type launchOpts struct {
	model  string
	url    string
	dryRun bool
	rest   []string
}

// parseLaunchArgs reads the launcher's own leading flags. The first other
// argument, or `--`, ends them; everything after is for the client.
func parseLaunchArgs(args []string) (launchOpts, error) {
	var o launchOpts
	i := 0
	for i < len(args) {
		a := args[i]
		if a == "--" {
			i++
			break
		}
		if a == "--dry-run" {
			o.dryRun = true
			i++
			continue
		}
		name, val, hasVal := strings.Cut(a, "=")
		if name != "-m" && name != "--model" && name != "--url" {
			break
		}
		if !hasVal {
			if i+1 >= len(args) {
				return o, fmt.Errorf("%s needs a value", name)
			}
			i++
			val = args[i]
		}
		if val == "" || strings.HasPrefix(val, "-") {
			return o, fmt.Errorf("%s needs a value, got %q", name, val)
		}
		if name == "--url" {
			o.url = val
		} else {
			o.model = val
		}
		i++
	}
	o.rest = append([]string(nil), args[i:]...)
	return o, nil
}

// launchPlan is what the launcher adds to the client's command line and
// environment. The key only ever appears in Set, under KeyVar.
type launchPlan struct {
	Args   []string // client arguments that come before the user's own
	Set    []string // NAME=value pairs added to the child environment
	Remove []string // names stripped from the inherited environment
	KeyVar string   // name of the variable that carries the bridge key
}

// tomlStr quotes s as a TOML string. Literal (single-quoted) strings need no
// escaping and keep double quotes out of the command line, which cmd.exe mangles.
func tomlStr(s string) string {
	if strings.ContainsAny(s, "'\r\n") {
		return strconv.Quote(s)
	}
	return "'" + s + "'"
}

// planLaunch builds the plan for kind ("claude" or "codex"). base is the bridge
// root without /v1. An empty key becomes "none".
func planLaunch(kind, base, key, model string) launchPlan {
	if key == "" {
		key = "none"
	}
	var p launchPlan
	if kind == "codex" {
		p.Args = []string{
			"-c", "model_provider=" + tomlStr("mlai"),
			"-c", "model_providers.mlai.name=" + tomlStr("mlai-bridge"),
			"-c", "model_providers.mlai.base_url=" + tomlStr(base+"/v1"),
			"-c", "model_providers.mlai.env_key=" + tomlStr("MLAI_BRIDGE_KEY"),
			"-c", "model_providers.mlai.wire_api=" + tomlStr("responses"),
			"-c", "model_providers.mlai.supports_websockets=false",
			"-m", model,
		}
		p.Set = []string{"MLAI_BRIDGE_KEY=" + key}
		p.Remove = []string{"MLAI_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"}
		p.KeyVar = "MLAI_BRIDGE_KEY"
		return p
	}
	p.Set = []string{
		"ANTHROPIC_BASE_URL=" + base,
		"ANTHROPIC_AUTH_TOKEN=" + key,
		"ANTHROPIC_MODEL=" + model,
		"ANTHROPIC_DEFAULT_OPUS_MODEL=" + model,
		"ANTHROPIC_DEFAULT_SONNET_MODEL=" + model,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=" + model,
		"ANTHROPIC_DEFAULT_FABLE_MODEL=" + model,
		"ANTHROPIC_SMALL_FAST_MODEL=" + model,
		"CLAUDE_CODE_SUBAGENT_MODEL=" + model,
		"CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1",
		"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1",
	}
	p.Remove = []string{
		"ANTHROPIC_API_KEY", "ANTHROPIC_BETAS", "ANTHROPIC_CUSTOM_HEADERS",
		"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
		"CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_MANTLE", "MLAI_API_KEY", "CLAUDECODE",
	}
	p.KeyVar = "ANTHROPIC_AUTH_TOKEN"
	return p
}

// environ returns parent without the removed and the replaced variables (names
// compare case-insensitively), plus the ones the plan sets.
func (p launchPlan) environ(parent []string) []string {
	drop := make(map[string]bool, len(p.Remove)+len(p.Set))
	for _, n := range p.Remove {
		drop[strings.ToUpper(n)] = true
	}
	for _, kv := range p.Set {
		name, _, _ := strings.Cut(kv, "=")
		drop[strings.ToUpper(name)] = true
	}
	out := make([]string, 0, len(parent)+len(p.Set))
	for _, kv := range parent {
		name, _, _ := strings.Cut(kv, "=")
		if !drop[strings.ToUpper(name)] {
			out = append(out, kv)
		}
	}
	out = append(out, p.Set...)
	// The bridge is local: never send its key through a proxy.
	for _, name := range []string{"NO_PROXY", "no_proxy"} {
		v := "127.0.0.1,localhost,::1"
		for i := len(out) - 1; i >= 0; i-- {
			if n, old, _ := strings.Cut(out[i], "="); n == name || runtime.GOOS == "windows" && strings.EqualFold(n, name) {
				if old != "" {
					v = old + "," + v
				}
				out = append(out[:i], out[i+1:]...)
			}
		}
		out = append(out, name+"="+v)
		if runtime.GOOS == "windows" { // names are case-insensitive there
			break
		}
	}
	return out
}

var plainArg = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

func displayArg(a string) string {
	if plainArg.MatchString(a) {
		return a
	}
	return strconv.Quote(a)
}

// printDry shows what would run: the command line, and variable names only
// (the key variable as ***).
func (p launchPlan) printDry(w io.Writer, client string, rest []string) {
	parts := []string{displayArg(client)}
	for _, a := range append(append([]string(nil), p.Args...), rest...) {
		parts = append(parts, displayArg(a))
	}
	fmt.Fprintf(w, "command: %s\n", strings.Join(parts, " "))
	fmt.Fprintln(w, "env set:")
	for _, kv := range p.Set {
		name, _, _ := strings.Cut(kv, "=")
		if name == p.KeyVar {
			name += "=***"
		}
		fmt.Fprintf(w, "  %s\n", name)
	}
	fmt.Fprintln(w, "env removed:")
	for _, n := range p.Remove {
		fmt.Fprintf(w, "  %s\n", n)
	}
}

// bridgeURL validates a bridge address and returns its root (no /v1, query or
// fragment). Plain http is only accepted for loopback hosts.
func bridgeURL(raw string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || u.Hostname() == "" || u.User != nil || (u.Scheme != "http" && u.Scheme != "https") {
		return "", errors.New("invalid bridge URL (want http://host:port or https://host)")
	}
	if u.Scheme == "http" && !isLoopbackHost(u.Hostname()) {
		return "", fmt.Errorf("refusing plain http to non-loopback host %q: the API key would be sent in clear; use https", u.Hostname())
	}
	path := strings.TrimSuffix(strings.TrimRight(u.Path, "/"), "/v1")
	return u.Scheme + "://" + u.Host + path, nil
}

// resolveBridgeURL picks flag value, then MLAI_BRIDGE_URL, then the default.
func resolveBridgeURL(flagVal string) (string, error) {
	raw := flagVal
	if raw == "" {
		raw = envStr("MLAI_BRIDGE_URL", defaultBridgeURL)
	}
	return bridgeURL(raw)
}

type healthDoc struct {
	Status   string `json:"status"`
	Version  string `json:"version"`
	Uptime   int    `json:"uptime_s"`
	SignedIn bool   `json:"signed_in"`
}

func getJSON(rawURL, key string, timeout time.Duration, v any) error {
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(v)
}

// checkHealth asks GET <base>/healthz (2s) and requires a bridge-shaped answer.
func checkHealth(base string) (healthDoc, error) {
	var h healthDoc
	if err := getJSON(base+"/healthz", "", 2*time.Second, &h); err != nil {
		return h, fmt.Errorf("no healthy bridge at %s: %w", base, err)
	}
	if h.Version == "" || h.Status == "" {
		return h, fmt.Errorf("%s/healthz does not look like an mlai-bridge", base)
	}
	return h, nil
}

// startWrapServer runs a private bridge on 127.0.0.1 with a random key. Its log
// goes to ~/mlai-bridge/wrap.log. The returned stop shuts everything down.
func startWrapServer() (base, key string, stop func(), err error) {
	raw := make([]byte, 32)
	if _, err = rand.Read(raw); err != nil {
		return "", "", nil, err
	}
	key = hex.EncodeToString(raw)
	home, err := os.UserHomeDir()
	if err != nil {
		return "", "", nil, err
	}
	dir := filepath.Join(home, "mlai-bridge")
	if err = os.MkdirAll(dir, 0o700); err != nil {
		return "", "", nil, err
	}
	f, err := os.OpenFile(filepath.Join(dir, "wrap.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return "", "", nil, err
	}
	f.Chmod(0o600)
	logger = log.New(f, "", log.LstdFlags|log.LUTC)

	cfg.Bind, cfg.Port, cfg.APIKey = "127.0.0.1", 0, key
	s := newServer()
	srv := newHTTPServer(s)
	ln, err := net.Listen("tcp", srv.Addr)
	if err != nil {
		s.keeper.Close()
		f.Close()
		return "", "", nil, err
	}
	go srv.Serve(ln)
	base = "http://" + ln.Addr().String()
	logger.Printf("mlai-bridge %s (wrap) on %s model=%s", version, base, cfg.DefaultModel)
	stop = func() {
		s.shutdown(srv)
		f.Close()
	}
	return base, key, stop, nil
}

// findClient looks the client up in PATH, preferring a real .exe on Windows.
func findClient(name string) (string, error) {
	if runtime.GOOS == "windows" {
		if p, err := lookPath(name + ".exe"); err == nil {
			return p, nil
		}
	}
	return lookPath(name)
}

// cmdUnsafe are the characters cmd.exe interprets inside a .cmd/.bat call.
const cmdUnsafe = "\"&|<>^%!\r\n"

var npmShimRe = regexp.MustCompile(`"%dp0%\\([^"%]+\.[cm]?js)"`)

// npmShimScript finds the node script an npm .cmd shim runs, and a node to run
// it with. It returns empty strings when the shim has another shape.
func npmShimScript(shim string) (node, script string) {
	raw, err := os.ReadFile(shim)
	if err != nil {
		return "", ""
	}
	m := npmShimRe.FindSubmatch(raw)
	if m == nil {
		return "", ""
	}
	dir := filepath.Dir(shim)
	script = filepath.Join(dir, filepath.FromSlash(strings.ReplaceAll(string(m[1]), `\`, "/")))
	if st, err := os.Stat(script); err != nil || st.IsDir() {
		return "", ""
	}
	node = filepath.Join(dir, "node.exe")
	if st, err := os.Stat(node); err != nil || st.IsDir() {
		if node, err = lookPath("node"); err != nil {
			return "", ""
		}
	}
	return node, script
}

// prepareClient makes path and args safe to start. A .cmd/.bat file goes
// through cmd.exe, so it is replaced by a sibling .exe or the npm shim's node
// script when possible, and otherwise refused for arguments cmd.exe would misread.
func prepareClient(path string, args []string) (string, []string, error) {
	ext := strings.ToLower(filepath.Ext(path))
	if ext != ".cmd" && ext != ".bat" {
		return path, args, nil
	}
	exe := strings.TrimSuffix(path, filepath.Ext(path)) + ".exe"
	if st, err := os.Stat(exe); err == nil && !st.IsDir() {
		return exe, args, nil
	}
	if node, script := npmShimScript(path); script != "" {
		return node, append([]string{script}, args...), nil
	}
	for _, a := range args {
		if strings.ContainsAny(a, cmdUnsafe) {
			return "", nil, fmt.Errorf("%s is a .cmd/.bat script and an argument contains one of the characters %q that cmd.exe would misread; install the native .exe or drop those characters", filepath.Base(path), cmdUnsafe)
		}
	}
	return path, args, nil
}

// exitCode maps a Wait error to the shell convention: the child's code, or
// 128+n when a signal killed it.
func exitCode(err error) int {
	if err == nil {
		return 0
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		if ws, ok := ee.Sys().(syscall.WaitStatus); ok && ws.Signaled() {
			return 128 + int(ws.Signal())
		}
		if c := ee.ExitCode(); c >= 0 {
			return c
		}
	}
	return 1
}

// runChild runs the client with inherited stdio. SIGINT is left to the child;
// SIGTERM and SIGHUP are forwarded.
func runChild(path string, args, env []string, stderr io.Writer) int {
	cmd := exec.Command(path, args...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr

	sigs := make(chan os.Signal, 4)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)
	defer signal.Stop(sigs)
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(stderr, "mlai-bridge: cannot start %s: %v\n", filepath.Base(path), err)
		if errors.Is(err, exec.ErrNotFound) || errors.Is(err, os.ErrNotExist) {
			return 127
		}
		return 126
	}
	done := make(chan struct{})
	go func() {
		for {
			select {
			case sig := <-sigs:
				if sig != syscall.SIGINT {
					if cmd.Process.Signal(sig) != nil {
						cmd.Process.Kill()
					}
				}
			case <-done:
				return
			}
		}
	}()
	err := cmd.Wait()
	close(done)
	return exitCode(err)
}

// runLaunch implements `claude` and `codex`.
func runLaunch(kind string, args []string, stdout, stderr io.Writer) int {
	o, err := parseLaunchArgs(args)
	if err != nil {
		fmt.Fprintf(stderr, "mlai-bridge %s: %v\n", kind, err)
		return 2
	}
	initFlags(nil) // defaults from the environment, as serve does
	model := o.model
	if model == "" {
		model = cfg.DefaultModel
	}

	path, err := findClient(kind)
	if err != nil {
		if !o.dryRun {
			fmt.Fprintf(stderr, "mlai-bridge: %s not found in PATH\n", kind)
			return 127
		}
		path = kind
	}

	var base, key string
	remote := o.url
	if remote == "" {
		remote = os.Getenv("MLAI_BRIDGE_URL")
	}
	switch {
	case remote != "":
		if base, err = bridgeURL(remote); err != nil {
			fmt.Fprintf(stderr, "mlai-bridge %s: %v\n", kind, err)
			return 2
		}
		key = strings.TrimSpace(os.Getenv("MLAI_API_KEY"))
		if !o.dryRun {
			if _, err := checkHealth(base); err != nil {
				fmt.Fprintf(stderr, "mlai-bridge %s: %v\n", kind, err)
				return 1
			}
		}
	case o.dryRun:
		base = "http://127.0.0.1:<port>"
	default:
		var stop func()
		if base, key, stop, err = startWrapServer(); err != nil {
			fmt.Fprintf(stderr, "mlai-bridge %s: cannot start the bridge: %v\n", kind, err)
			return 1
		}
		defer stop()
	}

	plan := planLaunch(kind, base, key, model)
	if o.dryRun {
		plan.printDry(stdout, path, o.rest)
		return 0
	}
	path, cargs, err := prepareClient(path, append(append([]string(nil), plan.Args...), o.rest...))
	if err != nil {
		fmt.Fprintf(stderr, "mlai-bridge %s: %v\n", kind, err)
		return 2
	}
	return runChild(path, cargs, plan.environ(os.Environ()), stderr)
}

// ---------- env, models, status, doctor ----------

func quotePOSIX(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func quotePowerShell(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

// runEnv prints variables for a running bridge. The key is a reference to
// MLAI_API_KEY, never its value.
func runEnv(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("mlai-bridge env", flag.ContinueOnError)
	fs.SetOutput(stderr)
	urlFlag := fs.String("url", "", "bridge URL (default MLAI_BRIDGE_URL or "+defaultBridgeURL+")")
	ps := fs.Bool("powershell", false, "print $env: lines")
	model := fs.String("model", "", "model id (default MLAI_MODEL or pixis-opus)")
	fs.StringVar(model, "m", "", "model id")
	err := fs.Parse(args)
	var pos []string
	for err == nil && fs.NArg() > 0 {
		pos = append(pos, fs.Arg(0))
		err = fs.Parse(fs.Args()[1:])
	}
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if len(pos) != 1 || (pos[0] != "claude" && pos[0] != "codex") {
		fmt.Fprintln(stderr, "usage: mlai-bridge env <claude|codex> [--url URL] [--powershell] [-m ID]")
		return 2
	}
	return printEnv(pos[0], *urlFlag, *model, *ps, stdout, stderr)
}

func printEnv(kind, urlFlag, model string, powershell bool, stdout, stderr io.Writer) int {
	base, err := resolveBridgeURL(urlFlag)
	if err != nil {
		fmt.Fprintf(stderr, "mlai-bridge env: %v\n", err)
		return 2
	}
	if model == "" {
		model = envStr("MLAI_MODEL", "pixis-opus")
	}
	plan := planLaunch(kind, base, "", model)
	for _, name := range plan.Remove {
		switch {
		case name == "MLAI_API_KEY" || name == "CLAUDECODE":
		case powershell:
			fmt.Fprintf(stdout, "Remove-Item Env:%s -ErrorAction SilentlyContinue\n", name)
		default:
			fmt.Fprintf(stdout, "unset %s\n", name)
		}
	}
	for _, kv := range plan.Set {
		name, val, _ := strings.Cut(kv, "=")
		switch {
		case name == plan.KeyVar && powershell:
			fmt.Fprintf(stdout, "$env:%s = if ($env:MLAI_API_KEY) { $env:MLAI_API_KEY } else { 'none' }\n", name)
		case name == plan.KeyVar:
			fmt.Fprintf(stdout, "export %s=\"${MLAI_API_KEY:-none}\"\n", name)
		case powershell:
			fmt.Fprintf(stdout, "$env:%s = %s\n", name, quotePowerShell(val))
		default:
			fmt.Fprintf(stdout, "export %s=%s\n", name, quotePOSIX(val))
		}
	}
	if len(plan.Args) > 0 {
		parts := make([]string, 0, len(plan.Args))
		for _, a := range plan.Args {
			parts = append(parts, displayArg(a))
		}
		fmt.Fprintf(stdout, "# then run: %s %s\n", kind, strings.Join(parts, " "))
	}
	return 0
}

// bridgeFlags parses the shared --url flag of models and status.
func bridgeFlags(name string, args []string, stderr io.Writer) (string, int) {
	fs := flag.NewFlagSet("mlai-bridge "+name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	urlFlag := fs.String("url", "", "bridge URL (default MLAI_BRIDGE_URL or "+defaultBridgeURL+")")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return "", 0
		}
		return "", 2
	}
	base, err := resolveBridgeURL(*urlFlag)
	if err != nil {
		fmt.Fprintf(stderr, "mlai-bridge %s: %v\n", name, err)
		return "", 2
	}
	return base, -1
}

func runModels(args []string, stdout, stderr io.Writer) int {
	base, code := bridgeFlags("models", args, stderr)
	if code >= 0 {
		return code
	}
	var doc struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	key := strings.TrimSpace(os.Getenv("MLAI_API_KEY"))
	if err := getJSON(base+"/v1/models", key, 5*time.Second, &doc); err != nil {
		fmt.Fprintf(stderr, "mlai-bridge models: %v\n", err)
		return 1
	}
	for _, m := range doc.Data {
		fmt.Fprintln(stdout, m.ID)
	}
	return 0
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func runStatus(args []string, stdout, stderr io.Writer) int {
	base, code := bridgeFlags("status", args, stderr)
	if code >= 0 {
		return code
	}
	h, err := checkHealth(base)
	if err != nil {
		fmt.Fprintf(stderr, "mlai-bridge status: down: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "mlai-bridge %s at %s: %s, signed in: %s, uptime %ds\n",
		h.Version, base, h.Status, yesNo(h.SignedIn), h.Uptime)
	return 0
}

// runDoctor prints yes/no for each check and fails when mlai is missing or
// signed out.
func runDoctor(stdout io.Writer) int {
	mlaiPath, err := lookPath(envStr("MLAI_BIN", "mlai"))
	mlaiOK := err == nil
	row := func(label string, ok bool) {
		fmt.Fprintf(stdout, "%-18s %s\n", label+":", yesNo(ok))
	}
	row("mlai in PATH", mlaiOK)
	for _, name := range []string{"claude", "codex"} {
		_, err := lookPath(name)
		row(name+" in PATH", err == nil)
	}
	signedIn := false
	if mlaiOK {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, mlaiPath, "whoami")
		cmd.Env = childEnv()
		cmd.WaitDelay = 5 * time.Second
		out, _ := cmd.CombinedOutput()
		signedIn = strings.HasPrefix(strings.TrimSpace(string(out)), "Signed in")
	}
	row("mlai signed in", signedIn)
	if !mlaiOK || !signedIn {
		return 1
	}
	return 0
}
