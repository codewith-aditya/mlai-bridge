# mlai-bridge

OpenAI- and Anthropic-compatible HTTP API in front of the **ML AI Code** CLI (`mlai`).

```
OpenAI client ─┐
               ├─► mlai-bridge :8197 ──► `mlai -p … --output-format stream-json` ──► code.ml.ai
Claude Code  ──┘
```

You keep using the ML AI Code subscription you already pay for; the bridge turns the
agent CLI into `/v1/chat/completions` and `/v1/messages` so any SDK, editor or gateway
can talk to it.

## Features

- `POST /v1/chat/completions` — OpenAI Chat Completions (stream + non-stream), tool calls
- `POST /v1/messages` — Anthropic Messages (stream + non-stream), `tool_use` blocks
- `POST /v1/messages/count_tokens` — token estimate
- `POST /v1/responses` (Codex; stateless)
- `GET /v1/models`, `GET /healthz` (status, version, `uptime_s`, `signed_in`; sign-in state
  is cached for 60 s), `GET /metrics`
- **Direct mode** (default): talks to `code.ml.ai/api/model/v1/chat/completions` with a
  live run token, so streams are real token deltas (TTFT < 1 s, ~110–160 tok/s) with no
  agent-harness prompt. Falls back to the CLI automatically if the gateway rejects the run.
- Token accounting in every response — `prompt/input`, `completion/output`, `total`,
  `cached_tokens`; also emitted as a final usage chunk on streams
- Identical-request cache (`-cache-ttl`, default 10m) with `X-Cache: HIT|MISS`
- Bounded concurrency with a queue (`-concurrency`, `-queue-wait`)
- Stable error codes mapped from CLI failures (see below)
- Circuit-free, dependency-free single binary (Go stdlib only)

## Direct mode and the run keeper

The gateway accepts a per-run bearer token (`mlrun_…`) that only the CLI can mint, and it
dies when the run ends. The bridge therefore keeps **one background CLI run alive**
(`mlai --bg -y "Run the shell command \`sleep 5400\`…"` in `<workdir>/.keeper`) and reuses
that run's token from `~/.mlai/sessions/<id>/engine/mcp.json`. When the gateway answers
`409 run_finished`, the bridge opens a fresh keeper run and retries the request once.
Keepers start one at a time, at least 60 s apart (longer after repeated failures), and a keeper
whose token the gateway rejected is stopped before a new one starts.

Direct requests are limited by `-direct-concurrency` (default 8) and by `-timeout`, the same
wall-clock limit the CLI path uses.

```
client ─► bridge ─► code.ml.ai/api/model/v1/chat/completions   (direct: real SSE)
                 └► mlai --bg (keeper: keeps the run token alive)
```

Disable it with `-direct=false` to always use the CLI path (slower: ~10–16 s to first
token, because the CLI buffers the whole answer).

## Models

Ten ids, as reported by the CLI for an Infinite account. Full table (display names, plan,
aliases): **[docs/MODELS.md](docs/MODELS.md)**.

| Id | What it is | Plan | TTFT / TPS (measured) |
| --- | --- | --- | --- |
| `pixis-opus` | Opus 5.5R — strongest, default | **unlimited** | 0.7–7.8 s · 60–135 |
| `pixis` | Pixis — fast coding | **unlimited** | 1.3 s · ~150 |
| `pixis-sol` | GPT 6.1R Sol — fast, large context | **unlimited** | 5.5 s · ~70 |
| `ml-ai-forge` | ML.ai Forge — text only | **unlimited** | 0.8 s · ~170 |
| `gpt-6-astra` | GPT 6 Astra | credits | 0.7 s · ~150 |
| `grok-4.7` | Grok 4.7 | credits | — |
| `mlai-coder` | ML.ai Coder | credits | — |
| `glm-5.3` | GLM 5.3 — text only | credits | 0.6 s · ~140 |
| `claude-sonnet-5.5` | Claude Sonnet 5.5 | credits | 0.8 s · ~155 |
| `claude-fable-5.2` | Claude Fable 5.2 | credits | — |

## Benchmarks

Measured 2026-11-23 through this bridge, streaming, one request at a time (4 vCPU VPS,
US egress). Method and the CLI-mode comparison: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

| Model | TTFT | Total | Prompt tok | Completion tok | TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `glm-5.3` | 0.64 s | 9.33 s | 43 | 1322 | 141.7 |
| `gpt-6-astra` | 0.66 s | 7.42 s | 43 | 1127 | 151.9 |
| `claude-sonnet-5.5` | 0.76 s | 8.52 s | 43 | 1334 | 156.5 |
| `ml-ai-forge` | 0.79 s | 10.57 s | 42 | 1812 | 171.5 |
| `pixis` | 1.32 s | 9.61 s | 43 | 1434 | 149.2 |
| `pixis-sol` | 5.47 s | 13.66 s | 43 | 951 | 69.6 |
| `pixis-opus` | 7.81 s | 15.42 s | 41 | 964 | 62.5 |

The prompt column is the point of direct mode: ~43 tokens instead of the CLI's ~14 700-token
harness prompt. `pixis-opus` on a warm keeper has measured **0.66–0.94 s TTFT** and
111–135 TPS; the row above includes a keeper refresh.

## Docs

- [docs/MODELS.md](docs/MODELS.md) — ids, display names, plan, aliases, how to refresh the list
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — methodology, direct vs CLI numbers
- [CONTRIBUTING.md](CONTRIBUTING.md) — rules, dev loop, how to add a model

## Requirements

- Go 1.26+ to build
- Node.js 20+ and the ML AI Code CLI on the host, signed in once: `mlai login`
- A plan that covers the models you call (Infinite covers Pixis/Forge without a meter)

## Install

```bash
git clone https://github.com/<you>/mlai-bridge && cd mlai-bridge
go build -o mlai-bridge .
npm install -g mlai-code      # the CLI the bridge shells out to
mlai login                    # browser/device sign-in, once
```

### Headless sign-in (VPS, no local browser)

`mlai login` listens on a random localhost port and prints an auth URL. On a remote box:

```bash
ssh -L <port>:127.0.0.1:<port> user@host   # keep this open
# then run `mlai login` on the host and open the printed URL on your laptop
```

## Run

```bash
./mlai-bridge                       # 127.0.0.1:8197, no key needed
./mlai-bridge -port 8197 -model pixis-opus -concurrency 2 -cache-ttl 15m
```

Without a key the bridge only starts on a loopback address, only accepts requests whose
`Host` header is `localhost` or a loopback IP (otherwise `403 forbidden_host`), and requires
`Content-Type: application/json` on every POST (otherwise `415 unsupported_media_type`).

### Systemd

The repo ships a hardened unit, `mlai-bridge.service`. It runs as a dedicated `mlai` user
with state in `/var/lib/mlai-bridge` and reads the key from an environment file instead of
the command line.

```bash
sudo useradd --system --home-dir /var/lib/mlai-bridge --shell /usr/sbin/nologin mlai
sudo install -d -m 0755 /etc/mlai-bridge
sudo install -m 0600 -o root -g root /dev/null /etc/mlai-bridge/env
echo "MLAI_API_KEY=$(openssl rand -hex 24)" | sudo tee /etc/mlai-bridge/env >/dev/null
sudo npm install --prefix /opt/mlai mlai-code
sudo install -D -m 0755 mlai-bridge /opt/mlai-bridge/mlai-bridge
sudo install -m 0644 mlai-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now mlai-bridge
```

Key points of the unit:

- `EnvironmentFile=/etc/mlai-bridge/env` (root:root, `0600`) holds `MLAI_API_KEY=...`.
- `User=mlai`, `StateDirectory=mlai-bridge` (`0700`), `HOME=/var/lib/mlai-bridge`, `UMask=0077`.
  Run `mlai login` as that user (with the same `HOME`) once, so `~/.mlai` lives there.
- `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `PrivateDevices`, no capabilities,
  `NoNewPrivileges`. `MemoryDenyWriteExecute` and `RestrictNamespaces` are left off on
  purpose: Node needs JIT and the CLI sandbox needs namespaces.
- Logs go to the journal: `journalctl -u mlai-bridge`.

## Run Claude Code or Codex through the bridge

```bash
ml-wrap claude                          # default model
ml-wrap claude -m pixis -- -p "fix the failing test"
ml-wrap codex -m pixis-sol
ml-wrap codex -- exec "explain this repo"
ml-wrap claude --dry-run                # show the command and variable names only
```

`ml-wrap` is the same binary (`make install` links it); `mlai-bridge claude` works too. On
Windows copy `mlai-bridge.exe` to `ml-wrap.exe`. The launcher starts a private bridge on
`127.0.0.1` with a random key, runs the client with it and stops the bridge when the client
exits (log: `~/mlai-bridge/wrap.log`). It sets variables (and `-c` options for Codex) for that
process only: no Claude Code or Codex file is changed. Use `--url` or `MLAI_BRIDGE_URL`, with
the key in `MLAI_API_KEY`, for a bridge that is already running.

| Command | What it does |
| --- | --- |
| `serve` | run the bridge (also the default without a command) |
| `claude`, `codex` | start the client against a bridge; arguments after `--` go to the client |
| `env <claude\|codex>` | print the variables for a running bridge (`--powershell` on Windows) |
| `models`, `status` | models and health of a running bridge |
| `doctor` | check `mlai`, `claude`, `codex` and the sign-in |
| `version`, `help` | |

Limits: an `env` block in `~/.claude/settings.json` overrides these variables. Codex shows a
"model metadata not found" warning and Claude Code an "unrecognized model" one; both continue.
Agent requests are larger than the 30,000 characters the CLI accepts, so these clients need
direct mode: the CLI fallback only helps for short prompts. Server-side tools such as web
search are not available.

## Client setup

OpenAI SDK / any compatible client:

```
base_url = http://127.0.0.1:8197/v1
api_key  = anything when no key is set on the bridge; otherwise the bridge's MLAI_API_KEY
model    = pixis-opus
```

Claude Code / Anthropic clients:

```
ANTHROPIC_BASE_URL = http://127.0.0.1:8197
ANTHROPIC_API_KEY  = anything when no key is set on the bridge; otherwise the bridge's MLAI_API_KEY
```

```bash
curl http://127.0.0.1:8197/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"pixis-opus",
  "messages":[{"role":"user","content":"say hi"}]
}'
```

## Configuration

| Flag / env | Default | Meaning |
| --- | --- | --- |
| `-port` `MLAI_PORT` | `8197` | listen port |
| `-mlai` `MLAI_BIN` | `mlai` | CLI path |
| `-workdir` `MLAI_WORKDIR` | `~/mlai-bridge/workspace` | folder the agent runs in |
| `-model` `MLAI_MODEL` | `pixis-opus` | default model |
| `-effort` `MLAI_EFFORT` | *empty* | `low\|medium\|high\|max` |
| `-concurrency` `MLAI_CONCURRENCY` | `1` | parallel CLI runs |
| `-direct-concurrency` `MLAI_DIRECT_CONCURRENCY` | `8` | parallel direct-mode requests |
| `-queue-wait` | `30s` | how long a request waits for a slot |
| `-timeout` | `10m` | wall clock per request, CLI and direct |
| `-cache-ttl` | `10m` | identical-request cache (`0` disables) |
| `-cache-size` | `512` | cache entries |
| `-bind` `MLAI_BIND` | `127.0.0.1` | listen address; a non-loopback address requires `MLAI_API_KEY` (or `-api-key`), otherwise the bridge refuses to start |
| `-api-key` `MLAI_API_KEY` | *empty* | require `Authorization: Bearer <key>` (or `x-api-key`, `api-key`) on `/v1/*`; `/healthz` and `/metrics` stay open. Prefer the env var: `-api-key` is visible in process lists |
| `-direct` `MLAI_DIRECT` | `true` | gateway direct mode; `false` = CLI only |

## Exposing it beyond localhost

Both shapes below need a key. Without one the bridge only binds to loopback, rejects any
`Host` header that is not `localhost` or a loopback IP, and requires
`Content-Type: application/json` on POST. A reverse proxy forwards a public `Host`, so
without a key every proxied request would fail with `403 forbidden_host`, and a bridge that
accepted it unauthenticated would be open to anyone who can reach the proxy. Set
`MLAI_API_KEY` on the bridge and have clients send it.

**Behind your own reverse proxy (recommended).** Keep the bridge on loopback with
`MLAI_API_KEY` set and add a TLS location:

```nginx
location ^~ /mlai/ {
    proxy_pass http://127.0.0.1:8197/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_buffering off;         # SSE must not be buffered
    proxy_cache off;
    proxy_read_timeout 3600s;
    client_max_body_size 8m;
}
```

Then clients use `https://your.host/mlai/v1` with the key.

**Direct bind.** `MLAI_API_KEY=<key> ./mlai-bridge -bind 0.0.0.0` — the bridge refuses to
start on a non-loopback address without a key. Prefer a firewall allowlist over the open
internet.

Generate a key and keep it out of git. Pass it through the environment, not `-api-key`,
which shows up in process lists:

```bash
(umask 077 && openssl rand -hex 24 > .apikey)
MLAI_API_KEY="$(cat .apikey)" ./mlai-bridge -bind 0.0.0.0
```

Auth accepts `Authorization: Bearer <key>`, `x-api-key: <key>` or `api-key: <key>`.
A key in the query string is not accepted. Wrong/missing key returns `401 invalid_api_key` in the provider's error shape. CORS preflight is answered for browser clients.

## Error codes

Errors are returned in the provider's native shape with a stable code:

| HTTP | code | when |
| --- | --- | --- |
| 400 | `invalid_request` | malformed JSON, missing `messages` |
| 400 | `upgrade_required` | CLI too old (`mlai update`) |
| 401 | `not_signed_in` | `mlai login` missing/expired |
| 403 | `forbidden_host` | no key set and the `Host` header is not `localhost` or a loopback IP |
| 404 | `model_not_found` | unknown model id |
| 415 | `unsupported_media_type` | POST without `Content-Type: application/json` (enforced when no key is set) |
| 429 | `concurrent` | no free slot within `-queue-wait` |
| 429 | `rate_limited` | upstream session/hourly ceiling |
| 429 | `daily_cap` | allowance/credits exhausted |
| 502 | `upstream_error` | CLI failed for another reason |
| 503 | `unavailable` | CLI missing/not executable |
| 504 | `timeout` | request exceeded `-timeout` |

`429` responses carry `Retry-After`; every response carries `X-Request-Id`.

Error messages sent to clients are fixed text. Raw CLI output appears only in the bridge
logs.

## Token accounting

Non-stream (OpenAI):

```json
"usage": {"prompt_tokens": 14619, "completion_tokens": 85,
          "total_tokens": 14704, "prompt_tokens_details": {"cached_tokens": 6144}}
```

Stream: a final chunk with `"choices": []` and the same `usage` object, then `[DONE]`.

Anthropic: `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`.

## Tool calls

Direct mode passes tools and tool results to the gateway and returns its tool calls as they
are. The rest of this section is about the CLI path.

The CLI has no native function-calling channel, so the bridge emulates it: tool
definitions are injected into the prompt and the model is asked to answer with
`{"tool_calls":[{"name":"…","arguments":{…}}]}`. The bridge parses that into OpenAI
`tool_calls` (`finish_reason: "tool_calls"`) or Anthropic `tool_use` blocks
(`stop_reason: "tool_use"`). Only tools the request declared are accepted; a call to any
other name is not turned into a tool call. It works in practice; it is not a wire-level
guarantee, so validate arguments on your side.

## Notes and limits

- The CLI runs the full agent harness: expect ~14k prompt tokens of system context per
  call. The Infinite plan covers Pixis/Forge without a meter; other models spend credits.
- `--mode ask` is used so the agent answers instead of editing files. The prompt is sent to
  the CLI on stdin.
- Streaming granularity is limited by the CLI: it emits the final answer as one event, so
  the bridge splits that text into chunks. Time-to-first-token equals the model's
  time-to-answer. Direct mode streams real deltas.
- The bridge listens on loopback by default. A non-loopback bind requires `MLAI_API_KEY`;
  for remote use see "Exposing it beyond localhost".

## License

MIT
