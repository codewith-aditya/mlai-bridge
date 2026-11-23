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
- `GET /v1/models`, `GET /healthz`, `GET /metrics`
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

```
client ─► bridge ─► code.ml.ai/api/model/v1/chat/completions   (direct: real SSE)
                 └► mlai --bg (keeper: keeps the run token alive)
```

Disable it with `-direct=false` to always use the CLI path (slower: ~10–16 s to first
token, because the CLI buffers the whole answer).

## Requirements

- Go 1.22+ to build
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
./mlai-bridge                       # 127.0.0.1:8197
./mlai-bridge -port 8197 -model pixis-opus -concurrency 2 -cache-ttl 15m
```

### Systemd

```ini
[Unit]
Description=mlai-bridge
After=network-online.target

[Service]
ExecStart=/opt/mlai-bridge/mlai-bridge -port 8197 -workdir /var/lib/mlai-bridge/workspace
Environment=MLAI_MODEL=pixis-opus
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## Client setup

OpenAI SDK / any compatible client:

```
base_url = http://127.0.0.1:8197/v1
api_key  = anything (the bridge is local; put it behind your own proxy for remote use)
model    = pixis-opus
```

Claude Code / Anthropic clients:

```
ANTHROPIC_BASE_URL = http://127.0.0.1:8197
ANTHROPIC_API_KEY  = anything
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
| `-port` `MLAI_PORT` | `8197` | listen port, loopback only |
| `-mlai` `MLAI_BIN` | `mlai` | CLI path |
| `-workdir` `MLAI_WORKDIR` | `/root/mlai-bridge/workspace` | folder the agent runs in |
| `-model` `MLAI_MODEL` | `pixis-opus` | default model |
| `-effort` `MLAI_EFFORT` | *empty* | `low\|medium\|high\|max` |
| `-concurrency` `MLAI_CONCURRENCY` | `1` | parallel CLI runs |
| `-queue-wait` | `30s` | how long a request waits for a slot |
| `-timeout` | `10m` | wall clock per request |
| `-cache-ttl` | `10m` | identical-request cache (`0` disables) |
| `-cache-size` | `512` | cache entries |

## Error codes

Errors are returned in the provider's native shape with a stable code:

| HTTP | code | when |
| --- | --- | --- |
| 400 | `invalid_request` | malformed JSON, missing `messages` |
| 400 | `upgrade_required` | CLI too old (`mlai update`) |
| 401 | `not_signed_in` | `mlai login` missing/expired |
| 404 | `model_not_found` | unknown model id |
| 429 | `concurrent` | no free slot within `-queue-wait` |
| 429 | `rate_limited` | upstream session/hourly ceiling |
| 429 | `daily_cap` | allowance/credits exhausted |
| 502 | `upstream_error` | CLI failed for another reason |
| 503 | `unavailable` | CLI missing/not executable |
| 504 | `timeout` | request exceeded `-timeout` |

`429` responses carry `Retry-After`; every response carries `X-Request-Id`.

## Token accounting

Non-stream (OpenAI):

```json
"usage": {"prompt_tokens": 14619, "completion_tokens": 85,
          "total_tokens": 14704, "prompt_tokens_details": {"cached_tokens": 6144}}
```

Stream: a final chunk with `"choices": []` and the same `usage` object, then `[DONE]`.

Anthropic: `usage.input_tokens`, `usage.output_tokens`, `usage.cache_read_input_tokens`.

## Tool calls

The CLI has no native function-calling channel, so the bridge emulates it: tool
definitions are injected into the prompt and the model is asked to answer with
`{"tool_calls":[{"name":"…","arguments":{…}}]}`. The bridge parses that into OpenAI
`tool_calls` (`finish_reason: "tool_calls"`) or Anthropic `tool_use` blocks
(`stop_reason: "tool_use"`). It works in practice; it is not a wire-level guarantee, so
validate arguments on your side.

## Notes and limits

- The CLI runs the full agent harness: expect ~14k prompt tokens of system context per
  call. The Infinite plan covers Pixis/Forge without a meter; other models spend credits.
- `--mode ask` is used so the agent answers instead of editing files.
- Streaming granularity is limited by the CLI: it emits the final answer as one event, so
  the bridge splits that text into chunks. Time-to-first-token equals the model's
  time-to-answer.
- The bridge listens on loopback. If you expose it, put authentication in front.

## License

MIT
