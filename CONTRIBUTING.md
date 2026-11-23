# Contributing

Thanks for the interest. This is a small Go bridge with two execution paths, so the rules
are narrow and the test surface is small.

## Ground rules

- **Never commit credentials.** `~/.mlai` (device token, run tokens), `.env` files, HAR
  files and logs stay out of git. `.gitignore` covers the common cases.
- **One account, one bridge.** PRs that add account pooling, key rotation across users, or
  anything meant to resell plan access are out of scope and will be closed.
- **Keep it dependency-free.** The bridge is Go stdlib only (`go.mod` has no requirements).
  Adding a framework or an SDK needs a strong argument.
- **Never break the fallback.** Direct mode may fail (run tokens expire); the CLI path has
  to keep working. Any change to `direct.go` must prove the fallback still answers.

## Development

```bash
git clone https://github.com/<you>/mlai-bridge && cd mlai-bridge
go build -o mlai-bridge .
go test ./...

# the bridge shells out to the CLI; sign in once on the host
npm install -g mlai-code
mlai login

./mlai-bridge -port 8197                 # direct mode (default)
./mlai-bridge -port 8197 -direct=false   # CLI-only mode
```

Pre-PR checks:

```bash
gofmt -l .        # must print nothing
go vet ./...
go test ./...
go build ./...
```

Smoke test against a running bridge:

```bash
curl -sS http://127.0.0.1:8197/healthz
curl -sS http://127.0.0.1:8197/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"pixis-opus","messages":[{"role":"user","content":"say ok"}]}'
curl -sS -N http://127.0.0.1:8197/v1/messages -H 'Content-Type: application/json' \
  -d '{"model":"pixis-opus","max_tokens":32,"stream":true,"messages":[{"role":"user","content":"say ok"}]}'
```

## What is worth a PR

| Area | Examples |
| --- | --- |
| Direct mode | keeper health, faster token refresh, `409` backoff, per-model keeper runs |
| Compatibility | OpenAI `stream_options`, `tool_choice`, Anthropic `thinking` blocks, `system` arrays |
| Reliability | request cancellation killing the CLI child, circuit breaker, structured logs |
| Docs | model catalogue updates, measured numbers with a reproducible command |
| Bench | more models, longer runs, TTFT breakdown (first delta vs first content) |

## Adding or updating a model

Model ids come from the CLI, not from us. Get the authoritative list with:

```bash
mlai -p hi -m not-a-model --mode ask      # prints the valid ids
mlai models                               # prints display names + plan notes
```

Then update `docs/MODELS.md` (id, display name, plan, notes) **and** the `ids` slice in
`handleModels` (`handlers.go`). Keep the order stable.

## Reporting bugs

Include:

- commit (`git rev-parse --short HEAD`) and mode (`-direct=true/false`)
- `curl -sS http://127.0.0.1:8197/healthz`
- the request body (redact prompts) and the response, with status + `X-Request-Id`
- relevant lines from `journalctl -u mlai-bridge` or the log file
- if direct mode is involved: whether the keeper run was fresh, and whether
  `409 run_finished` appeared in the logs

Never paste a device token or an `mlrun_…` token into an issue.

## Style

- Errors: return the provider's own error shape with a stable code (see `errors.go`); never
  swallow a non-2xx from upstream.
- Commits: `area: short imperative summary` (`direct: refresh keeper on 409`).
- Docs that claim a number must point at the command that produced it.

## License

By contributing you agree your work is licensed under the MIT license in [LICENSE](LICENSE).
