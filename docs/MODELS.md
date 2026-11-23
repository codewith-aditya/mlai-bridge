# Model catalogue

Ids are the values you send as `model` in `/v1/chat/completions` or `/v1/messages`.
They come from the ML AI Code CLI — this table mirrors what `mlai models` and
`mlai -p hi -m not-a-model` report for an Infinite account (checked 2026-11-23).

`GET /v1/models` on the bridge returns exactly these ids.

## Unlimited on the Infinite plan

| Id | Display name (CLI) | Notes | Measured (direct mode) |
| --- | --- | --- | --- |
| `pixis-opus` | **Opus 5.5R** (default) | Strongest for hard coding work; supports `reasoning_effort` low/medium/high/max | TTFT 0.7–7.8 s · 60–135 tok/s |
| `pixis` | **Pixis** | Fast, tuned for coding tasks | TTFT 1.3 s · ~150 tok/s |
| `pixis-sol` | **GPT 6.1R Sol** | Very fast, very large context | TTFT 5.5 s · ~70 tok/s |
| `ml-ai-forge` | **ML.ai Forge** | ML.ai's own model, text only (no images) | TTFT 0.8 s · ~170 tok/s |

## Spend plan credits

| Id | Display name (CLI) | Notes | Measured (direct mode) |
| --- | --- | --- | --- |
| `gpt-6-astra` | **GPT 6 Astra** | Most capable GPT model | TTFT 0.7 s · ~150 tok/s |
| `grok-4.7` | **Grok 4.7** | xAI reasoning model | not benchmarked |
| `mlai-coder` | **ML.ai Coder** | Strong agentic coder | not benchmarked |
| `glm-5.3` | **GLM 5.3** | Full GLM 5.3, text only | TTFT 0.6 s · ~140 tok/s |
| `claude-sonnet-5.5` | **Claude Sonnet 5.5** | Balanced Claude model | TTFT 0.8 s · ~155 tok/s |
| `claude-fable-5.2` | **Claude Fable 5.2** | Claude Fable from Anthropic | not benchmarked |

"Spend credits" models are billed from the 50 credits the free month includes, then at
list price — the bridge does not stop you from using them, it only reports usage.

## Aliases and routing

- `main` — the gateway's routing alias. If you send `model: "main"` straight to
  `code.ml.ai/api/model/v1/chat/completions`, the gateway picks a model for you (observed
  routing to `glm-5.3-flash`). The bridge does **not** expose `main`; send a real id.
- The response `model` field is whatever the gateway actually ran (e.g. `glm-5.3-flash`
  when `main` is used), not necessarily the id you sent.
- Display-name aliases such as `Opus 5.5R` are CLI UI labels; the API only accepts ids.

## Finding ids after a CLI update

```bash
mlai -p hi -m not-a-model --mode ask   # error lists every valid id
mlai models                            # display names + plan notes
```

Then update `docs/MODELS.md` and the `ids` slice in `handleModels` (`handlers.go`).
