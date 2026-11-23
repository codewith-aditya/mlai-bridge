# Benchmarks

All numbers below were produced through the bridge itself, streaming, one request at a
time, on a 4 vCPU AMD EPYC VPS with US egress. Reproduce with the command at the bottom.

## Methodology

- **TTFT** — time from `POST` to the first SSE delta of any kind (`reasoning_content`
  or `content`). Reasoning models emit thinking tokens first, so first *content* is later
  (see the second table).
- **TPS** — `usage.completion_tokens / total_wall_seconds` (end-to-end, not a decode-only
  estimate). Reasoning tokens count.
- **Prompt** — "Explain in about 150 words how TCP congestion control works, with one
  example." plus a nonce so the bridge cache never serves it.
- One run per model, `stream: true`, default effort. Variance between runs is real; treat
  ±30 % as noise.

## Direct mode (real streaming, gateway token)

| Model | TTFT | Total | Prompt tok | Completion tok | TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `glm-5.3` | **0.64 s** | 9.33 s | 43 | 1322 | 141.7 |
| `gpt-6-astra` | **0.66 s** | 7.42 s | 43 | 1127 | 151.9 |
| `claude-sonnet-5.5` | **0.76 s** | 8.52 s | 43 | 1334 | 156.5 |
| `ml-ai-forge` | **0.79 s** | 10.57 s | 42 | 1812 | 171.5 |
| `pixis` | **1.32 s** | 9.61 s | 43 | 1434 | 149.2 |
| `pixis-sol` | 5.47 s | 13.66 s | 43 | 951 | 69.6 |
| `pixis-opus` | 7.81 s | 15.42 s | 41 | 964 | 62.5 |

Earlier `pixis-opus` runs on a warm keeper measured **0.66–0.94 s TTFT** and 111–135 TPS;
the 7.81 s above includes a keeper refresh in the same request.

Note the prompt token count: **~43**, not ~15 000. Direct mode skips the CLI agent harness,
so you pay for your prompt only.

### First delta vs first content (pixis-opus)

| Run | first delta | first content | total | tokens | TPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm | 0.94 s | 4.86 s | 6.28 s | 848 | 135 |
| warm | 0.66 s | 0.80 s | 2.43 s | 270 | 111 |

## CLI mode (fallback, `-direct=false`)

The CLI buffers the whole answer before emitting its `answer` event, so TTFT equals total
time; the bridge then splits the text into chunks for the client.

| Model | TTFT (= total) | Completion tok | TPS |
| --- | ---: | ---: | ---: |
| `pixis-opus` | 15.0–16.2 s | 595–744 | 39.6–46.4 |
| `pixis` | 18.6 s | 794–841 | 42.7–45.3 |
| `pixis-sol` | 25.4 s / 77.8 s | 1541–1590 | 19.8 / 62.6 |

Fixed overhead in CLI mode is ~10.6 s per call (harness start + ~14.7 k prompt tokens).

## Reproducing

```bash
# direct mode numbers
python3 bench_ttft.py pixis-opus 3

# one-liner for any model
curl -sS -N http://127.0.0.1:8197/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","stream":true,"messages":[{"role":"user","content":"count to 50"}]}'
```

`bench_ttft.py` reports `ttft_s` (first content), `total_s`, tokens and both an
end-to-end and a decode-window TPS.
