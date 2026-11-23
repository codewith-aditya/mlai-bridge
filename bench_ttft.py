#!/usr/bin/env python3
"""Measure TTFT and TPS for a streaming request through mlai-bridge."""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8197/v1/chat/completions"
PROMPT = ("Write a detailed technical explanation of how TLS 1.3 handshakes work, "
          "covering key exchange, certificates, and session resumption. About 350 words.")


def run(model: str) -> dict:
    import secrets
    body = json.dumps({
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": PROMPT + f" (run {secrets.token_hex(4)})"}],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    first_text = None
    last_text = None
    chars = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content") or ""
                if delta:
                    now = time.time()
                    chars += len(delta)
                    if first_text is None:
                        first_text = now
                    last_text = now
    total = time.time() - t0
    ct = usage.get("completion_tokens") or 0
    pt = usage.get("prompt_tokens") or 0
    ttft = (first_text - t0) if first_text else None
    decode_window = (last_text - first_text) if (first_text and last_text) else 0
    return {
        "model": model,
        "ttft_s": round(ttft, 2) if ttft else None,
        "total_s": round(total, 2),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "chars": chars,
        "tps_end_to_end": round(ct / total, 1) if total and ct else None,
        "tps_decode_window": round(ct / decode_window, 1) if decode_window > 0.01 else None,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    }


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "pixis-opus"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for i in range(runs):
        print(json.dumps(run(model)), flush=True)
        time.sleep(2)
