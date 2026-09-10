#!/usr/bin/env python
"""Caller-protocol /v1/realtime latency probe.

One round = connect, start, batch N frames, collect output until the end
marker, send stop, close.  Reports per-round phases:

  connect   WebSocket handshake
  ready     start accepted → engine ready
  acks      all frame_ack received
  ttft      first output text byte (after last frame submission)
  total     connect → end marker received

Usage: perf_probe.py --url ws://host:port/v1/realtime --rounds N --frames K
       [--image path.jpg] [--json] [--prompt "..."]
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import statistics
import sys
import time

import websockets
from PIL import Image

DEFAULT_PROMPT = "请描述这些画面中正在发生的事情。"


def make_jpeg(width: int = 640, height: int = 384) -> bytes:
    img = Image.new("RGB", (width, height))
    for x in range(0, width, 32):
        for y in range(0, height, 32):
            c = ((x // 32) * 36 % 256, (y // 32) * 36 % 256, 128)
            for dx in range(32):
                for dy in range(32):
                    img.putpixel((x + dx, y + dy), c)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


async def one_round(url: str, frames: list[bytes], prompt: str, sampling: dict) -> dict:
    t0 = time.perf_counter()
    ws = await websockets.connect(url, max_size=32 << 20, open_timeout=10)
    connect_s = time.perf_counter() - t0

    start = {
        "type": "start",
        "prompt": prompt,
        "frame_queue_size": 32,
        "max_new_tokens": sampling.get("max_new_tokens", 512),
        "max_tokens_per_second": 160,
        "do_sample": False,
        "temperature": 0.2,
        "top_k": 20,
        "top_p": 0.8,
        "repetition_penalty": 1.05,
    }
    await ws.send(json.dumps(start, ensure_ascii=False))
    while True:
        m = json.loads(await ws.recv())
        if m.get("type") == "error":
            raise RuntimeError(m.get("message"))
        if m.get("type") == "ready":
            break
    ready_s = time.perf_counter() - t0

    last_ts = 0.0
    for i, data in enumerate(frames):
        last_ts = round(i * 1.0, 1)
        await ws.send(json.dumps({"type": "frame", "timestamp": last_ts}))
        await ws.send(data)
    acks = 0
    while acks < len(frames):
        m = json.loads(await ws.recv())
        if m.get("type") == "frame_ack":
            acks += 1
        elif m.get("type") == "error":
            raise RuntimeError(m.get("message"))
    acks_s = time.perf_counter() - t0

    text = ""
    ttft_s = None
    end_reason = None
    while True:
        m = json.loads(await ws.recv())
        if m.get("type") == "output":
            if ttft_s is None:
                ttft_s = time.perf_counter() - t0
            text += m.get("text", "")
            if "<|im_end|>" in text or "<|eot_id|>" in text or "<|endoftext|>" in text:
                end_reason = "end_marker"
                break
            if "<|silence|>" in text:
                end_reason = "silence"
                break
        elif m.get("type") == "error":
            raise RuntimeError(m.get("message"))
        if time.perf_counter() - t0 > 60:
            end_reason = "timeout"
            break

    await ws.send(json.dumps({"type": "stop"}))
    await ws.close()
    total_s = time.perf_counter() - t0
    return {
        "connect": connect_s,
        "ready": ready_s,
        "acks": acks_s,
        "ttft": ttft_s,
        "total": total_s,
        "text": text,
        "end_reason": end_reason,
        "chars": len(text.replace("<|im_end|>", "")),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--image", default=None, help="optional JPEG to send as frames")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.image:
        data = open(args.image, "rb").read()
        frames = [data] * args.frames
    else:
        frames = [make_jpeg() for _ in range(args.frames)]

    sampling = {"max_new_tokens": args.max_new_tokens}
    results = []
    failures = 0
    for r in range(args.rounds):
        try:
            res = await one_round(args.url, frames, args.prompt, sampling)
            results.append(res)
            if not args.json:
                print(
                    f"round {r + 1}/{args.rounds}: connect {res['connect'] * 1000:.0f}ms "
                    f"ready {res['ready'] * 1000:.0f}ms acks {res['acks'] * 1000:.0f}ms "
                    f"ttft {res['ttft'] * 1000 if res['ttft'] else -1:.0f}ms "
                    f"total {res['total'] * 1000:.0f}ms chars={res['chars']} "
                    f"end={res['end_reason']}"
                )
        except Exception as exc:
            failures += 1
            if not args.json:
                print(f"round {r + 1}/{args.rounds}: FAILED {exc}")

    if not results:
        print(json.dumps({"rounds": args.rounds, "failures": failures}, indent=2))
        return 1

    def stats(key: str) -> dict:
        vals = [r[key] * 1000 for r in results if r.get(key) is not None]
        if not vals:
            return {"p50": None, "p95": None, "max": None}
        vals.sort()
        return {
            "p50": round(vals[len(vals) // 2], 1),
            "p95": round(vals[min(len(vals) - 1, int(len(vals) * 0.95))], 1),
            "max": round(vals[-1], 1),
        }

    summary = {
        "url": args.url,
        "rounds": args.rounds,
        "failures": failures,
        "frames_per_round": args.frames,
        "total_ms": stats("total"),
        "ttft_ms": stats("ttft"),
        "ready_ms": stats("ready"),
        "acks_ms": stats("acks"),
        "chars": [r["chars"] for r in results],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
