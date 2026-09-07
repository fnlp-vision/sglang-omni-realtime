"""Bounded WS checks using the repository's realtime stability client."""

import asyncio
import json
import os
import time
import urllib.request
from pathlib import Path

from common import (
    CONFIG,
    ROOT,
    Child,
    events_for,
    free_port,
    groups,
    initial_prompt,
    load_module,
    server_command,
    write_json,
)


async def lane(url, case, count, index, phase):
    import websockets

    client = load_module(
        "delivery_stability", ROOT / "scripts/moss_vl_realtime_stability_sglang.py"
    )
    session = None
    error = None
    started = time.monotonic()
    frames = (
        CONFIG["stability_frames"] if phase == "window" else CONFIG["comparison_frames"]
    )
    trace = events_for(case, frames)
    try:
        ws = await websockets.connect(
            url, max_size=64 * 1024 * 1024, open_timeout=30, close_timeout=5
        )
        session = client.Session(ws)
        await session.wait_for(
            "session.created", timeout=CONFIG["event_timeout_seconds"]
        )
        await session.send_json(
            dict(
                type="session.configure",
                prompt=initial_prompt(case),
                system_prompt="You are a helpful visual assistant.",
                temperature=0,
                max_new_tokens=CONFIG["max_new_tokens"],
                max_tokens_per_turn=CONFIG["tokens_per_second"],
                input_queue_capacity=4,
                include_usage=True,
            )
        )
        await session.wait_for("session.ready", timeout=CONFIG["event_timeout_seconds"])
        ready = time.monotonic()
        for event in trace:
            await asyncio.sleep(max(0, ready + event["timestamp"] - time.monotonic()))
            seq = event["seq_no"]
            cursor = len(session.events)
            if event["type"] == "frame":
                await session.push_frame(
                    seq,
                    event["timestamp"],
                    Path(event["frame_path"]).read_bytes(),
                    "image/jpeg",
                    CONFIG["event_timeout_seconds"],
                )
                kind = "input.frame.accepted"
            else:
                await session.send_json(
                    dict(
                        type="input.prompt",
                        seq_no=seq,
                        prompt=event["prompt"],
                        final=True,
                    )
                )
                kind = "input.prompt.accepted"
            await session.wait_for(
                kind, seq_no=seq, after=cursor, timeout=CONFIG["event_timeout_seconds"]
            )
        await session.wait_for("session.done", timeout=CONFIG["final_timeout_seconds"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        events = list(session.events) if session else []
        if session:
            if not any(e["type"] == "session.done" for e in events):
                try:
                    await session.send_json({"type": "session.abort"})
                except Exception:
                    pass
            await session.close()
    processed = [e["seq_no"] for e in events if e["type"] == "input.frame.processed"]
    prompt_processed = [
        e["seq_no"] for e in events if e["type"] == "input.prompt.processed"
    ]
    text = "".join(e.get("delta", "") for e in events)
    accepted = {
        e["seq_no"]: e["_arrived"]
        for e in events
        if e["type"] == "input.frame.accepted"
    }
    errors = [e for e in events if e["type"] == "error"]
    checks = dict(
        no_transport_error=not error and not errors,
        ordered_frames=processed == list(range(frames)),
        prompt_processed=prompt_processed == [frames],
        visible_output=bool(text.strip()),
        session_done=sum(e["type"] == "session.done" for e in events) == 1,
    )
    passed = all(checks.values())
    return dict(
        backend="SGLang",
        scheduling="continuous_batching",
        sessions=count,
        lane=index,
        case=case["case_id"],
        phase=phase,
        status="PASS" if passed else "FAIL",
        error=error,
        errors=errors,
        checks=checks,
        frames=len(processed),
        expected_frames=frames,
        elapsed_seconds=time.monotonic() - started,
        text=text,
        events=events,
        accepted_to_processed_seconds=[
            e["_arrived"] - accepted[e["seq_no"]]
            for e in events
            if e["type"] == "input.frame.processed" and e["seq_no"] in accepted
        ],
    )


def telemetry(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows


async def check_group(url, cases, count, phase, path):
    start = time.time()
    results = await asyncio.gather(
        *(lane(url, case, count, i, phase) for i, case in enumerate(cases))
    )
    deadline = time.monotonic() + 30
    clean = False
    rows = []
    while time.monotonic() < deadline:
        rows = [r for r in telemetry(path) if r["time"] >= start]
        if (
            rows
            and rows[-1]["time"] > time.time() - 3
            and rows[-1]["sessions"] == 0
            and rows[-1]["free_kv"] == rows[-1]["pool_size"]
        ):
            clean = True
            break
        await asyncio.sleep(1)
    evicted = any(s["history"] > s["encoder"] for r in rows for s in r["states"])
    process_samples = [
        r["process_bytes"] for r in rows if r["process_bytes"] is not None
    ]
    for result in results:
        result["checks"]["kv_recovered"] = clean
        if phase == "window":
            result["checks"]["window_eviction"] = evicted
        result.update(
            cleanup="PASS" if clean else "FAIL",
            window_eviction_observed=evicted,
            kv_peak_gib=max((r["pool_size"] - r["free_kv"] for r in rows), default=0)
            * 192
            / 1048576,
            process_peak_gib=max(process_samples) / 2**30 if process_samples else None,
        )
        if not clean or (phase == "window" and not evicted):
            result["status"] = "FAIL"
    return results


def run(model_path, cases, output):
    port = free_port()
    trace = output / "allocator.jsonl"
    env = dict(os.environ, MOSS_DELIVERY_TRACE=str(trace))
    server = Child(
        server_command(model_path, port, probe=True),
        env,
        output / "server.log",
        new_group=False,
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    try:
        deadline = time.monotonic() + CONFIG["startup_timeout_seconds"]
        while True:
            if server.process.poll() is not None:
                raise RuntimeError(f'SGLang server exited; see {output / "server.log"}')
            try:
                with opener.open(
                    f"http://127.0.0.1:{port}/health", timeout=2
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("SGLang startup timed out")
            time.sleep(1)
        url = f"ws://127.0.0.1:{port}/v1/video/realtime"
        for count in CONFIG["session_counts"]:
            for group in groups(cases, count):
                rows.extend(
                    asyncio.run(check_group(url, group, count, "comparison", trace))
                )
                write_json(output / "sglang.json", rows)
                print(f"SGLang sessions={count} completed", flush=True)
        group = [cases[i % len(cases)] for i in range(CONFIG["max_sessions"])]
        rows.extend(
            asyncio.run(
                check_group(url, group, CONFIG["max_sessions"], "window", trace)
            )
        )
        write_json(output / "sglang.json", rows)
    finally:
        server.stop()
    return rows
