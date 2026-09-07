"""Portable single-GPU serving and TF/SGLang validation entrypoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common import (
    CONFIG,
    HERE,
    ROOT,
    Child,
    environment,
    free_port,
    gpu_inventory,
    prepare_cases,
    select_gpus,
    server_command,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("serve", "test", "worker"):
        command = sub.add_parser(mode)
        command.add_argument(
            "model_path",
            type=Path,
            help="Local MOSS-VL-Realtime-SGLANG model directory",
        )
        if mode != "worker":
            command.add_argument(
                "--gpus",
                help="Physical GPU indices/UUIDs; default selects one idle GPU",
            )
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="Print settings without GPU discovery or model loading",
            )
        if mode == "serve":
            command.add_argument("--host", default=CONFIG["host"])
            command.add_argument("--port", type=int, default=CONFIG["port"])
        else:
            command.add_argument("--output-dir", type=Path, required=mode == "worker")
        if mode == "worker":
            command.add_argument("--backend", choices=("tf", "sglang"), required=True)
    return parser.parse_args(argv)


def validate_model(path):
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(
            "Provide a downloaded local model directory containing config.json"
        )
    config = json.loads((path / "config.json").read_text())
    if not any("MossVL" in name for name in config.get("architectures", [])):
        raise ValueError("Expected a MOSS-VL checkpoint")
    return path


def execution_waves(gpus):
    if len(gpus) == 1:
        return [[("tf", gpus[0])], [("sglang", gpus[0])]]
    if len(gpus) == 2:
        return [[("tf", gpus[0]), ("sglang", gpus[1])]]
    raise ValueError("Tests accept one GPU, or two GPUs for parallel TF/SGLang workers")


def report(output, worker_codes, metadata):
    rows = []
    for backend in ("tf", "sglang"):
        path = output / f"{backend}.json"
        records = []
        if path.exists():
            records = json.loads(path.read_text())
            rows.extend(records)
        if not records or (
            worker_codes.get(backend) != 0
            and all(r["status"] == "PASS" for r in records)
        ):
            rows.append(
                dict(
                    backend=backend,
                    status="FAIL",
                    phase="worker",
                    sessions=0,
                    case="worker incomplete",
                    error=f"Exit code {worker_codes.get(backend)}",
                )
            )
    passed = bool(rows) and all(r["status"] == "PASS" for r in rows)
    write_json(
        output / "summary.json", dict(passed=passed, metadata=metadata, results=rows)
    )
    lines = [
        "# MOSS-VL Realtime Validation",
        "",
        "TF: one shared model with independent KV states, round-robin execution. ",
        "SGLang: independent WebSocket sessions with native continuous batching.",
        "",
        "Functional rows use identical bundled inputs. Output text is compared for review, ",
        "not as a bitwise/logit parity or semantic accuracy assertion. Session duration includes input pacing. ",
        "This suite does not include Demo memory, ASR or TTS.",
        "",
        f"GPU: {', '.join(r['name'] + ' / ' + str(r['total_mib']) + ' MiB' for r in metadata['gpus'])}",
        f"Code revision: `{metadata['revision']}`",
        "",
        "| Backend | Phase | Sessions | Case / Lane | Status | Frames | Seconds | Process VRAM sample peak (GiB) | KV peak (GiB) |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]

    def number(value):
        return "-" if value is None else f"{value:.2f}"

    for r in rows:
        lines.append(
            f"| {r['backend']} | {r['phase']} | {r['sessions']} | {r['case']} / {r.get('lane', '-')} | {r['status']} | "
            f"{r.get('frames', 0)}/{r.get('expected_frames', 0)} | {number(r.get('elapsed_seconds'))} | "
            f"{number(r.get('process_peak_gib'))} | {number(r.get('kv_peak_gib'))} |"
        )
    lines.extend(
        [
            "",
            "Process VRAM includes weights, preallocated pools and runtime allocations; it is not per-session VRAM. ",
            "KV peak is the shared SGLang pool usage during the whole group. TF KV is managed by its native cache. ",
            "The window phase requires observed visual eviction and full KV recovery after session closure.",
            "",
            "## Output Comparison",
            "",
        ]
    )
    for case in ("cars", "drawing"):
        lines.extend([f"### {case}", ""])
        for r in rows:
            if r.get("case") == case and r["phase"] == "comparison":
                lines.extend(
                    [
                        f"**{r['backend']} / {r['sessions']} session(s) / lane {r.get('lane', 0)}**",
                        "",
                        "```text",
                        r.get("text", "").replace("```", "` ` `")
                        or "(no visible output)",
                        "```",
                        "",
                    ]
                )
    lines.extend(
        [
            "## Artifacts",
            "",
            "`summary.json`: metadata and per-session results. ",
            "`tf.json`, `sglang.json`: model outputs and timing details. ",
            "`allocator.jsonl`: SGLang KV and process-memory samples. ",
            "`tf.log`, `sglang.log`, `server.log`: execution logs.",
            "",
        ]
    )
    if any(r.get("error") or r.get("status") == "FAIL" for r in rows):
        lines.extend(["## Errors", ""])
        for r in rows:
            if r.get("status") == "FAIL":
                reason = (
                    r.get("error")
                    or ", ".join(k for k, v in r.get("checks", {}).items() if not v)
                    or "See per-session details"
                )
                lines.append(
                    f"- {r['backend']} / {r['sessions']} / {r['case']} / lane {r.get('lane', '-')}: {str(reason).replace(chr(10), ' ')}"
                )
    (output / "report.md").write_text("\n".join(lines))
    return passed


def run_tests(args, gpus):
    output = args.output_dir or ROOT / "results/moss_vl_realtime" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:6]
    )
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = prepare_cases(output)
    write_json(output / "cases.json", cases)
    metadata = dict(
        config=CONFIG,
        gpus=gpus,
        model_path=str(args.model_path),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        versions={
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "sglang", "websockets")
        },
        delivery_sources={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in HERE.iterdir()
            if path.suffix in (".py", ".sh", ".json")
        },
        inputs={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "tests/data/cars.jpg", ROOT / "tests/data/draw.mp4")
        },
    )
    write_json(output / "metadata.json", metadata)
    print(f"Results: {output}", flush=True)
    children, codes = [], {}
    try:
        for wave in execution_waves(gpus):
            active = []
            for backend, gpu in wave:
                command = [
                    sys.executable,
                    "-u",
                    str(HERE / "entry.py"),
                    "worker",
                    str(args.model_path),
                    "--backend",
                    backend,
                    "--output-dir",
                    str(output),
                ]
                child = Child(
                    command, environment(gpu["uuid"]), output / f"{backend}.log"
                )
                children.append(child)
                active.append((backend, child))
                print(
                    f'Starting {backend} on GPU {gpu["index"]}; see {output / (backend + ".log")}',
                    flush=True,
                )
            for backend, child in active:
                try:
                    codes[backend] = child.wait(CONFIG["worker_timeout_seconds"])
                finally:
                    child.stop()
                    children.remove(child)
                print(f"{backend}: exit {codes[backend]}", flush=True)
    finally:
        for child in reversed(children):
            child.stop()
        passed = report(output, codes, metadata)
        print(f'Report: {output / "report.md"}', flush=True)
    return 0 if passed else 1


def main(argv=None):
    args = parse_args(argv)
    args.model_path = validate_model(args.model_path)
    if args.mode == "worker":
        cases = json.loads((args.output_dir / "cases.json").read_text())
        if args.backend == "tf":
            from tf_reference import run
        else:
            from sglang_validation import run
        rows = run(args.model_path, cases, args.output_dir)
        return 0 if rows and all(r["status"] == "PASS" for r in rows) else 1
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    mode=args.mode,
                    model_path=str(args.model_path),
                    config=CONFIG,
                    gpus=args.gpus or "one idle GPU",
                    single_gpu_default=True,
                ),
                indent=2,
            )
        )
        return 0
    if importlib.metadata.version("transformers") != "5.12.1":
        raise ValueError(
            "Activate the repository backend environment (Transformers 5.12.1)"
        )
    gpus = select_gpus(
        args.gpus, gpu_inventory(), os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if args.mode == "test":
        return run_tests(args, gpus)
    if len(gpus) != 1:
        raise ValueError(
            "start.sh launches a single-GPU instance; provide at most one --gpus value"
        )
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    free_port(args.host, args.port)
    command = server_command(args.model_path, args.port, args.host)
    print(
        f'GPU {gpus[0]["index"]}; {CONFIG["max_sessions"]} sessions; http://{args.host}:{args.port}/health',
        flush=True,
    )
    os.execve(sys.executable, command, environment(gpus[0]["uuid"]))


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
