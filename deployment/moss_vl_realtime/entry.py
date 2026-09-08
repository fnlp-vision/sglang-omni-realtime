"""Single-GPU MOSS-VL Realtime server launcher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import sys
import traceback
from pathlib import Path

from common import (
    CONFIG,
    environment,
    free_port,
    gpu_inventory,
    select_gpus,
    server_command,
    validate_model,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("model_path", type=Path)
    serve.add_argument("--gpus", help="Physical GPU index/UUID; default: one idle GPU")
    serve.add_argument("--dry-run", action="store_true")
    serve.add_argument("--host", default=CONFIG["host"])
    serve.add_argument("--port", type=int, default=CONFIG["port"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.model_path = validate_model(args.model_path)
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    model_path=str(args.model_path),
                    config=CONFIG,
                    host=args.host,
                    port=args.port,
                    gpus=args.gpus or "one idle GPU",
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
    if len(gpus) != 1:
        raise ValueError(
            "start.sh launches a single-GPU instance; provide at most one --gpus value"
        )
    free_port(args.host, args.port)
    command = server_command(args.model_path, args.port, args.host)
    print(
        f'GPU {gpus[0]["index"]}; {CONFIG["max_sessions"]} sessions; '
        f"http://{args.host}:{args.port}/health",
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
