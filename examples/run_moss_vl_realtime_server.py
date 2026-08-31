#!/usr/bin/env python3
"""Launch the MOSS-VL realtime pipeline and binary-frame WebSocket API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_python_bin_on_path() -> None:
    """Expose venv console scripts to FlashInfer JIT subprocesses."""
    python_bin = str(Path(sys.executable).parent)
    path_entries = [
        entry
        for entry in os.environ.get("PATH", "").split(os.pathsep)
        if entry and entry != python_bin
    ]
    os.environ["PATH"] = os.pathsep.join([python_bin, *path_entries])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids for TP deployment, one GPU per rank.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=0.40)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--parked-request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=1,
        help="Maximum concurrent realtime sessions (one live request each). "
        "Default 1; values above 1 enable multi-session serving.",
    )
    parser.add_argument(
        "--enable-decode-cuda-graph",
        dest="decode_cuda_graph",
        action="store_true",
        default=True,
        help="Capture CUDA graphs for stable-shape decode steps "
        "(frame extend stays eager). Default on; validated in P11.",
    )
    parser.add_argument(
        "--disable-decode-cuda-graph",
        dest="decode_cuda_graph",
        action="store_false",
        help="Fall back to eager decode.",
    )
    parser.add_argument(
        "--decode-attention-backend",
        default=None,
        help="Optional server_args override for the decode attention backend "
        "(e.g. flashinfer, to match a decode-graph run in comparisons). "
        "Must be flashinfer when decode CUDA graph is on.",
    )
    parser.add_argument(
        "--enable-async-decode",
        dest="enable_async_decode",
        action="store_true",
        default=False,
        help="Launch decode step N+1 before resolving step N (lookahead). "
        "Default off; validated in P10.5.",
    )
    parser.add_argument(
        "--enable-benchmark-mode",
        action="store_true",
        default=False,
        help="Allow benchmark-only WebSocket options such as "
        "benchmark_ignore_eos. Never enable for production traffic.",
    )
    parser.add_argument(
        "--disable-startup-warmup",
        action="store_true",
        help="Skip the default internal frame warmup for diagnostics.",
    )
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be at least 1")
    if args.max_running_requests < 1:
        parser.error("--max-running-requests must be at least 1")
    if args.tp_size > 1:
        if args.gpus is None:
            parser.error("--tp-size > 1 requires --gpus")
        try:
            args.gpus = [int(value.strip()) for value in args.gpus.split(",")]
        except ValueError:
            parser.error("--gpus must be a comma-separated list of integers")
        if len(args.gpus) != args.tp_size:
            parser.error("--gpus must contain exactly --tp-size GPU ids")
        if len(set(args.gpus)) != len(args.gpus):
            parser.error("--gpus must not contain duplicate GPU ids")
    elif args.gpus is not None:
        parser.error("--gpus only applies when --tp-size > 1; use --gpu for TP=1")
    if args.decode_cuda_graph and args.decode_attention_backend not in (
        None,
        "flashinfer",
    ):
        # fa3 decode-graph replay overflows req_to_token rows for the
        # encoder-prefix KV layout (see perf_p10_3/server_graph_blocking.log).
        parser.error(
            "--decode-attention-backend must be flashinfer when decode CUDA "
            "graph is enabled"
        )
    return args


def main() -> None:
    args = parse_args()
    _ensure_python_bin_on_path()
    from sglang_omni.models.moss_vl_realtime.config import MossVLRealtimePipelineConfig
    from sglang_omni.serve import launch_server

    config = MossVLRealtimePipelineConfig(model_path=args.model_path)
    stage = config.stages[0]
    stage.gpu = args.gpus if args.tp_size > 1 else args.gpu
    stage.tp_size = args.tp_size
    stage.parallelism.tp = args.tp_size
    factory_args = dict(stage.factory_args)
    factory_args.update(
        {
            "device": "cuda:0" if args.tp_size > 1 else f"cuda:{args.gpu}",
            "mem_fraction_static": args.mem_fraction_static,
            "context_length": args.context_length,
            "max_new_tokens": args.max_new_tokens,
            "parked_request_timeout_s": args.parked_request_timeout,
            "max_running_requests": args.max_running_requests,
            "disable_cuda_graph": not args.decode_cuda_graph,
            "page_size": 1,
            "enable_async_decode": args.enable_async_decode,
        }
    )
    server_args_overrides = dict(factory_args.get("server_args_overrides") or {})
    if args.decode_attention_backend is not None:
        server_args_overrides.update(
            {
                # Setting any single backend dimension stops the upstream MossVL
                # override from injecting its flashinfer prefill default; pin both.
                "prefill_attention_backend": "flashinfer",
                "decode_attention_backend": args.decode_attention_backend,
            }
        )
    if server_args_overrides:
        factory_args["server_args_overrides"] = server_args_overrides
    stage.factory_args = factory_args
    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name="moss-vl-realtime",
        video_realtime_warmup=not args.disable_startup_warmup,
        video_realtime_benchmark_mode=args.enable_benchmark_mode,
    )


if __name__ == "__main__":
    main()
