"""Pipeline stage factory for MOSS-VL realtime."""

from __future__ import annotations

from typing import Any


def create_sglang_moss_vl_realtime_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 1,
    max_new_tokens: int = 4096,
    context_length: int = 262144,
    mem_fraction_static: float | None = None,
    server_args_overrides: dict[str, Any] | None = None,
    parked_request_timeout_s: float = 300.0,
    disable_cuda_graph: bool = True,
    page_size: int = 1,
    enable_async_decode: bool = False,
):
    from sglang_omni.models.moss_vl_realtime.engine_builder import (
        MossVLRealtimeEngineBuilder,
    )

    return MossVLRealtimeEngineBuilder(
        max_running_requests=max_running_requests,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
        parked_request_timeout_s=parked_request_timeout_s,
        disable_cuda_graph=disable_cuda_graph,
        page_size=page_size,
        enable_async_decode=enable_async_decode,
    ).build(
        model_path,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


__all__ = ["create_sglang_moss_vl_realtime_executor"]
