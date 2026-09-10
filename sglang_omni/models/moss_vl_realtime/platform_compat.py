"""Platform-aware defaults for the MOSS-VL realtime pipeline.

The pipeline was built for CUDA (FlashInfer attention, decode CUDA graphs).
On Ascend NPU the equivalents resolve through ``sglang_omni.platforms``:
the ascend attention backend replaces FlashInfer, decode graphs stay off
by default, and the upstream Moss-VL FlashInfer guard is relaxed.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

CUDA_ATTENTION_BACKEND = "flashinfer"
NPU_ATTENTION_BACKEND = "ascend"

_FLASHINFER_GUARD_MESSAGE = (
    "MossVLForConditionalGeneration requires flashinfer prefill "
    "attention backend for cross-attention custom mask support."
)


def is_npu_platform() -> bool:
    from sglang_omni.platforms import current_platform

    return current_platform.device_type == "npu"


def platform_device_type() -> str:
    from sglang_omni.platforms import current_platform

    return current_platform.device_type


def device_spec(index: int) -> str:
    """Return the concrete single-card device spec for a placement index."""
    return f"{platform_device_type()}:{int(index)}"


def preferred_attention_backend() -> str:
    """Attention backend serving cross-attention on the resolved platform.

    FlashInfer is the CUDA decode backend (it also re-plans the packed
    cross-attention custom mask on every replay). The ascend backend is the
    NPU equivalent registered in sglang's attention registry.
    """
    return NPU_ATTENTION_BACKEND if is_npu_platform() else CUDA_ATTENTION_BACKEND


def recommended_deploy_params(tp_size: int) -> dict[str, Any]:
    """Recommended serving defaults for one instance of this topology.

    The deployment script consumes this instead of embedding per-platform
    numbers. CUDA keeps the historical defaults (32768 context, 0.60 static
    memory share); NPU TP>=2 needs a smaller context — moss_vl's KV is
    ~3.4 MB/token and a TP=2 pair cannot hold 32K sessions — plus a higher
    static-memory share for the doubled per-card weights.
    """
    if is_npu_platform() and int(tp_size) >= 2:
        return {"context_length": 8192, "mem_fraction": 0.80}
    return {"context_length": 32768, "mem_fraction": 0.60}


def relax_mossvl_flashinfer_guard() -> None:
    """Allow non-FlashInfer attention backends for Moss-VL on NPU.

    Upstream pins ``MossVLForConditionalGeneration`` to the FlashInfer
    prefill backend because only that backend consumes the packed
    cross-attention custom mask. On NPU the ascend backend serves
    cross-attention through native SDPA without the frame-level mask (a
    known behavioral difference), so the CUDA-only guard is relaxed there
    and left untouched on CUDA. Idempotent.
    """
    if not is_npu_platform():
        return
    from sglang.srt.server_args import ServerArgs

    for method_name in (
        "_handle_model_specific_adjustments",
        "_handle_attention_backend_compatibility",
    ):
        original = getattr(ServerArgs, method_name, None)
        if original is None or getattr(original, "_moss_npu_relaxed", False):
            continue

        @functools.wraps(original)
        def wrapper(self: Any, _original=original) -> None:
            try:
                _original(self)
            except AssertionError as exc:
                if _FLASHINFER_GUARD_MESSAGE not in str(exc):
                    raise
                logger.warning(
                    "NPU: relaxed the Moss-VL FlashInfer attention guard; the "
                    "ascend backend serves cross-attention without the "
                    "frame-level visibility mask."
                )

        wrapper._moss_npu_relaxed = True
        setattr(ServerArgs, method_name, wrapper)


__all__ = [
    "CUDA_ATTENTION_BACKEND",
    "NPU_ATTENTION_BACKEND",
    "device_spec",
    "is_npu_platform",
    "platform_device_type",
    "preferred_attention_backend",
    "recommended_deploy_params",
    "relax_mossvl_flashinfer_guard",
]
