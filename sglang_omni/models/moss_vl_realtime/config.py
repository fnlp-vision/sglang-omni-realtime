"""Pipeline configuration for MOSS-VL realtime."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.moss_vl_realtime"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="moss_vl_realtime",
            process="moss_vl_realtime",
            factory=f"{_PKG}.stages.create_sglang_moss_vl_realtime_executor",
            factory_args={
                "device": "cuda:0",
                "max_running_requests": 1,
                "max_new_tokens": 4096,
                "context_length": 32768,
                "mem_fraction_static": 0.25,
                "disable_cuda_graph": False,
                "page_size": 1,
                "enable_async_decode": False,
            },
            gpu=0,
            terminal=True,
        )
    ]


class MossVLRealtimePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "MossVLRealtimeForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MossVLForConditionalGeneration",
    )

    model_path: str
    entry_stage: str = "moss_vl_realtime"
    stages: list[StageConfig] = Field(default_factory=_stages)


EntryClass = MossVLRealtimePipelineConfig
