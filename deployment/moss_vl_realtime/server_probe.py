"""Launch the standard server with read-only validation telemetry."""

import json
import os
import time
from pathlib import Path
from types import MethodType

from common import ROOT, load_module


def create(
    model_path,
    device="cuda:0",
    gpu_id=0,
    tp_rank=0,
    tp_size=1,
    nccl_port=None,
    **kwargs,
):
    import torch

    from sglang_omni.models.moss_vl_realtime.stages import (
        create_sglang_moss_vl_realtime_executor,
    )
    from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

    scheduler = create_sglang_moss_vl_realtime_executor(
        model_path,
        device=device,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
        **kwargs,
    )
    original = scheduler.get_next_batch_to_run
    path = Path(os.environ["MOSS_DELIVERY_TRACE"])

    def observed(self):
        batch = original()
        if time.monotonic() < getattr(self, "_delivery_next_sample", 0):
            return batch
        self._delivery_next_sample = time.monotonic() + 1
        allocator = self.token_to_kv_pool_allocator
        states = []
        for rid in self.realtime_sessions._sessions:
            data = self._find_request_data(rid)
            state = getattr(data, "runtime_state", None)
            if state is not None:
                states.append(
                    dict(
                        decoder=state.decoder_length,
                        encoder=state.encoder_length,
                        history=state.effective_appended_encoder_length,
                    )
                )
        row = dict(
            time=time.time(),
            sessions=len(self.realtime_sessions._sessions),
            pool_size=int(allocator.size),
            free_kv=int(allocator.available_size()),
            process_bytes=get_process_gpu_memory_bytes(gpu_id),
            torch_reserved=torch.cuda.memory_reserved(gpu_id),
            states=states,
        )
        with path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        return batch

    scheduler.get_next_batch_to_run = MethodType(observed, scheduler)
    return scheduler


if __name__ == "__main__":
    import sglang_omni.serve as serve

    original_launch = serve.launch_server

    def launch(config, *args, **kwargs):
        config.stages[0].factory = "server_probe.create"
        return original_launch(config, *args, **kwargs)

    serve.launch_server = launch
    load_module(
        "delivery_launcher", ROOT / "examples/run_moss_vl_realtime_server.py"
    ).main()
