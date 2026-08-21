from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
from sglang_omni.models.moss_vl_realtime.config import (
    MossVLRealtimePipelineConfig,
)
from sglang_omni.models.moss_vl_realtime.request_builders import (
    make_moss_vl_realtime_scheduler_adapters,
    make_moss_vl_realtime_stream_output_builder,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    eos_token_id = 2
    vocab_size = 900

    def __len__(self) -> int:
        return 1000

    def apply_chat_template(self, messages, **kwargs):
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "Track the cup"}
        assert kwargs["add_generation_prompt"] is True
        return {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join({10: "A", 11: "B", 12: "C"}.get(i, "") for i in token_ids)


def _payload(*, stream: bool) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs={"initial_prompt": "Track the cup", "session_id": "session-1"},
            params={"stream": stream, "max_new_tokens": 20},
        ),
        data={},
    )


def test_request_builder_creates_pure_text_persistent_request() -> None:
    tokenizer = _Tokenizer()
    request_builder, result_adapter = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    data = request_builder(_payload(stream=False))

    assert data.initial_input_ids == [101, 102, 103]
    assert data.req.multimodal_inputs is None
    assert data.req.sampling_params.max_new_tokens == 20
    assert data.req.sampling_params.ignore_eos is True
    assert data.req.vocab_size == 1000
    state = getattr(data.req, RUNTIME_STATE_ATTR)
    assert state is data.runtime_state
    assert state.session_id == "session-1"

    data.generated_token_ids[:] = [10, 11]
    data.finish_reason = "stop"
    result = result_adapter(data)
    assert result.data["text"] == "AB"
    assert result.data["session_id"] == "session-1"
    assert result.data["completion_tokens"] == 2


def test_stream_builder_accumulates_sampled_tokens_not_injected_context() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    stream_builder = make_moss_vl_realtime_stream_output_builder(tokenizer=tokenizer)
    data.req._moss_vl_realtime_processed_event = {
        "seq_no": 3,
        "timestamp": 1.5,
        "frame_ref": "shm://frame",
        "final": False,
    }

    first = stream_builder("req-1", data, SimpleNamespace(data=10))
    second = stream_builder("req-1", data, SimpleNamespace(data=11))
    eos = stream_builder("req-1", data, SimpleNamespace(data=2))

    assert data.generated_token_ids == [10, 11]
    assert first[0].data == {
        "event": "input.frame.processed",
        "seq_no": 3,
        "timestamp": 1.5,
        "final": False,
        "modality": "control",
    }
    text_messages = [
        message for message in first + second if message.data.get("modality") == "text"
    ]
    assert [message.data["text"] for message in text_messages] == ["A", "B"]
    ready_messages = [
        message
        for message in first + second
        if message.data.get("event") == "session.ready"
    ]
    assert [message.data["session_id"] for message in ready_messages] == ["session-1"]
    assert eos == []


def test_stream_builder_emits_prompt_processed_control_event() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )
    data = request_builder(_payload(stream=True))
    data.req._moss_vl_realtime_processed_event = {
        "seq_no": 4,
        "timestamp": 2.0,
        "prompt": "How many?",
        "final": True,
    }
    stream_builder = make_moss_vl_realtime_stream_output_builder(tokenizer=tokenizer)

    messages = stream_builder("req-1", data, SimpleNamespace(data=None))

    assert messages[0].data["event"] == "input.prompt.processed"
    assert messages[0].data["seq_no"] == 4
    assert messages[0].data["final"] is True


def test_pipeline_config_targets_realtime_stage() -> None:
    config = MossVLRealtimePipelineConfig(model_path="/models/moss-vl")
    assert config.entry_stage == "moss_vl_realtime"
    assert config.stages[0].factory.endswith(
        "moss_vl_realtime.stages.create_sglang_moss_vl_realtime_executor"
    )


def test_request_builder_marks_benchmark_ignore_eos() -> None:
    tokenizer = _Tokenizer()
    request_builder, _ = make_moss_vl_realtime_scheduler_adapters(
        tokenizer=tokenizer,
        max_new_tokens=100,
    )

    payload = _payload(stream=True)
    payload.request.params["benchmark_ignore_eos"] = True
    benchmark_data = request_builder(payload)
    assert benchmark_data.req._moss_vl_realtime_keep_ignore_eos is True
    assert benchmark_data.req.sampling_params.ignore_eos is True

    default_data = request_builder(_payload(stream=True))
    assert not hasattr(default_data.req, "_moss_vl_realtime_keep_ignore_eos")
