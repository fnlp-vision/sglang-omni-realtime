"""StagePayload adapters and token streaming for MOSS-VL realtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.moss_vl_realtime.batch_adapter import RUNTIME_STATE_ATTR
from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimeRuntimeState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in real-time video analysis. "
    "The video streams to you frame by frame. At every frame, you decide "
    "independently whether to respond or stay silent - output `<|silence|>` "
    "when nothing relevant has happened, and respond when the visual content "
    "warrants it."
)


@dataclass
class MossVLRealtimeRequestData(SGLangARRequestData):
    runtime_state: MossVLRealtimeRuntimeState | None = None
    initial_input_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    ready_emitted: bool = False


def make_moss_vl_realtime_scheduler_adapters(
    *,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[
    Callable[[StagePayload], MossVLRealtimeRequestData],
    Callable[[MossVLRealtimeRequestData], StagePayload],
]:
    eos_token_id = int(tokenizer.eos_token_id)
    vocab_size = max(int(getattr(tokenizer, "vocab_size", 0)), len(tokenizer))

    def request_builder(payload: StagePayload) -> MossVLRealtimeRequestData:
        params = payload.request.params or {}
        source = _request_source(payload)
        session_id = str(source.get("session_id") or payload.request_id)
        system_prompt = str(source.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        initial_prompt = str(source.get("initial_prompt") or source.get("prompt") or "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            if "input_ids" not in encoded:
                raise ValueError("chat template output is missing input_ids")
            input_ids = encoded["input_ids"]
        else:
            input_ids = encoded
        input_ids = torch.as_tensor(input_ids, dtype=torch.long).flatten().tolist()
        request_max_new_tokens = int(params.get("max_new_tokens", max_new_tokens))
        if request_max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        temperature = float(params.get("temperature", 0.0) or 0.0)
        sampling_params = SamplingParams(
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            top_p=float(params.get("top_p", 1.0)),
            top_k=int(params.get("top_k", -1)),
            stop_token_ids=[eos_token_id],
            ignore_eos=True,
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )
        state = MossVLRealtimeRuntimeState(
            request_id=payload.request_id,
            session_id=session_id,
        )
        setattr(req, RUNTIME_STATE_ATTR, state)
        if params.get("benchmark_ignore_eos"):
            # Benchmark mode: keep ignore_eos=True even after the final extend so
            # the request decodes exactly max_new_tokens without an EOS stop.
            req._moss_vl_realtime_keep_ignore_eos = True
        return MossVLRealtimeRequestData(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            req=req,
            runtime_state=state,
            initial_input_ids=input_ids,
            max_new_tokens=request_max_new_tokens,
            temperature=temperature,
            stage_payload=payload,
        )

    def result_adapter(data: MossVLRealtimeRequestData) -> StagePayload:
        payload = data.stage_payload
        text = _decode(tokenizer, data.generated_token_ids)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "session_id": data.runtime_state.session_id,
                "completion_tokens": len(data.generated_token_ids),
                "finish_reason": data.finish_reason,
                "modality": "text",
            },
        )

    return request_builder, result_adapter


def make_moss_vl_realtime_stream_output_builder(
    *,
    tokenizer: Any,
) -> Callable[[str, MossVLRealtimeRequestData, Any], list[OutgoingMessage]]:
    eos_token_id = int(tokenizer.eos_token_id)

    def build(
        request_id: str,
        data: MossVLRealtimeRequestData,
        req_output: Any,
    ) -> list[OutgoingMessage]:
        messages: list[OutgoingMessage] = []
        processed_event = getattr(
            data.req,
            "_moss_vl_realtime_processed_event",
            None,
        )
        if processed_event is not None:
            processed_type = (
                "input.frame.processed"
                if processed_event.get("frame_ref") is not None
                else "input.prompt.processed"
            )
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": processed_type,
                        "seq_no": processed_event["seq_no"],
                        "timestamp": processed_event["timestamp"],
                        "final": processed_event["final"],
                        "modality": "control",
                    },
                    metadata={"modality": "control"},
                )
            )
            del data.req._moss_vl_realtime_processed_event
        token_data = req_output.data
        if token_data is None:
            return messages
        token_id = int(token_data)
        if not data.ready_emitted:
            messages.append(
                OutgoingMessage(
                    request_id=request_id,
                    type="stream",
                    data={
                        "event": "session.ready",
                        "session_id": data.runtime_state.session_id,
                        "modality": "control",
                    },
                    metadata={"modality": "control"},
                )
            )
            data.ready_emitted = True
        if token_id != eos_token_id:
            data.generated_token_ids.append(token_id)
        full_text = _decode(tokenizer, data.generated_token_ids)
        delta = _text_delta(data.emitted_text, full_text)
        data.emitted_text = full_text
        payload = data.stage_payload
        if not delta or not (payload.request.params or {}).get("stream", False):
            return messages
        messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data={"text": delta, "modality": "text"},
                metadata={"modality": "text", "token_id": token_id},
            )
        )
        return messages

    return build


def _request_source(payload: StagePayload) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if isinstance(payload.request.inputs, dict):
        source.update(payload.request.inputs)
    elif isinstance(payload.request.inputs, str):
        source["initial_prompt"] = payload.request.inputs
    if isinstance(payload.data, dict):
        source.update(payload.data)
    return source


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _text_delta(previous: str, current: str) -> str:
    if current.startswith(previous):
        return current[len(previous) :]
    common = 0
    limit = min(len(previous), len(current))
    while common < limit and previous[common] == current[common]:
        common += 1
    return current[common:]
