"""Stateful binary-frame WebSocket API for MOSS-VL realtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from fastapi import FastAPI, WebSocket
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.websockets import WebSocketDisconnect, WebSocketState

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.models.moss_vl_realtime.frame_store import SharedMemoryFrameStore
from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent

logger = logging.getLogger(__name__)


class VideoRealtimeSubmissionError(RuntimeError):
    """An input could not be submitted to the active model request."""


async def warmup_video_realtime(
    client: Client,
    *,
    model_name: str,
    timeout_s: float = 180.0,
) -> None:
    """Exercise the dynamic vision path before accepting user traffic."""
    if timeout_s <= 0:
        raise ValueError("video realtime warmup timeout must be positive")

    request_id = f"video_warmup_req_{uuid.uuid4().hex}"
    session_id = f"video_warmup_sess_{uuid.uuid4().hex}"
    frame_store = SharedMemoryFrameStore()
    frame_ref: str | None = None
    processed = False

    request = GenerateRequest(
        model=model_name,
        prompt={
            "initial_prompt": "Warm up the realtime vision path.",
            "session_id": session_id,
        },
        sampling=SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=8,
        ),
        stream=True,
        max_tokens=8,
        output_modalities=["text"],
        extra_params={"realtime_warmup": True},
    )

    async def _run() -> None:
        nonlocal frame_ref, processed
        submitted = False
        async for chunk in client.generate(request, request_id=request_id):
            if chunk.control_event == "session.ready" and not submitted:
                output = BytesIO()
                Image.new("RGB", (640, 352), color=(127, 127, 127)).save(
                    output,
                    format="PNG",
                )
                frame_ref = frame_store.put(request_id, output.getvalue())
                event = FramePromptEvent(
                    request_id=request_id,
                    session_id=session_id,
                    seq_no=0,
                    timestamp=0.0,
                    frame_ref=frame_ref,
                    final=True,
                )
                await client.update_request(request_id, event.to_dict())
                submitted = True
                continue
            if chunk.control_event == "input.frame.processed":
                processed = True
                if frame_ref is not None:
                    frame_store.forget(request_id, frame_ref)

        if not submitted:
            raise RuntimeError("video realtime warmup never reached session.ready")
        if not processed:
            raise RuntimeError("video realtime warmup frame was not processed")

    try:
        await asyncio.wait_for(_run(), timeout=timeout_s)
    finally:
        try:
            await client.abort(request_id)
        except Exception:
            logger.debug(
                "Video realtime warmup request was already closed", exc_info=True
            )
        frame_store.cleanup(request_id)


class VideoSessionConfigure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["session.configure"]
    prompt: str = "Describe relevant changes in the video."
    system_prompt: str | None = None
    max_new_tokens: int = Field(default=4096, gt=0)
    max_tokens_per_turn: float = Field(
        default=86400.0,
        gt=0.0,
        allow_inf_nan=False,
        description="Maximum generation rate in tokens per second.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    input_queue_capacity: int = Field(default=4, ge=1, le=256)
    benchmark_ignore_eos: bool = False


class VideoFrameMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input.frame"]
    seq_no: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    prompt: str | None = None
    final: bool = False
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]

    @field_validator("prompt")
    @classmethod
    def normalize_optional_prompt(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            return None
        return value


class VideoPromptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["input.prompt"]
    seq_no: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    final: bool = False

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must contain non-whitespace text")
        return value


@dataclass
class _PendingFrame:
    metadata: VideoFrameMetadata


class VideoRealtimeSession:
    """Own one persistent generation request and its ordered binary frames."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        frame_store: SharedMemoryFrameStore,
        allow_benchmark_mode: bool = False,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.frame_store = frame_store
        self.allow_benchmark_mode = bool(allow_benchmark_mode)
        self.session_id = f"video_sess_{uuid.uuid4().hex}"
        self.request_id = f"video_req_{uuid.uuid4().hex}"
        self.pending_frame: _PendingFrame | None = None
        self.input_queue_capacity = 4
        self.input_capacity_changed = asyncio.Condition()
        self.outstanding_seq_nos: set[int] = set()
        self.accepted_by_seq: dict[int, asyncio.Event] = {}
        self.frame_refs_by_seq: dict[int, str] = {}
        self.last_timestamp = 0.0
        self.response_task: asyncio.Task[None] | None = None
        self.send_lock = asyncio.Lock()
        self.configured = False
        self.ready = False
        self.current_turn_id = 0
        self.final_received = False
        self.request_finished = False
        self.abort_sent = False
        self.closed = False

    async def run(self) -> None:
        await self.send(
            {
                "type": "session.created",
                "session_id": self.session_id,
                "request_id": self.request_id,
                "model": self.model_name,
                "turn_id": self.current_turn_id,
            }
        )
        try:
            while not self.closed:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] != "websocket.receive":
                    continue
                try:
                    if message.get("bytes") is not None:
                        await self.handle_frame_bytes(message["bytes"])
                        continue
                    raw = message.get("text")
                    if raw is None:
                        raise ValueError("empty WebSocket message")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise TypeError("top-level message must be a JSON object")
                    await self.handle_json(payload)
                except (
                    json.JSONDecodeError,
                    TypeError,
                    ValidationError,
                    ValueError,
                ) as exc:
                    await self.send_error_safely(
                        str(exc),
                        code="invalid_request",
                    )
                except VideoRealtimeSubmissionError as exc:
                    logger.exception(
                        "Realtime input submission failed for request %s",
                        self.request_id,
                    )
                    await self.send_error_safely(
                        str(exc),
                        code="input_submission_failed",
                    )
                    self.closed = True
        finally:
            await self.teardown()

    async def handle_json(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "session.configure":
            await self.configure(VideoSessionConfigure.model_validate(payload))
        elif event_type == "input.frame":
            await self.prepare_frame(VideoFrameMetadata.model_validate(payload))
        elif event_type == "input.prompt":
            await self.handle_prompt(VideoPromptInput.model_validate(payload))
        elif event_type == "session.abort":
            self.closed = True
            await self.abort_request()
        else:
            raise ValueError(f"unsupported event type: {event_type!r}")

    async def configure(self, config: VideoSessionConfigure) -> None:
        if self.configured:
            raise ValueError("session is already configured")
        if config.benchmark_ignore_eos and not self.allow_benchmark_mode:
            raise ValueError("benchmark_ignore_eos requires server benchmark mode")
        self.input_queue_capacity = config.input_queue_capacity
        self.configured = True
        await self.send(
            {
                "type": "session.configured",
                "session_id": self.session_id,
                "request_id": self.request_id,
                "input_queue_capacity": self.input_queue_capacity,
                "max_tokens_per_turn": config.max_tokens_per_turn,
            }
        )
        extra_params: dict[str, Any] = {
            "max_tokens_per_turn": config.max_tokens_per_turn,
        }
        if config.benchmark_ignore_eos:
            extra_params["benchmark_ignore_eos"] = True
        request = GenerateRequest(
            model=self.model_name,
            prompt={
                "initial_prompt": config.prompt,
                "system_prompt": config.system_prompt,
                "session_id": self.session_id,
            },
            sampling=SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_new_tokens=config.max_new_tokens,
            ),
            stream=True,
            max_tokens=config.max_new_tokens,
            output_modalities=["text"],
            extra_params=extra_params,
        )
        self.response_task = asyncio.create_task(self.stream_response(request))

    async def prepare_frame(self, metadata: VideoFrameMetadata) -> None:
        if not self.configured:
            raise ValueError("configure the session before sending frames")
        if not self.ready:
            raise ValueError("wait for session.ready before sending frames")
        if self.final_received:
            raise ValueError("session already received its final frame")
        if self.pending_frame is not None:
            raise ValueError("previous frame metadata is still awaiting binary data")
        await self._reserve_input(metadata.seq_no)
        self.pending_frame = _PendingFrame(metadata=metadata)
        await self.send(
            {
                "type": "input.frame.ready",
                "seq_no": metadata.seq_no,
            }
        )

    async def handle_prompt(self, metadata: VideoPromptInput) -> None:
        if not self.configured:
            raise ValueError("configure the session before sending prompts")
        if not self.ready:
            raise ValueError("wait for session.ready before sending prompts")
        if self.final_received:
            raise ValueError("session already received its final event")
        if self.pending_frame is not None:
            raise ValueError("previous frame metadata is still awaiting binary data")
        await self._reserve_input(metadata.seq_no)
        try:
            event = FramePromptEvent(
                request_id=self.request_id,
                session_id=self.session_id,
                seq_no=metadata.seq_no,
                timestamp=self.last_timestamp,
                frame_ref=None,
                prompt=metadata.prompt,
                final=metadata.final,
            )
        except Exception:
            await self._release_input(metadata.seq_no)
            raise
        try:
            await self.client.update_request(self.request_id, event.to_dict())
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            raise VideoRealtimeSubmissionError(
                f"failed to submit prompt event {metadata.seq_no}: {exc}"
            ) from exc
        self.final_received = metadata.final
        await self._send_accepted(
            metadata.seq_no,
            {
                "type": "input.prompt.accepted",
                "seq_no": metadata.seq_no,
                "final": metadata.final,
                "pending_events": len(self.outstanding_seq_nos),
                "interrupts_current_turn": True,
            },
        )

    async def handle_frame_bytes(self, payload: bytes) -> None:
        pending = self.pending_frame
        if pending is None:
            raise ValueError("binary frame must follow input.frame metadata")
        self.pending_frame = None
        metadata = pending.metadata
        frame_ref: str | None = None
        try:
            frame_ref = self.frame_store.put(self.request_id, payload)
        except ValueError:
            await self._release_input(metadata.seq_no)
            raise
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            raise VideoRealtimeSubmissionError(
                f"failed to store frame event {metadata.seq_no}: {exc}"
            ) from exc
        try:
            event = FramePromptEvent(
                request_id=self.request_id,
                session_id=self.session_id,
                seq_no=metadata.seq_no,
                timestamp=metadata.timestamp,
                frame_ref=frame_ref,
                prompt=metadata.prompt,
                final=metadata.final,
                fingerprint=hashlib.sha256(payload).hexdigest(),
            )
        except Exception:
            await self._release_input(metadata.seq_no)
            self.frame_store.discard(self.request_id, frame_ref)
            raise
        assert frame_ref is not None
        self.frame_refs_by_seq[metadata.seq_no] = frame_ref
        try:
            await self.client.update_request(self.request_id, event.to_dict())
        except Exception as exc:
            await self._release_input(metadata.seq_no)
            self.frame_refs_by_seq.pop(metadata.seq_no, None)
            self.frame_store.discard(self.request_id, frame_ref)
            raise VideoRealtimeSubmissionError(
                f"failed to submit frame event {metadata.seq_no}: {exc}"
            ) from exc
        self.final_received = metadata.final
        self.last_timestamp = float(metadata.timestamp)
        await self._send_accepted(
            metadata.seq_no,
            {
                "type": "input.frame.accepted",
                "seq_no": metadata.seq_no,
                "timestamp": metadata.timestamp,
                "final": metadata.final,
                "pending_events": len(self.outstanding_seq_nos),
                "interrupts_current_turn": metadata.prompt is not None,
            },
        )

    async def stream_response(self, request: GenerateRequest) -> None:
        streamed_text = ""
        try:
            async for chunk in self.client.generate(
                request, request_id=self.request_id
            ):
                if chunk.control_event == "session.ready":
                    self.ready = True
                    await self.send(
                        {
                            "type": "session.ready",
                            "session_id": self.session_id,
                            "request_id": self.request_id,
                            "turn_id": int(
                                (chunk.control_data or {}).get("turn_id", 0)
                            ),
                        }
                    )
                    continue
                if chunk.control_event == "response.turn.interrupted":
                    control_data = dict(chunk.control_data or {})
                    seq_no = int(control_data["seq_no"])
                    accepted = self.accepted_by_seq.get(seq_no)
                    if accepted is not None:
                        await accepted.wait()
                    await self.send(
                        {
                            "type": "response.turn.interrupted",
                            "turn_id": int(control_data["turn_id"]),
                            "next_turn_id": int(control_data["next_turn_id"]),
                            "seq_no": seq_no,
                        }
                    )
                    continue
                if chunk.control_event == "response.turn.silence":
                    control_data = dict(chunk.control_data or {})
                    await self.send(
                        {
                            "type": "response.turn.silence",
                            "turn_id": int(control_data["turn_id"]),
                            "seq_no": control_data.get("seq_no"),
                            "timestamp": control_data.get("timestamp"),
                            "silence_seq": int(control_data["silence_seq"]),
                        }
                    )
                    continue
                if chunk.control_event in (
                    "input.frame.processed",
                    "input.prompt.processed",
                ):
                    control_data = dict(chunk.control_data or {})
                    seq_no = int(control_data["seq_no"])
                    accepted = self.accepted_by_seq.get(seq_no)
                    if accepted is not None:
                        await accepted.wait()
                    frame_ref = self.frame_refs_by_seq.pop(seq_no, None)
                    if frame_ref is not None:
                        self.frame_store.forget(self.request_id, frame_ref)
                    await self._release_input(seq_no)
                    processed_payload = {
                        "type": chunk.control_event,
                        "seq_no": seq_no,
                        "timestamp": control_data["timestamp"],
                        "final": control_data["final"],
                        "pending_events": len(self.outstanding_seq_nos),
                    }
                    if control_data.get("turn_id") is not None:
                        self.current_turn_id = int(control_data["turn_id"])
                        processed_payload.update(
                            {
                                "interrupted_turn_id": int(
                                    control_data["interrupted_turn_id"]
                                ),
                                "turn_id": self.current_turn_id,
                            }
                        )
                    await self.send(processed_payload)
                    continue
                text = chunk.text
                if text and chunk.finish_reason is None:
                    streamed_text += text
                elif text and streamed_text and text.startswith(streamed_text):
                    text = text[len(streamed_text) :]
                if text:
                    await self.send(
                        {
                            "type": "response.text.delta",
                            "delta": text,
                            "turn_id": (
                                self.current_turn_id
                                if chunk.turn_id is None
                                else int(chunk.turn_id)
                            ),
                        }
                    )
                if chunk.finish_reason is not None:
                    await self.send(
                        {
                            "type": "response.done",
                            "finish_reason": chunk.finish_reason,
                            "turn_id": self.current_turn_id,
                        }
                    )
            self.request_finished = True
            self.frame_store.cleanup(self.request_id)
            await self.send({"type": "session.done"})
            self.closed = True
            await self._close_input_queue()
            await self.close_websocket()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Realtime response failed for request %s", self.request_id)
            await self.send_error_safely(
                str(exc),
                code="response_failed",
            )
            self.closed = True
            await self._close_input_queue()
            await self.close_websocket()

    async def teardown(self) -> None:
        if self.closed and self.request_finished:
            return
        self.closed = True
        if not self.request_finished and self.configured:
            try:
                await self.abort_request()
            except Exception:
                logger.exception(
                    "Failed to abort realtime request %s during teardown",
                    self.request_id,
                )
        task = self.response_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.frame_store.cleanup(self.request_id)
        await self._close_input_queue()
        await self.close_websocket()

    async def abort_request(self) -> None:
        if self.abort_sent:
            return
        self.abort_sent = True
        await self.client.abort(self.request_id)

    async def close_websocket(self) -> None:
        try:
            if (
                self.websocket.application_state is WebSocketState.CONNECTED
                and self.websocket.client_state is WebSocketState.CONNECTED
            ):
                await self.websocket.close()
        except (OSError, RuntimeError, WebSocketDisconnect):
            logger.debug(
                "Realtime WebSocket was already closed for request %s",
                self.request_id,
                exc_info=True,
            )

    async def send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send_json(payload)

    async def _reserve_input(self, seq_no: int) -> None:
        async with self.input_capacity_changed:
            await self.input_capacity_changed.wait_for(
                lambda: self.closed
                or len(self.outstanding_seq_nos) < self.input_queue_capacity
            )
            if self.closed:
                raise RuntimeError("realtime session is closed")
            if seq_no in self.outstanding_seq_nos:
                raise ValueError(f"event {seq_no} is already pending")
            self.outstanding_seq_nos.add(seq_no)
            self.accepted_by_seq[seq_no] = asyncio.Event()

    async def _release_input(self, seq_no: int) -> None:
        async with self.input_capacity_changed:
            self.outstanding_seq_nos.discard(seq_no)
            accepted = self.accepted_by_seq.pop(seq_no, None)
            if accepted is not None:
                accepted.set()
            self.input_capacity_changed.notify_all()

    async def _close_input_queue(self) -> None:
        async with self.input_capacity_changed:
            for accepted in self.accepted_by_seq.values():
                accepted.set()
            self.accepted_by_seq.clear()
            self.outstanding_seq_nos.clear()
            self.input_capacity_changed.notify_all()

    async def _send_accepted(self, seq_no: int, payload: dict[str, Any]) -> None:
        await self.send(payload)
        accepted = self.accepted_by_seq.get(seq_no)
        if accepted is not None:
            accepted.set()

    async def send_error(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
    ) -> None:
        payload: dict[str, Any] = {"type": "error", "message": message}
        if code is not None:
            payload["code"] = code
        if retryable:
            payload["retryable"] = True
        await self.send(payload)

    async def send_error_safely(
        self,
        message: str,
        *,
        code: str,
        retryable: bool = False,
    ) -> bool:
        """Best-effort error delivery that never replaces the original failure."""
        try:
            await self.send_error(
                message,
                code=code,
                retryable=retryable,
            )
        except (OSError, RuntimeError, WebSocketDisconnect):
            logger.debug(
                "Could not deliver realtime error %s for request %s",
                code,
                self.request_id,
                exc_info=True,
            )
            return False
        return True


class VideoRealtimeSessionManager:
    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        allow_benchmark_mode: bool = False,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.allow_benchmark_mode = bool(allow_benchmark_mode)
        self.frame_store = SharedMemoryFrameStore()
        self.sessions: dict[str, VideoRealtimeSession] = {}

    def open(self, websocket: WebSocket) -> VideoRealtimeSession:
        if self.sessions:
            raise RuntimeError("video realtime service already has an active session")
        session = VideoRealtimeSession(
            websocket,
            client=self.client,
            model_name=self.model_name,
            frame_store=self.frame_store,
            allow_benchmark_mode=self.allow_benchmark_mode,
        )
        self.sessions[session.session_id] = session
        return session

    async def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is not None:
            await session.teardown()


def register_video_realtime(
    app: FastAPI,
    *,
    allow_benchmark_mode: bool = False,
) -> None:
    manager = VideoRealtimeSessionManager(
        client=app.state.client,
        model_name=app.state.model_name,
        allow_benchmark_mode=allow_benchmark_mode,
    )
    app.state.video_realtime_manager = manager

    @app.websocket("/v1/video/realtime")
    async def video_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            session = manager.open(websocket)
        except RuntimeError as exc:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "session_capacity_exceeded",
                    "message": str(exc),
                }
            )
            await websocket.close(code=1013)
            return
        try:
            await session.run()
        finally:
            await manager.close(session.session_id)
