"""Caller-compatible ``/v1/realtime`` WebSocket protocol for VL analysis.

One WebSocket connection per analysis round, as specified by the calling
side:

    connect → start(json) → ready → (frame json + binary JPEG)*n
    ← frame_ack (per frame) ← output (incremental text, <|im_end|> ends the
    round) → stop → close

This bridges onto the shared realtime engine (Client +
SharedMemoryFrameStore + FramePromptEvent) — the same path as
``/v1/video/realtime``. Accepted-but-ignored caller fields:
``frame_queue_size`` (queueing is engine-owned) and ``max_tokens_per_second``
(documented as an advisory pacing value). ``do_sample=false`` maps to greedy
decoding (temperature 0) regardless of the temperature field.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel, Field, field_validator
from starlette.websockets import WebSocketDisconnect

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.models.moss_vl_realtime.frame_store import SharedMemoryFrameStore
from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent

logger = logging.getLogger(__name__)

BUSY_MESSAGE = "realtime session is already active"
END_MARKER = "<|im_end|>"
MAX_FRAME_BYTES = 10 * 1024 * 1024


class VisionRealtimeStart(BaseModel):
    type: str = "start"
    prompt: str = Field(min_length=1)
    frame_queue_size: int = 32
    max_new_tokens: int = Field(default=512, ge=1)
    max_tokens_per_second: int = 160
    do_sample: bool = False
    temperature: float = 0.2
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 1.05

    @field_validator("prompt")
    @classmethod
    def _strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value


class VisionRealtimeFrame(BaseModel):
    type: str = "frame"
    timestamp: float = Field(ge=0.0)


class VisionRealtimeSession:
    """One analysis round: start → frames → output → stop."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        frame_store: SharedMemoryFrameStore,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.frame_store = frame_store
        self.session_id = f"vision_sess_{uuid.uuid4().hex}"
        self.request_id = f"vision_req_{uuid.uuid4().hex}"
        self.events: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self.gen_task: asyncio.Task[None] | None = None
        self.gen_request: GenerateRequest | None = None
        self.abort_sent = False
        self.next_seq = 0
        self.outstanding_refs: list[str] = []
        self.pending_frames: list[tuple[float, bytes]] = []
        self.flush_task: asyncio.Task[None] | None = None
        self.batch_flushed = False
        self.frame_gap_s = 0.5

    # --- plumbing ---------------------------------------------------------

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))

    async def _abort(self) -> None:
        if self.abort_sent:
            return
        self.abort_sent = True
        if self.gen_task is not None:
            self.gen_task.cancel()
        try:
            await self.client.abort(self.request_id)
        except Exception:
            logger.debug("vision realtime abort already closed", exc_info=True)

    def _build_request(self, start: VisionRealtimeStart) -> GenerateRequest:
        # do_sample=false means greedy decoding; the temperature field is then
        # ignored by the caller's own contract.
        temperature = 0.0 if not start.do_sample else start.temperature
        return GenerateRequest(
            model=self.model_name,
            prompt={
                "initial_prompt": start.prompt,
                "session_id": self.session_id,
            },
            sampling=SamplingParams(
                temperature=temperature,
                top_k=start.top_k,
                top_p=start.top_p,
                repetition_penalty=start.repetition_penalty,
                max_new_tokens=start.max_new_tokens,
            ),
            stream=True,
            max_tokens=start.max_new_tokens,
            output_modalities=["text"],
        )

    # --- engine stream pump -----------------------------------------------

    async def _pump_generation(self) -> None:
        streamed = ""
        try:
            async for chunk in self.client.generate(
                self.gen_request, request_id=self.request_id
            ):
                event = chunk.control_event
                if event == "session.ready":
                    await self.events.put(("ready", None))
                elif event == "input.frame.processed":
                    if self.outstanding_refs:
                        ref = self.outstanding_refs.pop(0)
                        try:
                            self.frame_store.forget(self.request_id, ref)
                        except Exception:
                            logger.debug("frame forget failed", exc_info=True)
                elif event == "response.done":
                    await self.events.put(("round_done", None))
                elif event == "response.turn.silence":
                    await self.events.put(("silence", None))
                elif event == "session.done":
                    await self.events.put(("engine_end", None))
                text = chunk.text
                if not text:
                    continue
                if chunk.finish_reason is None:
                    streamed += text
                    await self.events.put(("delta", text))
                elif streamed and text.startswith(streamed):
                    tail = text[len(streamed):]
                    if tail:
                        await self.events.put(("delta", tail))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.events.put(("engine_error", str(exc)))

    # --- frames ------------------------------------------------------------

    async def _submit_frame(
        self, timestamp: float, data: bytes, *, is_final: bool
    ) -> None:
        frame_ref = self.frame_store.put(self.request_id, data)
        self.outstanding_refs.append(frame_ref)
        event = FramePromptEvent(
            request_id=self.request_id,
            session_id=self.session_id,
            seq_no=self.next_seq,
            timestamp=timestamp,
            frame_ref=frame_ref,
            final=is_final,
        )
        self.next_seq += 1
        await self.client.update_request(self.request_id, event.to_dict())

    async def _flush_pending_batch(self) -> None:
        """Submit the buffered frame batch; the last frame closes the turn.

        The caller protocol carries no explicit end-of-batch marker, so a
        short inter-frame gap (default 500ms) marks the boundary. A batch
        may only be flushed once per connection: later frames after the
        final event are rejected by the session contract.
        """
        frames, self.pending_frames = self.pending_frames, []
        self.batch_flushed = True
        for index, (timestamp, data) in enumerate(frames):
            await self._submit_frame(
                timestamp, data, is_final=index == len(frames) - 1
            )

    def _schedule_batch_flush(self) -> None:
        if self.flush_task is not None and not self.flush_task.done():
            self.flush_task.cancel()
        self.flush_task = asyncio.create_task(self._flush_after_gap())

    async def _flush_after_gap(self) -> None:
        await asyncio.sleep(self.frame_gap_s)
        if self.pending_frames and not self.batch_flushed:
            await self._flush_pending_batch()

    # --- main loop ----------------------------------------------------------

    async def run(self) -> None:
        try:
            raw = await asyncio.wait_for(self.websocket.receive_text(), timeout=10.0)
        except asyncio.TimeoutError:
            await self.send_json({"type": "error", "message": "start not received in time"})
            return
        try:
            start = VisionRealtimeStart(**json.loads(raw))
        except Exception as exc:
            await self.send_json({"type": "error", "message": f"invalid start: {exc}"})
            return

        self.gen_request = self._build_request(start)
        self.gen_task = asyncio.create_task(self._pump_generation())

        try:
            kind, payload = await asyncio.wait_for(self.events.get(), timeout=10.0)
        except asyncio.TimeoutError:
            await self.send_json(
                {"type": "error", "message": "engine did not become ready in time"}
            )
            return
        if kind != "ready":
            message = BUSY_MESSAGE if kind == "engine_error" else f"unexpected engine event {kind}"
            await self.send_json({"type": "error", "message": message or BUSY_MESSAGE})
            return
        await self.send_json({"type": "ready"})

        pending_timestamp: float | None = None
        round_ended = False
        round_text = ""
        try:
            while True:
                get_task = asyncio.create_task(self.events.get())
                recv_task = asyncio.create_task(self.websocket.receive())
                done, pending = await asyncio.wait(
                    {get_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                if get_task in done:
                    kind, payload = get_task.result()
                    if round_ended and kind not in ("engine_error",):
                        # Round already closed (answered or silent); the model
                        # may keep decoding until its allowance — swallow it.
                        continue
                    if kind == "delta":
                        round_text += str(payload)
                        await self.send_json({"type": "output", "text": str(payload)})
                    elif kind == "silence":
                        if self.batch_flushed:
                            # The model decided the (flushed) frames carry
                            # nothing to report; the caller recognizes
                            # <|silence|> as a terminal marker.
                            await self.send_json(
                                {"type": "output", "text": "<|silence|>"}
                            )
                            round_ended = True
                    elif kind in ("round_done", "engine_end"):
                        # Round completed with visible text: deterministic
                        # terminal token for the caller's collector.
                        await self.send_json({"type": "output", "text": END_MARKER})
                        round_ended = True
                        round_text = ""
                    elif kind == "engine_error":
                        await self.send_json(
                            {"type": "error", "message": str(payload or "engine error")}
                        )
                        return

                if recv_task in done:
                    message = recv_task.result()
                    if isinstance(message, dict) and message.get(
                        "type"
                    ) == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        text = (message.get("text") or "").strip()
                        if text == "stop":
                            return
                        try:
                            frame = VisionRealtimeFrame(**json.loads(text))
                        except Exception as exc:
                            await self.send_json(
                                {"type": "error", "message": f"invalid frame meta: {exc}"}
                            )
                            return
                        pending_timestamp = frame.timestamp
                    elif message.get("bytes") is not None:
                        data = message["bytes"]
                        if len(data) > MAX_FRAME_BYTES:
                            await self.send_json(
                                {
                                    "type": "error",
                                    "message": f"frame exceeds {MAX_FRAME_BYTES} bytes",
                                }
                            )
                            return
                        if self.batch_flushed:
                            await self.send_json(
                                {
                                    "type": "error",
                                    "message": "frame received after batch flush",
                                }
                            )
                            return
                        timestamp = (
                            pending_timestamp if pending_timestamp is not None else 0.0
                        )
                        pending_timestamp = None
                        self.pending_frames.append((timestamp, data))
                        await self.send_json({"type": "frame_ack"})
                        self._schedule_batch_flush()
                    else:
                        return
        except WebSocketDisconnect:
            return
        finally:
            if self.flush_task is not None:
                self.flush_task.cancel()
            await self._abort()


class VisionRealtimeManager:
    """Track active compat sessions to expose the caller's busy semantics."""

    def __init__(
        self,
        *,
        client: Client,
        model_name: str,
        frame_store: SharedMemoryFrameStore,
        max_sessions: int,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.frame_store = frame_store
        self.max_sessions = int(max_sessions)
        self.active = 0

    def try_acquire(self) -> bool:
        if self.active >= self.max_sessions:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)


def register_vision_realtime(
    app: FastAPI,
    *,
    max_sessions: int | None = None,
) -> None:
    """Mount the caller-compatible ``/v1/realtime`` VL analysis endpoint."""

    manager = VisionRealtimeManager(
        client=app.state.client,
        model_name=app.state.model_name,
        frame_store=SharedMemoryFrameStore(),
        max_sessions=max_sessions or 1,
    )
    app.state.vision_realtime_manager = manager

    @app.websocket("/v1/realtime")
    async def vision_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        if not manager.try_acquire():
            await websocket.send_text(
                json.dumps({"type": "error", "message": BUSY_MESSAGE})
            )
            await websocket.close(code=1013)
            return
        session = VisionRealtimeSession(
            websocket,
            client=manager.client,
            model_name=manager.model_name,
            frame_store=manager.frame_store,
        )
        try:
            await session.run()
        finally:
            manager.release()
