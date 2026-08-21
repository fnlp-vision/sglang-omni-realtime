from __future__ import annotations

import asyncio
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.moss_vl_realtime.frame_store import SharedMemoryFrameStore
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.video_realtime import VideoRealtimeSession


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), color=(40, 50, 60)).save(output, format="PNG")
    return output.getvalue()


class _Client:
    def __init__(self) -> None:
        self.update_event = asyncio.Event()
        self.updates: list[tuple[str, dict]] = []
        self.aborted: list[str] = []

    async def generate(self, request, request_id=None):
        del request
        await self.update_event.wait()
        update = self.updates[0][1]
        control_event = (
            "input.frame.processed"
            if update.get("frame_ref") is not None
            else "input.prompt.processed"
        )
        yield GenerateChunk(
            request_id=request_id,
            modality="control",
            control_event=control_event,
            control_data={
                "seq_no": update["seq_no"],
                "timestamp": update["timestamp"],
                "final": update["final"],
            },
        )
        yield GenerateChunk(request_id=request_id, text="car")
        yield GenerateChunk(request_id=request_id, text="car", finish_reason="stop")

    async def update_request(self, request_id, data, *, stage_name=None):
        del stage_name
        self.updates.append((request_id, data))
        self.update_event.set()

    async def abort(self, request_id):
        self.aborted.append(request_id)
        return True


class _HoldingClient(_Client):
    async def generate(self, request, request_id=None):
        del request
        await asyncio.Event().wait()
        yield GenerateChunk(request_id=request_id)

    async def update_request(self, request_id, data, *, stage_name=None):
        del stage_name
        self.updates.append((request_id, data))


def test_video_realtime_binary_frame_reaches_request_update() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Track the car.",
                "max_new_tokens": 8,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 1.5,
                "final": True,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json() == {"type": "input.frame.ready", "seq_no": 0}
        websocket.send_bytes(_png_bytes())

        events = []
        event_types = set()
        while "response.done" not in event_types:
            event = websocket.receive_json()
            events.append(event)
            event_types.add(event["type"])

    assert {
        "input.frame.accepted",
        "input.frame.processed",
        "response.text.delta",
        "response.done",
    } <= event_types
    assert [
        event["delta"] for event in events if event["type"] == "response.text.delta"
    ] == ["car"]
    assert len(client.updates) == 1
    request_id, update = client.updates[0]
    assert request_id == created["request_id"]
    assert update["session_id"] == created["session_id"]
    assert update["timestamp"] == 1.5
    assert update["frame_ref"].startswith("shm://")


def test_video_realtime_allows_only_one_websocket() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
    )
    with (
        TestClient(app) as test_client,
        test_client.websocket_connect("/v1/video/realtime") as first,
    ):
        assert first.receive_json()["type"] == "session.created"
        with test_client.websocket_connect("/v1/video/realtime") as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "session_capacity_exceeded"


def test_video_realtime_prompt_only_event_reaches_request_update() -> None:
    client = _Client()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        created = websocket.receive_json()
        websocket.send_json({"type": "session.configure", "prompt": "Watch."})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json(
            {
                "type": "input.prompt",
                "seq_no": 0,
                "prompt": "How many?",
                "final": True,
            }
        )
        event_types = set()
        while "response.done" not in event_types:
            event_types.add(websocket.receive_json()["type"])

    assert {"input.prompt.accepted", "input.prompt.processed"} <= event_types
    request_id, update = client.updates[0]
    assert request_id == created["request_id"]
    assert update["session_id"] == created["session_id"]
    assert update["prompt"] == "How many?"
    assert update["final"] is True
    assert "frame_ref" not in update


def test_video_realtime_accepts_multiple_inflight_frames() -> None:
    client = _HoldingClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        assert websocket.receive_json()["type"] == "session.created"
        websocket.send_json({"type": "session.configure"})
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 0,
                "timestamp": 0.0,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        assert websocket.receive_json()["type"] == "input.frame.accepted"
        websocket.send_json(
            {
                "type": "input.frame",
                "seq_no": 1,
                "timestamp": 1.0,
                "mime_type": "image/png",
            }
        )
        assert websocket.receive_json()["type"] == "input.frame.ready"
        websocket.send_bytes(_png_bytes())
        accepted = websocket.receive_json()
        assert accepted["type"] == "input.frame.accepted"
        assert accepted["pending_events"] == 2

    assert [update[1]["seq_no"] for update in client.updates] == [0, 1]


def test_video_realtime_applies_bounded_input_backpressure() -> None:
    async def _run() -> None:
        session = VideoRealtimeSession(
            None,  # type: ignore[arg-type]
            client=None,  # type: ignore[arg-type]
            model_name="moss-vl-realtime",
            frame_store=SharedMemoryFrameStore(),
        )
        session.input_queue_capacity = 1
        await session._reserve_input(0)

        waiting = asyncio.create_task(session._reserve_input(1))
        await asyncio.sleep(0)
        assert waiting.done() is False

        await session._release_input(0)
        await asyncio.wait_for(waiting, timeout=1.0)
        assert session.outstanding_seq_nos == {1}
        await session._close_input_queue()

    asyncio.run(_run())


class _RequestCaptureClient(_Client):
    captured_request = None

    async def generate(self, request, request_id=None):
        self.captured_request = request
        async for chunk in super().generate(request, request_id=request_id):
            yield chunk


def test_video_realtime_benchmark_ignore_eos_flows_to_extra_params() -> None:
    client = _RequestCaptureClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
        video_realtime_benchmark_mode=True,
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Watch.",
                "benchmark_ignore_eos": True,
            }
        )
        assert websocket.receive_json()["type"] == "session.configured"
        websocket.send_json(
            {
                "type": "input.prompt",
                "seq_no": 0,
                "prompt": "How many?",
                "final": True,
            }
        )
        event_types = set()
        while "response.done" not in event_types:
            event_types.add(websocket.receive_json()["type"])

    assert client.captured_request is not None
    assert client.captured_request.extra_params == {"benchmark_ignore_eos": True}


def test_video_realtime_rejects_benchmark_option_in_production_mode() -> None:
    client = _HoldingClient()
    app = create_app(
        client,  # type: ignore[arg-type]
        model_name="moss-vl-realtime",
        architectures=["MossVLRealtimeForConditionalGeneration"],
    )
    with TestClient(app).websocket_connect("/v1/video/realtime") as websocket:
        websocket.receive_json()
        websocket.send_json(
            {
                "type": "session.configure",
                "prompt": "Watch.",
                "benchmark_ignore_eos": True,
            }
        )
        error = websocket.receive_json()

    assert error["type"] == "error"
    assert "requires server benchmark mode" in error["message"]
    assert client.updates == []
