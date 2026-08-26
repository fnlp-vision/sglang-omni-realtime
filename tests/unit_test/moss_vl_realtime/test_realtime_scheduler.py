from __future__ import annotations

from array import array
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    FramePromptEvent,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.scheduler import (
    MossVLRealtimeScheduler,
    _append_segment_to_request,
    _undo_appended_segment,
    bind_realtime_page_row,
)
from sglang_omni.models.moss_vl_realtime.session_state import (
    MossVLRealtimeSessionController,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler

Range = namedtuple("Range", ["start", "end"])
Range.length = property(lambda self: self.end - self.start)


class _Req:
    def __init__(self) -> None:
        self.req_pool_idx = 0
        self.origin_input_ids = array("q", [101, 102])
        self.output_ids = array("q", [201])
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids
        self.sampling_params = SimpleNamespace(max_new_tokens=10)
        self.multimodal_inputs = SimpleNamespace(name="old-mm")
        self.extend_range = Range(0, 2)
        self.prefix_indices = torch.tensor([11, 12])
        self.skip_radix_cache_insert = False

    def _refresh_fill_ids(self) -> None:
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def set_extend_range(self, start: int, end: int) -> None:
        self.extend_range = Range(start, end)


def _segment():
    event = FramePromptEvent(
        request_id="req-1",
        session_id="session-1",
        seq_no=0,
        timestamp=0.0,
        frame_ref="relay://frame-0",
    )
    return SimpleNamespace(
        events=(event,),
        raw_append_ids=(-101, -101, 301, 302),
        full_grid_thw=torch.tensor([[1, 2, 2]]),
        multimodal_inputs=SimpleNamespace(
            name="new-mm",
            mrope_positions=torch.tensor([[1, 2], [1, 2], [1, 2]]),
            visible_frame_counts=torch.tensor([0, 1]),
        ),
    )


def test_append_segment_keeps_pending_token_before_frame_event() -> None:
    req = _Req()
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        req_pool_index=0,
        decoder_length=2,
        next_mrope_position=2,
    )
    req._moss_vl_realtime_state = state
    page_table = torch.zeros((1, 12), dtype=torch.int32)
    page_table[0, :2] = torch.tensor([11, 12])
    bind_realtime_page_row(req, page_table)

    segment = _segment()
    _append_segment_to_request(req, state, segment)

    assert req.output_ids.tolist() == [201, -101, -101, 301, 302]
    assert state.pending_token_id == 201
    assert req.full_untruncated_fill_ids[-5:].tolist() == [
        201,
        -101,
        -101,
        301,
        302,
    ]
    assert req.prefix_indices.tolist() == [11, 12]
    assert req.prefix_indices.dtype is torch.int64
    assert req.extend_range == Range(2, 7)
    assert req.sampling_params.max_new_tokens == 14
    assert req.multimodal_inputs.name == "new-mm"
    assert req.skip_radix_cache_insert is True
    assert req._moss_vl_realtime_staged_visible_frame_counts.tolist() == [0, 1]

    _undo_appended_segment(req, segment)
    assert req.output_ids.tolist() == [201]
    assert req.sampling_params.max_new_tokens == 10
    assert req.multimodal_inputs.name == "old-mm"
    assert req.extend_range == Range(0, 2)
    assert req.prefix_indices.tolist() == [11, 12]
    assert req.skip_radix_cache_insert is False
    assert state.pending_token_id is None
    assert not hasattr(req, "_moss_vl_realtime_staged_visible_frame_counts")


def test_append_segment_stages_multi_event_turn_transitions() -> None:
    req = _Req()
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        req_pool_index=0,
        decoder_length=2,
        next_mrope_position=2,
        turn_id=3,
    )
    req._moss_vl_realtime_state = state
    page_table = torch.zeros((1, 12), dtype=torch.int32)
    page_table[0, :2] = torch.tensor([11, 12])
    bind_realtime_page_row(req, page_table)

    events = (
        FramePromptEvent(
            request_id="req-1",
            session_id="session-1",
            seq_no=0,
            timestamp=0.0,
            frame_ref=None,
            prompt="one",
        ),
        FramePromptEvent(
            request_id="req-1",
            session_id="session-1",
            seq_no=1,
            timestamp=1.0,
            frame_ref="relay://frame-1",
        ),
        FramePromptEvent(
            request_id="req-1",
            session_id="session-1",
            seq_no=2,
            timestamp=2.0,
            frame_ref=None,
            prompt="two",
        ),
    )
    segment = SimpleNamespace(
        events=events,
        raw_append_ids=(301, 302),
        full_grid_thw=torch.tensor([[1, 2, 2]]),
        multimodal_inputs=SimpleNamespace(
            name="new-mm",
            mrope_positions=torch.tensor([[1, 2], [1, 2], [1, 2]]),
            visible_frame_counts=torch.tensor([0, 1]),
        ),
    )

    _append_segment_to_request(req, state, segment)

    assert req._moss_vl_realtime_staged_turn_transition == {
        "interrupted_turn_id": 3,
        "turn_id": 5,
        "prompt_seq_nos": [0, 2],
    }
    assert [event["seq_no"] for event in req._moss_vl_realtime_staged_events] == [
        0,
        1,
        2,
    ]

    _undo_appended_segment(req, segment)
    assert not hasattr(req, "_moss_vl_realtime_staged_events")
    assert not hasattr(req, "_moss_vl_realtime_staged_turn_transition")


def test_silence_parks_request_without_releasing_kv() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.silence_token_ids = (70, 77)
    scheduler.parked_reqs = {}
    scheduler.parked_since = {}
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    scheduler.realtime_sessions.open("req-park", "session-park")
    state = MossVLRealtimeRuntimeState(
        request_id="req-park",
        session_id="session-park",
        phase=MossVLRealtimePhase.DECODING,
    )
    req = SimpleNamespace(
        rid="req-park",
        output_ids=array("q", [5, 70, 77]),
        _omni_data=SimpleNamespace(runtime_state=state),
        finished=lambda: False,
    )
    req._moss_vl_realtime_state = state
    kept: list[list[int]] = []
    batch = SimpleNamespace(
        reqs=[req],
        batch_is_full=True,
        filter_batch=lambda *, keep_indices: (
            kept.append(keep_indices),
            setattr(batch, "reqs", [batch.reqs[i] for i in keep_indices]),
        ),
    )

    scheduler._park_silent_requests(batch)

    assert kept == [[]]
    assert batch.reqs == []
    assert scheduler.parked_reqs == {"req-park": req}
    assert state.phase is MossVLRealtimePhase.WAITING_FOR_EVENT


def test_pending_event_prevents_silence_park() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.silence_token_ids = (77,)
    scheduler.parked_reqs = {}
    scheduler.parked_since = {}
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    session = scheduler.realtime_sessions.open("req-live", "session-live")
    session.accept(
        FramePromptEvent(
            request_id="req-live",
            session_id="session-live",
            seq_no=0,
            timestamp=0.0,
            frame_ref="relay://frame-live",
        )
    )
    state = MossVLRealtimeRuntimeState(
        request_id="req-live",
        session_id="session-live",
        phase=MossVLRealtimePhase.DECODING,
    )
    req = SimpleNamespace(
        rid="req-live",
        output_ids=array("q", [77]),
        finished=lambda: False,
    )
    req._moss_vl_realtime_state = state
    batch = SimpleNamespace(
        reqs=[req],
        batch_is_full=False,
        filter_batch=lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    scheduler._park_silent_requests(batch)

    assert batch.reqs == [req]
    assert scheduler.parked_reqs == {}
    assert state.phase is MossVLRealtimePhase.DECODING


def test_result_processing_delegates_before_park(monkeypatch) -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    calls = []
    monkeypatch.setattr(
        OmniScheduler,
        "process_batch_result",
        lambda self, batch, result: calls.append(("upstream", batch, result)),
    )
    scheduler._park_silent_requests = lambda batch: calls.append(("park", batch))
    batch = SimpleNamespace(reqs=[])
    result = object()

    scheduler.process_batch_result(batch, result)

    assert calls == [("upstream", batch, result), ("park", batch)]


def test_decode_rate_gate_uses_monotonic_deadline(monkeypatch) -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    state = MossVLRealtimeRuntimeState(
        request_id="req-rate",
        session_id="session-rate",
        max_tokens_per_turn=4,
        next_decode_not_before=10.25,
    )
    req = SimpleNamespace(_moss_vl_realtime_state=state)
    scheduler.running_batch = SimpleNamespace(reqs=[req])

    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    assert scheduler._decode_rate_limited() is True

    monkeypatch.setattr("time.monotonic", lambda: 10.25)
    assert scheduler._decode_rate_limited() is False


def test_batch_launch_sets_next_decode_deadline(monkeypatch) -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    state = MossVLRealtimeRuntimeState(
        request_id="req-rate",
        session_id="session-rate",
        max_tokens_per_turn=4,
    )
    req = SimpleNamespace(_moss_vl_realtime_state=state)
    batch = SimpleNamespace(reqs=[req])
    monkeypatch.setattr(
        OmniScheduler,
        "_stamp_batch_launch",
        lambda self, value: setattr(value, "launch_ts", 12.0),
    )

    scheduler._stamp_batch_launch(batch)

    assert state.next_decode_not_before == pytest.approx(12.25)


def test_realtime_extend_bypasses_decode_rate_gate(monkeypatch) -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler._expire_parked_requests = lambda: None
    scheduler._async_pending = None
    scheduler._realtime_extend_batch = object()
    scheduler._decode_rate_limited = lambda: (_ for _ in ()).throw(
        AssertionError("rate gate must not inspect a realtime extend")
    )
    expected = object()
    monkeypatch.setattr(
        OmniScheduler,
        "get_next_batch_to_run",
        lambda self: expected,
    )

    assert scheduler.get_next_batch_to_run() is expected


def _ingest_scheduler() -> tuple[MossVLRealtimeScheduler, list, list]:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    scheduler.realtime_sessions.open("req-ingest", "session-ingest")
    errors: list[tuple[str, Exception]] = []
    aborted: list[str] = []
    scheduler._emit_request_error = lambda rid, exc: errors.append((rid, exc))
    scheduler.abort = lambda rid, *, defer_running_cleanup=True: aborted.append(rid)
    return scheduler, errors, aborted


def _ingest_req_data() -> SimpleNamespace:
    return SimpleNamespace(
        runtime_state=MossVLRealtimeRuntimeState(
            request_id="req-ingest",
            session_id="session-ingest",
            phase=MossVLRealtimePhase.DECODING,
        )
    )


def test_ingest_request_update_accepts_in_order_event() -> None:
    scheduler, errors, aborted = _ingest_scheduler()
    event = FramePromptEvent(
        request_id="req-ingest",
        session_id="session-ingest",
        seq_no=0,
        timestamp=0.0,
        frame_ref="relay://frame-0",
    )

    scheduler._ingest_request_update(_ingest_req_data(), event)

    session = scheduler.realtime_sessions.get("req-ingest")
    assert errors == []
    assert aborted == []
    assert list(session.pending_events) == [event]


def test_out_of_order_update_aborts_request_without_raising() -> None:
    scheduler, errors, aborted = _ingest_scheduler()
    event = FramePromptEvent(
        request_id="req-ingest",
        session_id="session-ingest",
        seq_no=5,
        timestamp=1.0,
        frame_ref="relay://frame-bad",
    )

    scheduler._ingest_request_update(_ingest_req_data(), event)

    assert [rid for rid, _ in errors] == ["req-ingest"]
    assert isinstance(errors[0][1], ValueError)
    assert aborted == ["req-ingest"]
    session = scheduler.realtime_sessions.get("req-ingest")
    assert not session.pending_events


def test_event_after_final_aborts_request_without_raising() -> None:
    scheduler, errors, aborted = _ingest_scheduler()
    final_event = FramePromptEvent(
        request_id="req-ingest",
        session_id="session-ingest",
        seq_no=0,
        timestamp=0.0,
        frame_ref="relay://frame-0",
        final=True,
    )
    scheduler._ingest_request_update(_ingest_req_data(), final_event)
    assert errors == []

    late_event = FramePromptEvent(
        request_id="req-ingest",
        session_id="session-ingest",
        seq_no=1,
        timestamp=1.0,
        frame_ref="relay://frame-late",
    )
    scheduler._ingest_request_update(_ingest_req_data(), late_event)

    assert [rid for rid, _ in errors] == ["req-ingest"]
    assert isinstance(errors[0][1], RuntimeError)
    assert aborted == ["req-ingest"]


def test_parked_overrun_free_reconciles_request_accounting() -> None:
    """Freeing a parked row's overrun slot must also repair its bookkeeping,
    or the later abort path would free the same slot a second time."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.page_size = 1
    scheduler.server_args = SimpleNamespace(disable_radix_cache=False)
    freed: list[list[int]] = []
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        free=lambda slots: freed.append([int(v) for v in slots])
    )
    req_to_token = torch.zeros((1, 16), dtype=torch.int64)
    req_to_token[0, 8] = 77  # overrun slot at the committed offset
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    state = MossVLRealtimeRuntimeState(
        request_id="req-park",
        session_id="session-park",
        req_pool_index=0,
        encoder_length=3,
        decoder_length=5,
        phase=MossVLRealtimePhase.WAITING_FOR_EVENT,
    )
    req = SimpleNamespace(
        rid="req-park",
        kv_committed_len=8,
        kv=SimpleNamespace(kv_allocated_len=9),
    )
    req._moss_vl_realtime_state = state
    batch = SimpleNamespace(reqs=[req], out_cache_loc=torch.tensor([77]))

    scheduler._free_parked_overrun_step_slots(batch, [0])

    assert freed == [[77]]
    assert req.kv.kv_allocated_len == 8
    assert req.kv_committed_len == 8
    assert req_to_token[0, 8].item() == 0


def test_park_also_filters_live_running_batch() -> None:
    """Async resolve parks via a snapshot; the live batch must drop the row
    too, otherwise the parked request ghost-decodes every step."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.silence_token_ids = (77,)
    scheduler.parked_reqs = {}
    scheduler.parked_since = {}
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    scheduler.realtime_sessions.open("req-park", "session-park")
    state = MossVLRealtimeRuntimeState(
        request_id="req-park",
        session_id="session-park",
        phase=MossVLRealtimePhase.DECODING,
    )
    req = SimpleNamespace(
        rid="req-park",
        output_ids=array("q", [77]),
        finished=lambda: False,
    )
    req._moss_vl_realtime_state = state

    def _batch():
        batch = SimpleNamespace(reqs=[req], batch_is_full=True)
        def _filter(*, keep_indices, batch=batch):
            batch.reqs = [batch.reqs[i] for i in keep_indices]
        batch.filter_batch = _filter
        return batch

    snapshot = _batch()
    live = _batch()
    scheduler.running_batch = live

    scheduler._park_silent_requests(snapshot)

    assert snapshot.reqs == []
    assert live.reqs == []
    assert scheduler.parked_reqs == {"req-park": req}
    assert state.phase is MossVLRealtimePhase.WAITING_FOR_EVENT


def test_context_capacity_guard_rejects_overlength_extend() -> None:
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        _guard_realtime_context_capacity,
    )

    state = MossVLRealtimeRuntimeState(
        request_id="req-full",
        session_id="session-full",
        req_pool_index=0,
        encoder_length=10,
        decoder_length=20,
    )
    segment = SimpleNamespace(raw_append_ids=tuple(range(50)))
    pool = SimpleNamespace(req_to_token=torch.zeros((1, 64)))

    with pytest.raises(RuntimeError, match="context length"):
        _guard_realtime_context_capacity(SimpleNamespace(), state, segment, pool)

    small = SimpleNamespace(raw_append_ids=tuple(range(10)))
    _guard_realtime_context_capacity(SimpleNamespace(), state, small, pool)


def test_tp_frame_resolution_broadcasts_rank0_pixels(monkeypatch) -> None:
    import sglang_omni.models.moss_vl_realtime.scheduler as moss_scheduler
    from PIL import Image

    called: list[str] = []
    source = Image.new("RGB", (3, 2), color=(10, 20, 30))

    def _resolver(event):
        called.append(event.frame_ref)
        return source

    def _scheduler(tp_rank: int) -> MossVLRealtimeScheduler:
        scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
        scheduler.frame_resolver = _resolver
        scheduler.tp_size = 2
        scheduler.tp_rank = tp_rank
        scheduler.tp_group = SimpleNamespace(rank=tp_rank, ranks=[0, 1])
        scheduler.tp_cpu_group = object()
        return scheduler

    box: dict[str, Any] = {}

    def _root_broadcast(payload, rank, group, src):
        box["marker"] = payload
        return payload

    def _follower_broadcast(payload, rank, group, src):
        # The collective contract: non-root contributes None and receives
        # rank 0's payload.
        assert payload is None
        return box["marker"]

    event = FramePromptEvent(
        request_id="req-tp",
        session_id="session-tp",
        seq_no=0,
        timestamp=0.0,
        frame_ref="shm://frame-0",
    )

    root = _scheduler(tp_rank=0)
    monkeypatch.setattr(moss_scheduler, "broadcast_pyobj", _root_broadcast)
    images = root._resolve_frame_events_tp([event])
    assert called == ["shm://frame-0"]
    assert images[0].tobytes() == source.tobytes()

    monkeypatch.setattr(moss_scheduler, "broadcast_pyobj", _follower_broadcast)
    non_root = _scheduler(tp_rank=1)
    images = non_root._resolve_frame_events_tp([event])
    assert called == ["shm://frame-0"]  # non-root never opens the reference
    assert len(images) == 1
    assert images[0].size == (3, 2)
    assert images[0].mode == "RGB"
    assert images[0].tobytes() == source.tobytes()

    def _boom(event):
        raise FileNotFoundError("gone")

    root.frame_resolver = _boom
    monkeypatch.setattr(moss_scheduler, "broadcast_pyobj", _root_broadcast)
    with pytest.raises(RuntimeError, match="rank-0 frame resolution failed"):
        root._resolve_frame_events_tp([event])


def test_single_rank_resolution_uses_resolver_directly() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.tp_size = 1
    called: list[str] = []

    def _resolver(event):
        called.append(event.frame_ref)
        return object()

    scheduler.frame_resolver = _resolver
    event = FramePromptEvent(
        request_id="req-tp",
        session_id="session-tp",
        seq_no=0,
        timestamp=0.0,
        frame_ref="shm://frame-0",
    )
    images = scheduler._resolve_frame_events_tp([event])
    assert called == ["shm://frame-0"]
    assert len(images) == 1
