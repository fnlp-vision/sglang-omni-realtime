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


def test_decode_rate_gate_with_multiple_live_requests(monkeypatch) -> None:
    """Any due request releases the shared decode step; all-not-due defers."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)

    def _req(rid: str, deadline: float) -> SimpleNamespace:
        state = MossVLRealtimeRuntimeState(
            request_id=rid,
            session_id=f"session-{rid}",
            max_tokens_per_turn=4,
            next_decode_not_before=deadline,
        )
        return SimpleNamespace(_moss_vl_realtime_state=state)

    scheduler.running_batch = SimpleNamespace(
        reqs=[_req("req-a", 10.5), _req("req-b", 11.5)]
    )
    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    assert scheduler._decode_rate_limited() is True

    # req-a is due: the batch decodes even though req-b is still waiting.
    monkeypatch.setattr("time.monotonic", lambda: 10.5)
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


def test_materialize_extensions_batches_two_sessions(monkeypatch) -> None:
    """A running session and a parked session materialize into one extend batch."""
    import sglang_omni.models.moss_vl_realtime.scheduler as moss_scheduler

    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.realtime_sessions = MossVLRealtimeSessionController()
    scheduler.frame_window_config = None
    scheduler.enable_priority_scheduling = False
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = None
    scheduler.tree_cache = None
    scheduler.model_config = None
    scheduler.token_to_kv_pool_allocator = None
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=None)

    for rid in ("req-a", "req-b"):
        scheduler.realtime_sessions.open(rid, f"session-{rid}")
        scheduler.realtime_sessions.ingest(
            rid,
            FramePromptEvent(
                request_id=rid,
                session_id=f"session-{rid}",
                seq_no=0,
                timestamp=0.0,
                frame_ref=f"relay://{rid}-0",
            ),
        )

    def _req(rid: str, phase: MossVLRealtimePhase) -> SimpleNamespace:
        state = MossVLRealtimeRuntimeState(
            request_id=rid, session_id=f"session-{rid}", phase=phase
        )
        return SimpleNamespace(
            rid=rid, _moss_vl_realtime_state=state, finished=lambda: False
        )

    req_a = _req("req-a", MossVLRealtimePhase.DECODING)
    req_b = _req("req-b", MossVLRealtimePhase.WAITING_FOR_EVENT)

    filtered: dict[str, list[int]] = {}
    running_batch = SimpleNamespace(
        reqs=[req_a],
        batch_is_full=True,
        filter_batch=lambda *, keep_indices: filtered.update(
            keep_indices=list(keep_indices)
        ),
    )
    scheduler.running_batch = running_batch
    scheduler.parked_reqs = {"req-b": req_b}
    scheduler.parked_since = {"req-b": 1.0}

    scheduler._build_segment = lambda req, state, events: f"segment-{req.rid}"
    monkeypatch.setattr(
        moss_scheduler, "_guard_realtime_context_capacity", lambda *a: None
    )
    monkeypatch.setattr(moss_scheduler, "bind_realtime_page_row", lambda *a: None)
    monkeypatch.setattr(moss_scheduler, "_append_segment_to_request", lambda *a: None)
    monkeypatch.setattr(moss_scheduler, "PrefillStats", lambda **kw: ("stats", kw))
    monkeypatch.setattr(
        moss_scheduler.QueueCount, "from_reqs", staticmethod(lambda *a: None)
    )

    captured: dict[str, list] = {}
    fake_batch = SimpleNamespace(
        prepare_for_extend=lambda: None, extend_lens=[1, 1], prefix_lens=[0, 0]
    )

    def _init_new(*, reqs, **kwargs):
        captured["reqs"] = list(reqs)
        return fake_batch

    monkeypatch.setattr(
        moss_scheduler.MossVLRealtimeScheduleBatch, "init_new", _init_new
    )
    scheduler._emit_request_error = lambda rid, exc: (_ for _ in ()).throw(
        AssertionError(f"unexpected error for {rid}: {exc}")
    )
    scheduler.abort = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("unexpected abort")
    )

    batch = scheduler._materialize_realtime_extensions()

    assert batch is fake_batch
    # Both sessions extend in one batch; each contributed exactly its own event.
    assert [req.rid for req in captured["reqs"]] == ["req-a", "req-b"]
    # The parked session woke and left the parked tables.
    assert scheduler.parked_reqs == {}
    assert scheduler.parked_since == {}
    # The running session moved into the extend batch (dropped from running).
    assert filtered["keep_indices"] == []
    # Both queues drained.
    assert not scheduler.realtime_sessions.get("req-a").pending_events
    assert not scheduler.realtime_sessions.get("req-b").pending_events


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


def _pressure_req(
    rid: str, *, encoder_length: int, is_retracted: bool = False
) -> SimpleNamespace:
    state = MossVLRealtimeRuntimeState(
        request_id=rid,
        session_id=f"session-{rid}",
        encoder_length=encoder_length,
        decoder_length=2,
    )
    return SimpleNamespace(
        rid=rid,
        _moss_vl_realtime_state=state,
        is_retracted=is_retracted,
        output_ids=array("q", [201]),
        origin_input_ids=array("q", [101, 102]),
    )


def test_preempt_decode_memory_pressure_aborts_heaviest_session() -> None:
    """When the KV pool cannot fit the next decode round, the heaviest
    session is aborted through the realtime-aware path (not retracted)."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    heavy = _pressure_req("req-heavy", encoder_length=900)
    light = _pressure_req("req-light", encoder_length=100)

    class _Batch:
        def __init__(self) -> None:
            self.reqs = [light, heavy]

        def is_empty(self) -> bool:
            return not self.reqs

        def check_decode_mem(self) -> bool:
            # Fits only once the heavy session is gone.
            return heavy not in self.reqs

    scheduler.running_batch = _Batch()
    aborted: list[str] = []
    errors: list[str] = []

    def _abort(rid: str, *, defer_running_cleanup: bool = True) -> None:
        assert defer_running_cleanup is False
        aborted.append(rid)
        scheduler.running_batch.reqs = [
            r for r in scheduler.running_batch.reqs if r.rid != rid
        ]

    scheduler.abort = _abort
    scheduler._emit_request_error = lambda rid, exc: errors.append(rid)

    scheduler._preempt_decode_memory_pressure()

    assert aborted == ["req-heavy"]
    assert errors == ["req-heavy"]
    assert [r.rid for r in scheduler.running_batch.reqs] == ["req-light"]


def test_preempt_decode_memory_pressure_keeps_last_request() -> None:
    """A single remaining request is left to upstream's graceful OOM abort."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    only = _pressure_req("req-only", encoder_length=900)

    class _Batch:
        reqs = [only]

        def is_empty(self) -> bool:
            return False

        def check_decode_mem(self) -> bool:
            return False

    scheduler.running_batch = _Batch()
    scheduler.abort = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not abort the last remaining request")
    )
    scheduler._emit_request_error = lambda *a, **k: None

    scheduler._preempt_decode_memory_pressure()
    assert scheduler.running_batch.reqs == [only]


def test_abort_retracted_realtime_requests() -> None:
    """A realtime request that ended up retracted (e.g. SGLANG_TEST_RETRACT)
    is aborted cleanly from the waiting queue instead of being re-prefilled;
    non-realtime retracted requests are left for upstream."""
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    retracted = _pressure_req("req-retracted", encoder_length=50, is_retracted=True)
    normal = _pressure_req("req-normal", encoder_length=50)
    non_realtime = SimpleNamespace(
        rid="req-other", is_retracted=True, output_ids=array("q", [201])
    )
    scheduler.waiting_queue = [retracted, normal, non_realtime]
    aborted: list[str] = []
    errors: list[str] = []
    scheduler.abort = lambda rid, **kw: aborted.append(rid)
    scheduler._emit_request_error = lambda rid, exc: errors.append(rid)

    scheduler._abort_retracted_realtime_requests()

    assert aborted == ["req-retracted"]
    assert errors == ["req-retracted"]


def test_abort_retracted_realtime_requests_noop_when_empty() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.waiting_queue = []
    scheduler.abort = lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    scheduler._abort_retracted_realtime_requests()


def _staged_candidate(rid: str, *, kv_cost: int):
    """A request with a segment already staged via _append_segment_to_request.

    The append requires state lengths consistent with the tiny fixture, so the
    "committed" encoder length used for victim ordering is applied afterwards.
    """
    req = _Req()
    req.rid = rid
    state = MossVLRealtimeRuntimeState(
        request_id=rid,
        session_id=f"session-{rid}",
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
    state.encoder_length = kv_cost - state.decoder_length
    return req, [segment.events[0]], segment


class _FakeAllocator:
    def __init__(self, available: int) -> None:
        self._available = available

    def available_size(self) -> int:
        return self._available


def test_enforce_extend_memory_budget_noop_when_fits() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.token_to_kv_pool_allocator = _FakeAllocator(available=10_000)
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.parked_reqs = {}
    scheduler.abort = lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    scheduler._emit_request_error = lambda *a, **k: None
    selected = [_staged_candidate("req-a", kv_cost=100)]
    assert scheduler._enforce_extend_memory_budget(selected) == selected


def test_enforce_extend_memory_budget_aborts_heaviest_and_undoes() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    # Two candidates need (4+2)*2 = 12 tokens; only 8 available -> the
    # heaviest session must be aborted and its staged append undone.
    scheduler.token_to_kv_pool_allocator = _FakeAllocator(available=8)
    heavy = _staged_candidate("req-heavy", kv_cost=900)
    light = _staged_candidate("req-light", kv_cost=100)
    scheduler.running_batch = SimpleNamespace(reqs=[heavy[0], light[0]])
    scheduler.parked_reqs = {}
    aborted: list[str] = []
    errors: list[str] = []

    def _abort(rid: str, *, defer_running_cleanup: bool = True) -> None:
        assert defer_running_cleanup is False
        aborted.append(rid)

    scheduler.abort = _abort
    scheduler._emit_request_error = lambda rid, exc: errors.append(rid)

    remaining = scheduler._enforce_extend_memory_budget([heavy, light])

    assert aborted == ["req-heavy"]
    assert errors == ["req-heavy"]
    assert [entry[0].rid for entry in remaining] == ["req-light"]
    # The victim's staged append was rolled back (token-level undo).
    assert heavy[0].output_ids.tolist() == [201]
    assert heavy[0].sampling_params.max_new_tokens == 10
    assert heavy[0]._moss_vl_realtime_state.pending_token_id is None
    # The survivor keeps its staged segment untouched.
    assert light[0].output_ids.tolist() == [201, -101, -101, 301, 302]


def test_enforce_extend_memory_budget_can_victim_parked_session() -> None:
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.token_to_kv_pool_allocator = _FakeAllocator(available=4)
    light = _staged_candidate("req-light", kv_cost=50)
    parked = _pressure_req("req-parked", encoder_length=800)
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.parked_reqs = {"req-parked": parked}
    aborted: list[str] = []
    scheduler.abort = lambda rid, **kw: aborted.append(rid)
    scheduler._emit_request_error = lambda rid, exc: None

    remaining = scheduler._enforce_extend_memory_budget([light])

    assert aborted == ["req-parked"]
    assert [entry[0].rid for entry in remaining] == ["req-light"]


def _pool_args(*, size: int, context_length: int, max_running: int = 1):
    allocator = SimpleNamespace(size=size)
    server_args = SimpleNamespace(
        context_length=context_length, max_running_requests=max_running
    )
    return allocator, server_args


def test_validate_kv_pool_capacity_rejects_pool_smaller_than_context() -> None:
    allocator, server_args = _pool_args(size=100_000, context_length=262_144)
    with pytest.raises(ValueError, match="mem-fraction-static"):
        MossVLRealtimeScheduler._validate_kv_pool_capacity(allocator, server_args)


def test_validate_kv_pool_capacity_warns_on_overcommit(caplog) -> None:
    # Pool holds 300k tokens, two sessions of 262k each -> legal overcommit,
    # but must warn so the operator knows degradation is heaviest-first.
    allocator, server_args = _pool_args(
        size=300_000, context_length=262_144, max_running=2
    )
    with caplog.at_level("WARNING"):
        MossVLRealtimeScheduler._validate_kv_pool_capacity(allocator, server_args)
    assert any("cannot hold" in record.message for record in caplog.records)


def test_validate_kv_pool_capacity_passes_when_pool_is_sufficient() -> None:
    allocator, server_args = _pool_args(size=600_000, context_length=262_144)
    MossVLRealtimeScheduler._validate_kv_pool_capacity(allocator, server_args)


def test_validate_kv_pool_capacity_skips_when_values_unknown() -> None:
    allocator, server_args = _pool_args(size=0, context_length=0)
    MossVLRealtimeScheduler._validate_kv_pool_capacity(allocator, server_args)
