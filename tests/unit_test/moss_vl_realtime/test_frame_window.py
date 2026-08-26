from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    FramePromptEvent,
    MossVLRealtimeRuntimeState,
    RealtimeFrameRecord,
    RealtimeFrameWindowConfig,
    apply_frame_window_plan,
    plan_frame_window,
    stage_segment_frame_records,
)
from sglang_omni.models.moss_vl_realtime.frame_window import (
    DEFAULT_POOL_RATIO,
    DEFAULT_POOL_WINDOW_S,
    DEFAULT_RAW_WINDOW_S,
    FRAME_RECORDS_STAGED_ATTR,
    covered_spans,
    visible_count_remap,
)


def _record(ts: float, *, slots: int = 2, pooled: bool = False, sources: int = 1):
    return RealtimeFrameRecord(
        timestamp=ts,
        grid_h=2,
        grid_w=2,
        slots=slots,
        pooled=pooled,
        pooled_sources=sources,
    )


def _config(**overrides) -> RealtimeFrameWindowConfig:
    base: dict = {
        "enabled": True,
        "raw_window_s": 35.0,
        "pool_window_s": 120.0,
        "pool_ratio": 2,
    }
    base.update(overrides)
    return RealtimeFrameWindowConfig(**base)


class _FakeAllocator:
    def __init__(self, free_ids: list[int], *, fail_alloc: bool = False) -> None:
        self._free = list(free_ids)
        self._fail_alloc = fail_alloc
        self.freed: list[list[int]] = []

    def alloc(self, n: int):
        if self._fail_alloc or n > len(self._free):
            return None
        out = torch.tensor(self._free[:n], dtype=torch.int64)
        self._free = self._free[n:]
        return out

    def free(self, slots) -> None:
        self.freed.extend(int(v) for v in slots.tolist())

    def available_size(self) -> int:
        return len(self._free)


class _FakeKVPool:
    def __init__(self, size: int, layers: int = 2, heads: int = 2, dim: int = 4):
        self.k_buffer = [torch.zeros(size, heads, dim) for _ in range(layers)]
        self.v_buffer = [torch.zeros(size, heads, dim) for _ in range(layers)]


def _enabled_state(records, *, grid_rows=None):
    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
        encoder_length=sum(r.slots for r in records),
        decoder_length=3,
        visible_frame_count=len(records),
    )
    state.frame_records = list(records)
    rows = grid_rows if grid_rows is not None else [r.grid_row for r in records]
    state.full_grid_thw = torch.tensor(rows, dtype=torch.long)
    return state


def _harness(records, *, allocator_ids=(900, 901, 902, 903, 904, 905)):
    """One request with 2-slot frames and a 3-token decoder tail."""
    state = _enabled_state(records)
    req_to_token = torch.zeros((1, 64), dtype=torch.int64)
    encoder_slots = list(range(100, 100 + state.encoder_length))
    decoder_slots = [50, 60, 70]
    req_to_token[0, : len(encoder_slots) + 3] = torch.tensor(
        encoder_slots + decoder_slots
    )
    allocator = _FakeAllocator(list(allocator_ids))
    pool = _FakeKVPool(size=1024)
    req = SimpleNamespace(
        rid="req-fw",
        multimodal_inputs=SimpleNamespace(
            visible_frame_counts=None,
            num_image_tokens=state.encoder_length,
            media_nums_per_sample=[state.full_grid_thw.shape[0]],
            mm_items=[
                SimpleNamespace(model_specific_data={"realtime_full_grid_thw": None})
            ],
        ),
        kv=SimpleNamespace(kv_allocated_len=state.encoder_length + 3),
        kv_committed_len=state.encoder_length + 3,
    )
    return state, req, req_to_token, allocator, pool


# --- config resolution -----------------------------------------------------


def test_config_defaults_disabled() -> None:
    config = RealtimeFrameWindowConfig.resolve(env={})
    assert config.enabled is False
    assert config.raw_window_s == DEFAULT_RAW_WINDOW_S == 60.0
    assert config.pool_window_s == DEFAULT_POOL_WINDOW_S == 240.0
    assert config.pool_ratio == DEFAULT_POOL_RATIO == 4


def test_config_env_overrides_explicit_values() -> None:
    config = RealtimeFrameWindowConfig.resolve(
        enabled=False,
        raw_window_s=5.0,
        pool_window_s=50.0,
        pool_ratio=2,
        env={
            "REALTIME_FRAME_WINDOW_ENABLED": "1",
            "REALTIME_FRAME_WINDOW_RAW_S": "45",
            "REALTIME_FRAME_POOL_WINDOW_S": "300",
            "REALTIME_FRAME_POOL_RATIO": "3",
        },
    )
    assert config.enabled is True
    assert config.raw_window_s == 45.0
    assert config.pool_window_s == 300.0
    assert config.pool_ratio == 3


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="raw_window_s"):
        RealtimeFrameWindowConfig(enabled=True, raw_window_s=0)
    with pytest.raises(ValueError, match="pool_ratio"):
        RealtimeFrameWindowConfig(enabled=True, pool_ratio=1)
    with pytest.raises(ValueError, match="boolean"):
        RealtimeFrameWindowConfig.resolve(
            env={"REALTIME_FRAME_WINDOW_ENABLED": "maybe"}
        )


# --- planning ---------------------------------------------------------------


def test_plan_noop_when_disabled_or_within_window() -> None:
    records = [_record(0.0), _record(10.0), _record(20.0)]
    assert plan_frame_window(records, RealtimeFrameWindowConfig()) is None
    # Newest=20, oldest=0: span 20 < window 35 -> nothing ages out.
    assert plan_frame_window(records, _config()) is None


def test_plan_raw_window_boundary_is_strictly_greater() -> None:
    records = [_record(0.0), _record(35.0)]
    # Span exactly equals the window: the oldest frame is kept.
    assert plan_frame_window(records, _config()) is None
    records = [_record(0.0), _record(1.0), _record(36.0 + 1e-6)]
    plan = plan_frame_window(records, _config())
    assert plan is not None
    assert plan.pooled_raw_count == 2
    assert plan.dropped_raw_count == 0
    # One full ratio group folds into a virtual copy ahead of the raw region.
    assert [r.timestamp for r in plan.new_records] == [1.0, 36.0 + 1e-6]
    assert plan.new_records[0].pooled is True
    assert plan.new_records[0].pooled_sources == 2
    assert plan.new_records[1].pooled is False


def test_plan_holds_aged_remainder_until_ratio_group_fills() -> None:
    """1fps aging produces one aged frame per evaluation; pooling partial
    tail chunks would compress 1:1 forever. The remainder stays raw (a soft
    overshoot of at most ratio-1 frames) until a full group is available."""
    config = _config(raw_window_s=10.0, pool_ratio=4)
    # 33 frames 1s apart; newest=32: frames with 32-ts>10 (ts<22) are aged.
    records = [_record(float(i)) for i in range(33)]
    plan = plan_frame_window(records, config)
    assert plan is not None
    # 22 aged, floor to 4 full groups; ts 20 and 21 stay raw beyond window.
    assert plan.raw_chunks == ((0, 4, True), (4, 4, True), (8, 4, True), (12, 4, True), (16, 4, True))
    assert plan.pooled_raw_count == 20
    assert plan.produced_virtual_count == 5
    kept_raw = [r.timestamp for r in plan.new_records if not r.pooled]
    assert kept_raw == [float(i) for i in range(20, 33)]
    assert len(plan.new_records) == 5 + 13
    produced = [r for r in plan.new_records if r.pooled]
    assert [r.pooled_sources for r in produced] == [4] * 5
    assert [r.timestamp for r in produced] == [3.0, 7.0, 11.0, 15.0, 19.0]

    # With only 3 aged frames (< ratio), nothing pools yet.
    records = [_record(float(i)) for i in range(14)]
    assert plan_frame_window(records, config) is None


def test_plan_pools_aged_frames_into_virtual_prefix() -> None:
    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    plan = plan_frame_window(records, _config())
    assert plan is not None
    # Newest raw = 50; frames trailing by >35s are 0 and 10.
    assert plan.raw_chunks == ((0, 2, True),)
    assert plan.pooled_raw_count == 2
    assert plan.produced_virtual_count == 1
    assert plan.evicted_virtual_count == 0
    timestamps = [r.timestamp for r in plan.new_records]
    assert timestamps == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert plan.new_records[0].pooled is True
    assert plan.new_records[0].pooled_sources == 2
    assert all(not r.pooled for r in plan.new_records[1:])


def test_plan_dropped_chunks_for_mixed_grids() -> None:
    mixed = [
        _record(0.0, slots=2),
        RealtimeFrameRecord(timestamp=10.0, grid_h=4, grid_w=4, slots=5),
        _record(20.0),
        _record(50.0),
    ]
    config = _config(raw_window_s=35.0)
    plan = plan_frame_window(mixed, config)
    assert plan is not None
    # Chunk (0,10) has mixed grids: dropped, not pooled.
    assert plan.raw_chunks == ((0, 2, False),)
    assert plan.dropped_raw_count == 2
    assert plan.produced_virtual_count == 0
    assert [r.timestamp for r in plan.new_records] == [20.0, 50.0]


def test_plan_virtual_window_boundary() -> None:
    records = [
        _record(0.0, pooled=True, sources=2),
        _record(100.0, pooled=True, sources=2),
        _record(200.0, pooled=True, sources=2),
        _record(300.0),
    ]
    # 200 - 0 > 120 -> oldest virtual evicted.
    plan = plan_frame_window(records, _config())
    assert plan is not None
    assert plan.evicted_virtual_count == 1
    assert [r.timestamp for r in plan.new_records] == [100.0, 200.0, 300.0]

    boundary = [
        _record(80.0, pooled=True, sources=2),
        _record(200.0, pooled=True, sources=2),
        _record(300.0),
    ]
    # Span exactly 120 == window: kept (strictly-greater eviction).
    assert plan_frame_window(boundary, _config()) is None


def test_plan_rejects_pooled_record_after_raw() -> None:
    records = [_record(0.0), _record(10.0, pooled=True, sources=2)]
    with pytest.raises(RuntimeError, match="leading encoder region"):
        plan_frame_window(records, _config())


# --- record staging ---------------------------------------------------------


def test_stage_segment_frame_records_orders_and_spans() -> None:
    events = [
        FramePromptEvent(
            request_id="r",
            session_id="s",
            seq_no=1,
            timestamp=20.0,
            frame_ref="x://b",
        ),
        FramePromptEvent(
            request_id="r",
            session_id="s",
            seq_no=0,
            timestamp=5.0,
            frame_ref="x://a",
        ),
    ]
    segment = SimpleNamespace(
        events=events,
        full_grid_thw=torch.tensor([[1, 2, 2], [1, 4, 4]], dtype=torch.long),
    )
    records = stage_segment_frame_records(segment, merge_size=2)
    assert records is not None
    assert [r.timestamp for r in records] == [5.0, 20.0]
    # 2x2 grid with merge 2 -> 1 vision token + separator = 2 slots.
    assert records[0].slots == 2
    # 4x4 grid -> 4 vision tokens + separator = 5 slots.
    assert records[1].slots == 5
    assert all(not r.pooled for r in records)


def test_stage_segment_frame_records_none_for_text_only() -> None:
    event = FramePromptEvent(
        request_id="r",
        session_id="s",
        seq_no=0,
        timestamp=0.0,
        frame_ref=None,
        prompt="hello",
    )
    segment = SimpleNamespace(
        events=[event], full_grid_thw=torch.empty((0, 3), dtype=torch.long)
    )
    assert stage_segment_frame_records(segment, merge_size=2) is None


# --- metadata remap ---------------------------------------------------------


def test_visible_count_remap_pooled_group() -> None:
    spans = ((0, 2), (2, 3), (3, 4))  # rows 0-1 pooled to one row
    assert visible_count_remap(4, spans) == [0, 1, 1, 2, 3]


def test_visible_count_remap_identity_gaps_and_partial_visibility() -> None:
    # Identity spans leave counts untouched.
    assert visible_count_remap(3, ((0, 1), (1, 2), (2, 3))) == [0, 1, 2, 3]
    # Leading rows dropped (e.g. evicted virtuals / non-poolable chunks).
    assert visible_count_remap(4, ((2, 4),)) == [0, 0, 0, 1, 1]
    # A token sees the pooled virtual once it saw any member of the chunk.
    assert visible_count_remap(4, ((0, 2), (2, 4))) == [0, 1, 1, 2, 2]


def test_covered_spans_mixed_layout() -> None:
    records = [
        _record(0.0, pooled=True, sources=2),
        _record(10.0, pooled=True, sources=2),
        _record(20.0),
        _record(23.0),
        _record(27.0),
        _record(40.0),
    ]
    plan = plan_frame_window(
        records, _config(raw_window_s=15.0, pool_window_s=10.0)
    )
    assert plan is not None
    # Raw window 15: aged raw = 20 and 23 (40-23=17>15; 40-27=13 stays).
    # One full chunk (2 members) pooled -> produced virtual ts=23. Virtuals
    # 0,10,23 with pool window 10: 23-0=23>10 evict 0, 23-10=13>10 evict 1.
    assert plan.evicted_virtual_count == 2
    spans = covered_spans(records, plan)
    # new_records: produced virtual covering old raw rows 2-3, kept raws 4,5.
    assert spans == ((2, 4), (4, 5), (5, 6))
    assert [r.timestamp for r in plan.new_records] == [23.0, 27.0, 40.0]


# --- apply: page-table compaction + pooling ---------------------------------


def test_apply_pools_two_frames_and_compacts_row() -> None:
    records = [_record(t) for t in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    state.visible_frame_counts = torch.tensor(
        [0, 3, 5], dtype=torch.int32
    )
    req.multimodal_inputs.visible_frame_counts = torch.tensor(
        [3, 5], dtype=torch.int32
    )
    # Fill the members' K/V with recognizable per-slot values.
    for layer in range(2):
        for slot in (100, 101):  # frame ts=0
            pool.k_buffer[layer][slot] = 1.0
            pool.v_buffer[layer][slot] = 2.0
        for slot in (102, 103):  # frame ts=10
            pool.k_buffer[layer][slot] = 3.0
            pool.v_buffer[layer][slot] = 4.0

    plan = plan_frame_window(records, _config())
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )

    assert (event.pooled_raw_frames, event.produced_virtual_frames) == (2, 1)
    assert (event.encoder_length_before, event.encoder_length_after) == (12, 10)
    # Pooled into fresh slots 900-901; rows: virtual, raws 20..50, decoder.
    expected = [900, 901, 104, 105, 106, 107, 108, 109, 110, 111, 50, 60, 70]
    assert req_to_token[0, :13].tolist() == expected
    assert req_to_token[0, 13:].count_nonzero().item() == 0
    assert sorted(allocator.freed) == [100, 101, 102, 103]
    # Pooling math: mean of the two members, per layer, at the dst slots.
    assert torch.allclose(pool.k_buffer[0][900], torch.full((2, 4), 2.0))
    assert torch.allclose(pool.v_buffer[1][901], torch.full((2, 4), 3.0))
    # Metadata resync.
    assert state.encoder_length == 10
    assert state.visible_frame_count == 5
    assert state.evicted_frame_count == 0
    assert state.full_grid_thw.tolist() == [[1, 2, 2]] * 5
    # Kept-count remap: K = [0,1,1,2,3,4,5]; [0,3,5] -> [0,2,4].
    assert state.visible_frame_counts.tolist() == [0, 2, 4]
    assert req.multimodal_inputs.visible_frame_counts.tolist() == [2, 4]
    assert req.multimodal_inputs.num_image_tokens == 10
    assert req.multimodal_inputs.media_nums_per_sample == [5]
    assert (
        req.multimodal_inputs.mm_items[0]
        .model_specific_data["realtime_full_grid_thw"]
        .tolist()
        == [[1, 2, 2]] * 5
    )
    # The state and request copies of the counts stay distinct objects.
    assert state.visible_frame_counts is not req.multimodal_inputs.visible_frame_counts
    assert req.kv.kv_allocated_len == 13
    assert req.kv_committed_len == 13
    assert [r.pooled for r in state.frame_records] == [True, False, False, False, False]
    assert event.pool_free_slots == allocator.available_size()


def test_apply_degrades_to_eviction_when_alloc_fails() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    allocator._fail_alloc = True

    plan = plan_frame_window(records, _config())
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert event.produced_virtual_frames == 0
    assert event.dropped_raw_frames == 2
    assert event.evicted_virtual_frames == 0
    assert event.encoder_length_after == 2
    assert req_to_token[0, :5].tolist() == [104, 105, 50, 60, 70]
    assert sorted(allocator.freed) == [100, 101, 102, 103]
    assert state.evicted_frame_count == 2
    assert state.full_grid_thw.tolist() == [[1, 2, 2]]


def test_apply_evicts_virtuals_and_counts_pooled_sources() -> None:
    records = [
        _record(0.0, pooled=True, sources=4),
        _record(200.0, pooled=True, sources=4),
        _record(300.0),
        _record(310.0),
    ]
    state, req, req_to_token, allocator, pool = _harness(records)
    plan = plan_frame_window(records, _config())  # 200-0 > 120 -> evict oldest
    assert plan is not None
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
    )
    assert event.evicted_virtual_frames == 1
    # Evicting the oldest virtual frame counts its 4 original frames.
    assert state.evicted_frame_count == 4
    assert req_to_token[0, :9].tolist() == [102, 103, 104, 105, 106, 107, 50, 60, 70]
    assert sorted(allocator.freed) == [100, 101]
    assert state.frame_records[0].pooled is True


def test_apply_updates_running_batch_encoder_lens() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    state, req, req_to_token, allocator, pool = _harness(records)
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="other"), req],
        encoder_lens_cpu=[7, 6],
        encoder_lens=torch.tensor([7, 6]),
    )
    plan = plan_frame_window(records, _config())
    apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: pool,
        running_batch=batch,
    )
    assert batch.encoder_lens_cpu == [7, 4]
    assert batch.encoder_lens.tolist() == [7, 4]


def test_apply_rejects_mismatched_record_spans() -> None:
    records = [_record(0.0, slots=3), _record(1.0, slots=3), _record(50.0)]
    state, req, req_to_token, allocator, pool = _harness([_record(t) for t in (0.0, 1.0, 50.0)])
    plan = plan_frame_window(records, _config())
    assert plan is not None
    with pytest.raises(RuntimeError, match="disagree with the encoder"):
        apply_frame_window_plan(
            req,
            state,
            plan,
            records=records,
            req_to_token=req_to_token,
            allocator=allocator,
            kv_pool_provider=lambda: pool,
        )


# --- scheduler hook ---------------------------------------------------------


def _hook_scheduler(records, config):
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        MossVLRealtimeScheduler,
    )

    state, req, req_to_token, allocator, pool = _harness(records)
    req._moss_vl_realtime_state = state
    req.finished = lambda: False
    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.frame_window_config = config
    scheduler.running_batch = SimpleNamespace(reqs=[req])
    scheduler.parked_reqs = {}
    scheduler._realtime_extend_batch = None
    scheduler._async_pending = None
    scheduler.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler._token_to_kv_pool = lambda: pool
    return scheduler, state, req, allocator


def test_scheduler_hook_noop_when_window_disabled() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, None)
    scheduler._evaluate_frame_window()
    assert allocator.freed == []
    assert state.encoder_length == 6


def test_scheduler_hook_evicts_during_decode_gap() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, _config())
    scheduler._evaluate_frame_window()
    assert state.encoder_length == 4  # 0 and 10 pooled into one virtual frame
    assert sorted(allocator.freed) == [100, 101, 102, 103]


def test_scheduler_hook_skips_mid_transaction() -> None:
    records = [_record(t) for t in (0.0, 10.0, 50.0)]
    scheduler, state, _req, allocator = _hook_scheduler(records, _config())
    state._append_inflight = True
    scheduler._evaluate_frame_window()
    assert allocator.freed == []
    assert state.encoder_length == 6


# --- model-runner record consumption --------------------------------------


def test_post_prefill_consumes_staged_frame_records() -> None:
    from sglang_omni.models.moss_vl_realtime.model_runner import (
        MossVLRealtimeModelRunner,
    )

    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
    )
    req = SimpleNamespace()
    req._moss_vl_realtime_state = state
    records = [_record(0.0)]
    setattr(req, FRAME_RECORDS_STAGED_ATTR, records)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.frame_records == records
    assert not hasattr(req, FRAME_RECORDS_STAGED_ATTR)


def test_post_prefill_without_records_is_unchanged() -> None:
    from sglang_omni.models.moss_vl_realtime.model_runner import (
        MossVLRealtimeModelRunner,
    )

    state = MossVLRealtimeRuntimeState(
        request_id="req-fw",
        session_id="session-fw",
        req_pool_index=0,
    )
    req = SimpleNamespace()
    req._moss_vl_realtime_state = state
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.frame_records is None


# --- regression: extend materialization after a virtual-frame eviction ------


def _post_eviction_harness():
    """State as if 3 frames were appended and one virtual frame evicted.

    Token space keeps the evicted frame's pad placeholders
    (appended_encoder_length=6) while the KV encoder region shrank to 4 slots.
    """
    from array import array
    from collections import namedtuple

    Range = namedtuple("Range", ["start", "end"])
    Range.length = property(lambda self: self.end - self.start)

    records = [
        _record(0.0, pooled=True, sources=2),
        _record(3.0, pooled=True, sources=2),
        _record(10.0),
    ]
    state = _enabled_state(records)
    state.decoder_length = 2
    state.appended_encoder_length = 6
    req_to_token = torch.zeros((1, 32), dtype=torch.int64)
    req_to_token[0, :8] = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22])
    allocator = _FakeAllocator([])
    req = SimpleNamespace(
        rid="req-fw",
        kv=SimpleNamespace(kv_allocated_len=8),
        kv_committed_len=8,
        multimodal_inputs=SimpleNamespace(
            visible_frame_counts=None,
            num_image_tokens=6,
            media_nums_per_sample=[3],
            mm_items=[
                SimpleNamespace(model_specific_data={"realtime_full_grid_thw": None})
            ],
        ),
    )
    req._moss_vl_realtime_state = state
    req._moss_vl_realtime_page_row = req_to_token[0]

    # Fill ids: origin tokens + token-space history + one pending sampled id.
    req.origin_input_ids = array("q", [101])
    token_history = array(
        "q", [-101] * 6  # two slots per historical frame (incl. evicted)
    )
    req.output_ids = token_history + array("q", [5, 999])  # 1 text + pending
    req.sampling_params = SimpleNamespace(max_new_tokens=16)
    req.skip_radix_cache_insert = False
    req.prefix_indices = torch.tensor([11, 12, 13, 14, 15, 16, 21, 22])
    req.extend_range = Range(0, 9)

    def _refresh():
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids

    req._refresh_fill_ids = _refresh
    req.set_extend_range = lambda start, end: setattr(req, "extend_range", Range(start, end))
    _refresh()
    return state, req, req_to_token, allocator, records, Range


def test_extend_materialization_after_virtual_eviction() -> None:
    """Dropping a virtual frame must not break the token-space invariants the
    next realtime extend relies on (previously raised "must coalesce with
    exactly one pending sampled token")."""
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        _append_segment_to_request,
    )

    state, req, req_to_token, allocator, records, Range = _post_eviction_harness()
    assert len(req.full_untruncated_fill_ids) == 9  # 1 origin + 6 pads + 2 + pending

    # pool window 2s: newest virtual ts=3, oldest ts=0 trails by >2 -> evicted.
    plan = plan_frame_window(records, _config(raw_window_s=100.0, pool_window_s=2.0))
    assert plan is not None and plan.evicted_virtual_count == 1
    event = apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: None,
    )
    assert event.encoder_length_after == 4
    assert allocator.freed == [11, 12]
    assert state.encoder_length == 4
    assert state.effective_appended_encoder_length == 6

    # The next drain materializes on top of the compacted row without raising.
    event_frame = FramePromptEvent(
        request_id="req-fw",
        session_id="session-fw",
        seq_no=3,
        timestamp=11.0,
        frame_ref="x://c",
    )
    segment = SimpleNamespace(
        events=(event_frame,),
        raw_append_ids=(-101, -101, 301),  # 2 encoder pads + 1 text token
        full_grid_thw=torch.tensor([[1, 2, 2]] * 3),
        multimodal_inputs=SimpleNamespace(
            mrope_positions=torch.tensor([[0, 1, 7], [0, 1, 7], [0, 1, 7]]),
            visible_frame_counts=torch.tensor([0, 1, 2], dtype=torch.int32),
        ),
    )
    _append_segment_to_request(req, state, segment)

    # Token space: 6 pads + 2 text committed; extend covers pending + appends.
    assert req.extend_range == Range(8, 12)
    assert len(req.prefix_indices) == 8
    assert req.prefix_indices[:6].tolist() == [13, 14, 15, 16, 21, 22]
    assert state.pending_token_id == 999
    # Committed KV region is untouched by the token-space bookkeeping.
    assert req_to_token[0, :6].tolist() == [13, 14, 15, 16, 21, 22]


def test_build_segment_pending_check_uses_token_space() -> None:
    """_build_segment must count the pending token against token history, not
    the compacted KV encoder length."""
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        MossVLRealtimeScheduler,
        _guard_realtime_context_capacity,
    )

    state, req, req_to_token, allocator, records, _ = _post_eviction_harness()
    plan = plan_frame_window(records, _config(raw_window_s=100.0, pool_window_s=2.0))
    apply_frame_window_plan(
        req,
        state,
        plan,
        records=records,
        req_to_token=req_to_token,
        allocator=allocator,
        kv_pool_provider=lambda: None,
    )

    captured: dict = {}

    class _FakeBuilder:
        def build(self, events, images, **common):
            captured.update(common)
            return "segment"

    scheduler = MossVLRealtimeScheduler.__new__(MossVLRealtimeScheduler)
    scheduler.segment_builder = _FakeBuilder()
    scheduler._resolve_frame_events_tp = lambda events: [object()] * len(events)

    events = [
        FramePromptEvent(
            request_id="req-fw",
            session_id="session-fw",
            seq_no=3,
            timestamp=11.0,
            frame_ref="x://c",
        )
    ]
    assert scheduler._build_segment(req, state, events) == "segment"
    # KV-space values still flow to the segment builder for mask sizing.
    assert captured["committed_encoder_length"] == 4
    assert captured["committed_decoder_length"] == 2

    # Context guard covers the wider of token/KV spaces: fill ids will reach
    # 8 tokens committed + 3 appends, so an 11-wide row is too small.
    pool_ns = SimpleNamespace(req_to_token=torch.zeros((1, 11)))
    with pytest.raises(RuntimeError, match="context length"):
        _guard_realtime_context_capacity(
            req,
            state,
            SimpleNamespace(raw_append_ids=(-101, -101, 301)),
            pool_ns,
        )
    _guard_realtime_context_capacity(
        req,
        state,
        SimpleNamespace(raw_append_ids=(-101, -101, 301)),
        SimpleNamespace(req_to_token=torch.zeros((1, 12))),
    )


def test_guard_without_eviction_uses_kv_space_as_before() -> None:
    state = _enabled_state([_record(0.0), _record(1.0)])
    state.decoder_length = 2
    from sglang_omni.models.moss_vl_realtime.scheduler import (
        _guard_realtime_context_capacity,
    )

    # token == KV (no eviction): 4+2+3+1 = 10 needs a 10-wide row.
    segment = SimpleNamespace(raw_append_ids=(-101, -101, 301))
    _guard_realtime_context_capacity(
        SimpleNamespace(), state, segment, SimpleNamespace(req_to_token=torch.zeros((1, 10)))
    )
    with pytest.raises(RuntimeError, match="context length"):
        _guard_realtime_context_capacity(
            SimpleNamespace(), state, segment, SimpleNamespace(req_to_token=torch.zeros((1, 9)))
        )
