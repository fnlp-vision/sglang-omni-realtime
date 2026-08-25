from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.moss_vl_realtime import (
    RUNTIME_STATE_ATTR,
    MossVLRealtimeModelRunner,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
    MossVLRealtimeScheduleBatch,
)


def test_realtime_schedule_batch_owns_incremental_encoder_hooks() -> None:
    assert "prepare_encoder_info_extend" in MossVLRealtimeScheduleBatch.__dict__
    assert "prepare_for_extend" in MossVLRealtimeScheduleBatch.__dict__


def test_model_runner_advances_committed_decode_state() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-1",
        session_id="session-1",
        encoder_length=5,
        decoder_length=8,
        next_mrope_position=11,
        phase=MossVLRealtimePhase.DECODING,
    )
    req = SimpleNamespace()
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_decode(None, None, batch, [])

    assert state.decoder_length == 9
    assert state.next_mrope_position == 12
    assert state.phase is MossVLRealtimePhase.DECODING


def test_model_runner_commits_cumulative_realtime_metadata() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-meta",
        session_id="session-meta",
        req_pool_index=0,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_staged_mrope_positions="mrope",
        _moss_vl_realtime_staged_visible_frame_counts="visible",
        _moss_vl_realtime_staged_full_grid_thw="grid",
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req])
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.mrope_positions == "mrope"
    assert state.visible_frame_counts == "visible"
    assert state.full_grid_thw == "grid"
    assert not hasattr(req, "_moss_vl_realtime_staged_full_grid_thw")


def test_model_runner_initializes_committed_text_prefill_state() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-initial",
        session_id="session-initial",
    )
    req = SimpleNamespace(req_pool_idx=3)
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([7]))
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.req_pool_index == 3
    assert state.encoder_length == 0
    assert state.decoder_length == 7
    assert state.next_mrope_position == 7
    assert state.phase is MossVLRealtimePhase.DECODING


def test_model_runner_commits_prompt_turn_transition() -> None:
    state = MossVLRealtimeRuntimeState(
        request_id="req-turn",
        session_id="session-turn",
        req_pool_index=0,
        turn_id=2,
    )
    req = SimpleNamespace(
        _moss_vl_realtime_staged_event={
            "seq_no": 7,
            "timestamp": 4.0,
            "prompt": "What changed?",
            "final": False,
        },
        _moss_vl_realtime_staged_turn_transition={
            "interrupted_turn_id": 2,
            "turn_id": 3,
        },
    )
    setattr(req, RUNTIME_STATE_ATTR, state)
    batch = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([9]))
    runner = MossVLRealtimeModelRunner.__new__(MossVLRealtimeModelRunner)

    runner.post_prefill(None, None, batch, [])

    assert state.turn_id == 3
    assert req._moss_vl_realtime_processed_event["interrupted_turn_id"] == 2
    assert req._moss_vl_realtime_processed_event["turn_id"] == 3
    assert not hasattr(req, "_moss_vl_realtime_staged_turn_transition")
