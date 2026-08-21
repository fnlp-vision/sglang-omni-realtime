from __future__ import annotations

from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch

from sglang_omni.models.moss_vl_realtime import (
    RUNTIME_STATE_ATTR,
    MossVLRealtimeModelRunner,
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.sglang_patch import (
    install_moss_vl_realtime_paged_decode_alloc_patch,
    install_moss_vl_realtime_schedule_batch_patch,
)


def test_schedule_batch_patch_is_idempotent_and_preserves_original() -> None:
    install_moss_vl_realtime_schedule_batch_patch()
    first = ScheduleBatch.prepare_encoder_info_extend
    first_full_prepare = ScheduleBatch.prepare_for_extend
    install_moss_vl_realtime_schedule_batch_patch()

    assert ScheduleBatch.prepare_encoder_info_extend is first
    assert ScheduleBatch.prepare_for_extend is first_full_prepare
    assert first._sglang_omni_patch_owner == "sglang_omni.moss_vl_realtime"
    assert callable(first._sglang_omni_original)
    assert first_full_prepare._sglang_omni_patch_owner == (
        "sglang_omni.moss_vl_realtime"
    )
    assert callable(first_full_prepare._sglang_omni_original)


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


def test_paged_decode_alloc_patch_keys_on_total_row_length(monkeypatch) -> None:
    import torch
    from sglang.srt.managers import schedule_batch as schedule_batch_mod
    from sglang.srt.mem_cache import allocation as allocation_mod

    install_moss_vl_realtime_paged_decode_alloc_patch()
    install_moss_vl_realtime_paged_decode_alloc_patch()
    patched = schedule_batch_mod.alloc_for_decode
    assert patched._sglang_omni_patch_owner == "sglang_omni.moss_vl_realtime"

    captured = {}

    def fake_alloc(**kwargs):
        captured.update(kwargs)
        return torch.tensor([777], dtype=torch.int64)

    monkeypatch.setattr(allocation_mod, "alloc_paged_token_slots_decode", fake_alloc)
    monkeypatch.setattr(allocation_mod, "_alloc_page_size", lambda batch: 16)
    monkeypatch.setattr(
        allocation_mod, "_compute_dsv4_state_lens", lambda *a, **k: None
    )

    # Row layout: 5 encoder slots followed by 3 decoder slots.
    req_to_token = torch.zeros((1, 32), dtype=torch.int32)
    req_to_token[0, :8] = torch.tensor([11, 12, 13, 14, 15, 21, 22, 23])
    writes = []
    pool = SimpleNamespace(
        req_to_token=req_to_token,
        write=lambda indices, values: writes.append((indices, values)),
    )
    req = SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=8))
    batch = SimpleNamespace(
        model_config=SimpleNamespace(is_encoder_decoder=True),
        maybe_evict_swa=lambda: None,
        seq_lens=torch.tensor([3], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
        encoder_lens=torch.tensor([5], dtype=torch.int64),
        encoder_lens_cpu=[5],
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        req_to_token_pool=pool,
        tree_cache=None,
        reqs=[req],
    )

    out = patched(batch, 1)

    assert out.tolist() == [777]
    # last_loc is read at the total row tail (5 encoder + 3 decoder - 1 = 7),
    # not at the decoder-only position (2) which holds an encoder slot.
    assert captured["last_loc"].tolist() == [23]
    # Page-boundary accounting is keyed on the total row length.
    assert captured["seq_lens"].tolist() == [9]
    assert captured["seq_lens_cpu"].tolist() == [9]
    # The new decode slot is written at the encoder-offset row position.
    ((indices, values),) = writes
    assert indices[1].tolist() == [8]
    assert values.tolist() == [777]
    assert req.kv.kv_allocated_len == 9


def test_paged_decode_alloc_patch_passes_through_for_decoder_only(
    monkeypatch,
) -> None:
    from sglang.srt.managers import schedule_batch as schedule_batch_mod

    sentinel_calls = []

    def sentinel(batch, token_per_req):
        sentinel_calls.append((batch, token_per_req))
        return "original"

    # Wrap the sentinel so the delegation itself is observable.
    monkeypatch.setattr(schedule_batch_mod, "alloc_for_decode", sentinel)
    install_moss_vl_realtime_paged_decode_alloc_patch()
    wrapped = schedule_batch_mod.alloc_for_decode

    batch = SimpleNamespace(model_config=SimpleNamespace(is_encoder_decoder=False))
    assert wrapped(batch, 1) == "original"
    assert sentinel_calls == [(batch, 1)]
