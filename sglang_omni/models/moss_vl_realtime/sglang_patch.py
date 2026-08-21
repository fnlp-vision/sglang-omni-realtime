"""Narrow SGLang scheduling hook for MOSS-VL realtime extend batches."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    is_moss_vl_realtime_batch,
    prepare_moss_vl_realtime_encoder_info_extend,
    rollback_moss_vl_realtime_batch,
)

_PATCH_OWNER = "sglang_omni.moss_vl_realtime"


def install_moss_vl_realtime_paged_decode_alloc_patch() -> None:
    """Fix upstream paged decode allocation for encoder-decoder layouts.

    Upstream ``alloc_for_decode`` reads ``last_loc`` at the decoder-only row
    position (``seq_lens - 1``) and keys page-boundary decisions on
    decoder-only lengths, while the write path uses
    ``encoder_lens + seq_lens``. For our [encoder prefix | decoder text] row
    layout that mismatch reads an encoder slot as ``last_loc``; with
    page_size > 1 it trips the allocator's debug continuity assert and would
    otherwise silently allocate decode KV into encoder slots. This wrapper
    keys both the ``last_loc`` read and the page accounting on the total row
    length (encoder + decoder). page_size == 1 is unaffected upstream and
    keeps the original path.
    """
    import torch
    from sglang.srt.managers import schedule_batch as schedule_batch_mod
    from sglang.srt.mem_cache import allocation as allocation_mod

    original = schedule_batch_mod.alloc_for_decode
    if getattr(original, "_sglang_omni_patch_owner", None) == _PATCH_OWNER:
        return

    def alloc_for_decode(batch: Any, token_per_req: int) -> Any:
        if (
            not batch.model_config.is_encoder_decoder
            or allocation_mod._alloc_page_size(batch) == 1
        ):
            return original(batch, token_per_req)

        batch.maybe_evict_swa()
        seq_lens_gpu = batch.seq_lens
        encoder_lens = batch.encoder_lens
        last_loc = batch.req_to_token_pool.req_to_token[
            batch.req_pool_indices, seq_lens_gpu - 1 + encoder_lens
        ]
        out_cache_loc = allocation_mod.alloc_paged_token_slots_decode(
            tree_cache=batch.tree_cache,
            seq_lens=seq_lens_gpu + encoder_lens + token_per_req,
            seq_lens_cpu=batch.seq_lens_cpu
            + torch.tensor(batch.encoder_lens_cpu, dtype=torch.int64)
            + token_per_req,
            last_loc=last_loc,
            token_per_req=token_per_req,
            req_pool_indices=batch.req_pool_indices,
            dsv4_state_lens=allocation_mod._compute_dsv4_state_lens(
                batch, is_decode=True
            ),
            batch=batch,
        )
        batch.req_to_token_pool.write(
            (batch.req_pool_indices, encoder_lens + seq_lens_gpu),
            out_cache_loc.to(torch.int32),
        )
        if allocation_mod._is_npu:
            allocation_mod.maybe_write_dsv4_decode(
                batch, batch.seq_lens_cpu + token_per_req, token_per_req
            )
        for req in batch.reqs:
            req.kv.kv_allocated_len += token_per_req
        return out_cache_loc

    alloc_for_decode._sglang_omni_patch_owner = _PATCH_OWNER
    alloc_for_decode._sglang_omni_original = original
    schedule_batch_mod.alloc_for_decode = alloc_for_decode


def install_moss_vl_realtime_schedule_batch_patch() -> None:
    """Dispatch only marked requests to incremental encoder preparation."""
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    current_encoder_prepare = ScheduleBatch.prepare_encoder_info_extend
    if (
        getattr(current_encoder_prepare, "_sglang_omni_patch_owner", None)
        != _PATCH_OWNER
    ):

        def prepare_encoder_info_extend(
            self: Any,
            input_ids: list[Any],
            seq_lens: list[int],
        ) -> None:
            if is_moss_vl_realtime_batch(self):
                prepare_moss_vl_realtime_encoder_info_extend(
                    self,
                    input_ids,
                    seq_lens,
                )
                return
            current_encoder_prepare(self, input_ids, seq_lens)

        prepare_encoder_info_extend._sglang_omni_patch_owner = _PATCH_OWNER
        prepare_encoder_info_extend._sglang_omni_original = current_encoder_prepare
        ScheduleBatch.prepare_encoder_info_extend = prepare_encoder_info_extend

    current_prepare = ScheduleBatch.prepare_for_extend
    if getattr(current_prepare, "_sglang_omni_patch_owner", None) == _PATCH_OWNER:
        return

    def prepare_for_extend(self: Any) -> None:
        if not is_moss_vl_realtime_batch(self):
            current_prepare(self)
            return
        try:
            current_prepare(self)
        except Exception:
            rollback_moss_vl_realtime_batch(self)
            raise

    prepare_for_extend._sglang_omni_patch_owner = _PATCH_OWNER
    prepare_for_extend._sglang_omni_original = current_prepare
    ScheduleBatch.prepare_for_extend = prepare_for_extend
