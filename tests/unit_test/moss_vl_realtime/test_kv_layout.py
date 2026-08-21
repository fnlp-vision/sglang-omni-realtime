from __future__ import annotations

import pytest
import torch

from sglang_omni.models.moss_vl_realtime import (
    MossVLRealtimeKVLayout,
    count_shared_tail_page_slots,
    insert_encoder_slots,
    read_req_to_token_layout,
    write_req_to_token_layout,
)


def test_incremental_vision_slots_are_inserted_before_text_mapping() -> None:
    req_to_token = torch.zeros((2, 10), dtype=torch.int64)
    req_to_token[1, :4] = torch.tensor([11, 12, 21, 22])

    layout = insert_encoder_slots(
        req_to_token,
        req_pool_index=1,
        encoder_length=2,
        decoder_length=2,
        new_slots=torch.tensor([13, 14], dtype=torch.int64),
    )

    assert layout.encoder_slots == (11, 12, 13, 14)
    assert layout.decoder_slots == (21, 22)
    assert req_to_token[1, :6].tolist() == [11, 12, 13, 14, 21, 22]
    # Decoder KV remains in the original physical slots. Only the row mapping moved.
    assert set(layout.decoder_slots) == {21, 22}


def test_encoder_and_decoder_can_grow_without_slot_overlap() -> None:
    layout = MossVLRealtimeKVLayout(
        encoder_slots=(10, 11),
        decoder_slots=(20,),
    )
    layout = layout.append_decoder([21, 22]).append_encoder([12])

    assert layout.as_tuple() == (10, 11, 12, 20, 21, 22)
    assert layout.encoder_length == 3
    assert layout.decoder_length == 3


def test_layout_round_trip_uses_committed_lengths_not_nonzero_tail() -> None:
    req_to_token = torch.full((1, 8), 99, dtype=torch.int32)
    layout = MossVLRealtimeKVLayout((1, 2), (7, 8, 9))
    write_req_to_token_layout(
        req_to_token,
        req_pool_index=0,
        layout=layout,
        clear_tail=True,
    )

    restored = read_req_to_token_layout(
        req_to_token,
        req_pool_index=0,
        encoder_length=2,
        decoder_length=3,
    )
    assert restored == layout
    assert req_to_token[0, 5:].tolist() == [0, 0, 0]


def test_layout_rejects_duplicate_or_overlapping_slots() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MossVLRealtimeKVLayout((1, 1), ())
    with pytest.raises(ValueError, match="must not overlap"):
        MossVLRealtimeKVLayout((1, 2), (2, 3))


def test_layout_rejects_row_capacity_overflow_without_partial_write() -> None:
    req_to_token = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    before = req_to_token.clone()

    with pytest.raises(ValueError, match="does not fit"):
        insert_encoder_slots(
            req_to_token,
            req_pool_index=0,
            encoder_length=2,
            decoder_length=2,
            new_slots=[5],
        )

    assert torch.equal(req_to_token, before)


def test_shared_tail_page_slots_follow_paged_alloc_extend_layout() -> None:
    # alloc_extend fills the old partial page first, so exactly
    # (-committed) % page_size leading append slots share the committed tail
    # page and must be retained on rollback.
    assert (
        count_shared_tail_page_slots(
            committed_length=20, append_length=20, page_size=16
        )
        == 12
    )
    # Page-aligned or empty committed prefix: no shared page.
    assert (
        count_shared_tail_page_slots(committed_length=32, append_length=5, page_size=16)
        == 0
    )
    assert (
        count_shared_tail_page_slots(committed_length=0, append_length=5, page_size=16)
        == 0
    )
    # Append smaller than the remaining tail room: fully retained.
    assert (
        count_shared_tail_page_slots(committed_length=4, append_length=5, page_size=16)
        == 5
    )
    # Token-granular allocation keeps the old release-everything behavior.
    assert (
        count_shared_tail_page_slots(committed_length=3, append_length=5, page_size=1)
        == 0
    )


def test_shared_tail_page_slots_validates_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        count_shared_tail_page_slots(committed_length=0, append_length=0, page_size=0)
    with pytest.raises(ValueError, match="non-negative"):
        count_shared_tail_page_slots(committed_length=-1, append_length=0, page_size=16)
    with pytest.raises(TypeError):
        count_shared_tail_page_slots(
            committed_length=0, append_length=0, page_size=True
        )
