"""Convert one frame event into an incremental SGLang multimodal segment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)

from sglang_omni.models.moss_vl_realtime.model_step import (
    compute_realtime_mrope_for_segment,
)
from sglang_omni.models.moss_vl_realtime.payload_types import (
    FramePromptEvent,
    build_realtime_append_text,
)
from sglang_omni.models.moss_vl_realtime.visibility import (
    append_visible_frame_counts,
)

REALTIME_FULL_GRID_THW_KEY = "realtime_full_grid_thw"


@dataclass(slots=True)
class MossVLRealtimeSegment:
    """New encoder/text inputs plus full request-level attention metadata."""

    event: FramePromptEvent
    event_input_ids: torch.LongTensor
    encoder_pad_ids: tuple[int, ...]
    multimodal_inputs: MultimodalInputs
    encoder_delta_length: int
    next_mrope_position: int
    full_grid_thw: torch.LongTensor

    @property
    def raw_append_ids(self) -> tuple[int, ...]:
        return self.encoder_pad_ids + tuple(int(v) for v in self.event_input_ids)


class MossVLRealtimeSegmentBuilder:
    """Build the delta encoded at this step and cumulative mask metadata."""

    def __init__(
        self,
        processor: Any,
        *,
        image_token_id: int,
        merge_size: int,
    ) -> None:
        if merge_size <= 0:
            raise ValueError("merge_size must be positive")
        self.processor = processor
        self.image_token_id = int(image_token_id)
        self.merge_size = int(merge_size)

    def build(
        self,
        event: FramePromptEvent,
        image: Any,
        *,
        previous_mrope_positions: torch.Tensor | None = None,
        previous_visible_counts: torch.Tensor | None = None,
        previous_grid_thw: torch.Tensor | None = None,
        next_mrope_position: int = 0,
        committed_encoder_length: int = 0,
        committed_decoder_length: int | None = None,
        pending_text_tokens: int = 0,
    ) -> MossVLRealtimeSegment:
        if pending_text_tokens < 0:
            raise ValueError("pending_text_tokens must be non-negative")
        append_text = build_realtime_append_text(
            prompt=event.prompt,
            frame_timestamps=(event.timestamp,),
        )
        processed = self.processor(
            text=append_text,
            images=[image],
            add_special_tokens=False,
            return_tensors="pt",
        )
        event_input_ids = torch.as_tensor(processed["input_ids"], dtype=torch.long)
        if event_input_ids.ndim != 2 or event_input_ids.shape[0] != 1:
            raise ValueError("processor input_ids must have shape (1, sequence_length)")
        grid_thw = torch.as_tensor(processed["grid_thw"], dtype=torch.long)
        if grid_thw.ndim == 1:
            grid_thw = grid_thw.unsqueeze(0)
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError("processor grid_thw must have shape (frames, 3)")
        if grid_thw.shape[0] != 1 or int(grid_thw[0, 0].item()) != 1:
            raise ValueError("one realtime event must contain exactly one frame")

        previous_position_length = (
            0
            if previous_mrope_positions is None
            else int(previous_mrope_positions.shape[1])
        )
        if committed_decoder_length is None:
            committed_decoder_length = previous_position_length
        if committed_decoder_length < previous_position_length:
            raise ValueError(
                "committed_decoder_length cannot be shorter than MRoPE history"
            )
        missing_committed_tokens = committed_decoder_length - previous_position_length
        if next_mrope_position < missing_committed_tokens:
            raise ValueError("next_mrope_position is inconsistent with decoder length")

        event_start_position = next_mrope_position + pending_text_tokens
        event_positions, vision_positions, next_position = (
            compute_realtime_mrope_for_segment(
                new_input_ids=event_input_ids,
                new_grid_thw=grid_thw,
                start_position=event_start_position,
                image_token_id=self.image_token_id,
                merge_size=self.merge_size,
            )
        )
        generated_positions = torch.arange(
            next_mrope_position - missing_committed_tokens,
            event_start_position,
            dtype=torch.long,
        ).reshape(1, 1, missing_committed_tokens + pending_text_tokens)
        generated_positions = generated_positions.expand(3, 1, -1)
        new_positions = torch.cat([generated_positions, event_positions.cpu()], dim=-1)
        full_positions = _append_mrope_positions(
            previous_mrope_positions,
            new_positions.squeeze(1),
        )

        generated_ids = torch.full(
            (missing_committed_tokens + pending_text_tokens,),
            fill_value=-1,
            dtype=torch.long,
        )
        visibility_ids = torch.cat([generated_ids, event_input_ids.flatten().cpu()])
        full_visible_counts = append_visible_frame_counts(
            previous_visible_counts,
            visibility_ids,
            image_token_id=self.image_token_id,
        )
        full_grid_thw = _append_grid_thw(previous_grid_thw, grid_thw)

        encoder_delta_length = _encoder_length(grid_thw, self.merge_size)
        total_encoder_length = committed_encoder_length + encoder_delta_length
        pixel_values = torch.as_tensor(processed["pixel_values"])
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=_event_hash(event),
            feature=pixel_values,
            model_specific_data={
                "grid_thw": grid_thw,
                REALTIME_FULL_GRID_THW_KEY: full_grid_thw,
            },
        )
        item.set_pad_value()
        mm_inputs = MultimodalInputs(
            mm_items=[item],
            num_image_tokens=total_encoder_length,
            mrope_positions=full_positions,
            mrope_position_delta=torch.tensor(
                [next_position - full_positions.shape[1]], dtype=torch.long
            ),
            vision_position_ids=(
                vision_positions.squeeze(1).cpu()
                if vision_positions is not None
                else None
            ),
            media_nums_per_sample=[full_grid_thw.shape[0]],
            visible_frame_counts=full_visible_counts,
        )
        return MossVLRealtimeSegment(
            event=event,
            event_input_ids=event_input_ids.flatten().cpu(),
            encoder_pad_ids=(item.pad_value,) * encoder_delta_length,
            multimodal_inputs=mm_inputs,
            encoder_delta_length=encoder_delta_length,
            next_mrope_position=next_position,
            full_grid_thw=full_grid_thw,
        )

    def build_prompt(
        self,
        event: FramePromptEvent,
        *,
        previous_mrope_positions: torch.Tensor | None = None,
        previous_visible_counts: torch.Tensor | None = None,
        previous_grid_thw: torch.Tensor | None = None,
        next_mrope_position: int = 0,
        committed_encoder_length: int = 0,
        committed_decoder_length: int | None = None,
        pending_text_tokens: int = 0,
    ) -> MossVLRealtimeSegment:
        """Build a pure-text user turn without inventing a media input."""
        if event.has_frame or event.prompt is None:
            raise ValueError("prompt segment requires a text-only event")
        if pending_text_tokens < 0:
            raise ValueError("pending_text_tokens must be non-negative")
        append_text = build_realtime_append_text(prompt=event.prompt)
        processed = self.processor.tokenizer(
            append_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        event_input_ids = torch.as_tensor(processed["input_ids"], dtype=torch.long)
        if event_input_ids.ndim != 2 or event_input_ids.shape[0] != 1:
            raise ValueError("tokenizer input_ids must have shape (1, sequence_length)")

        previous_position_length = (
            0
            if previous_mrope_positions is None
            else int(previous_mrope_positions.shape[1])
        )
        if committed_decoder_length is None:
            committed_decoder_length = previous_position_length
        if committed_decoder_length < previous_position_length:
            raise ValueError(
                "committed_decoder_length cannot be shorter than MRoPE history"
            )
        missing_committed_tokens = committed_decoder_length - previous_position_length
        if next_mrope_position < missing_committed_tokens:
            raise ValueError("next_mrope_position is inconsistent with decoder length")

        event_start_position = next_mrope_position + pending_text_tokens
        event_positions, _, next_position = compute_realtime_mrope_for_segment(
            new_input_ids=event_input_ids,
            new_grid_thw=None,
            start_position=event_start_position,
            image_token_id=self.image_token_id,
            merge_size=self.merge_size,
        )
        generated_positions = torch.arange(
            next_mrope_position - missing_committed_tokens,
            event_start_position,
            dtype=torch.long,
        ).reshape(1, 1, missing_committed_tokens + pending_text_tokens)
        generated_positions = generated_positions.expand(3, 1, -1)
        new_positions = torch.cat([generated_positions, event_positions.cpu()], dim=-1)
        full_positions = _append_mrope_positions(
            previous_mrope_positions,
            new_positions.squeeze(1),
        )

        generated_ids = torch.full(
            (missing_committed_tokens + pending_text_tokens,),
            fill_value=-1,
            dtype=torch.long,
        )
        visibility_ids = torch.cat([generated_ids, event_input_ids.flatten().cpu()])
        full_visible_counts = append_visible_frame_counts(
            previous_visible_counts,
            visibility_ids,
            image_token_id=self.image_token_id,
        )
        full_grid_thw = (
            torch.empty((0, 3), dtype=torch.long)
            if previous_grid_thw is None
            else torch.as_tensor(previous_grid_thw, dtype=torch.long).cpu()
        )
        mm_inputs = MultimodalInputs(
            mm_items=[],
            num_image_tokens=committed_encoder_length,
            mrope_positions=full_positions,
            mrope_position_delta=torch.tensor(
                [next_position - full_positions.shape[1]], dtype=torch.long
            ),
            vision_position_ids=None,
            media_nums_per_sample=[full_grid_thw.shape[0]],
            visible_frame_counts=full_visible_counts,
        )
        return MossVLRealtimeSegment(
            event=event,
            event_input_ids=event_input_ids.flatten().cpu(),
            encoder_pad_ids=(),
            multimodal_inputs=mm_inputs,
            encoder_delta_length=0,
            next_mrope_position=next_position,
            full_grid_thw=full_grid_thw,
        )


def _append_mrope_positions(
    previous: torch.Tensor | None,
    new: torch.Tensor,
) -> torch.Tensor:
    new = torch.as_tensor(new, dtype=torch.long).cpu()
    if new.ndim != 2 or new.shape[0] != 3:
        raise ValueError("MRoPE positions must have shape (3, sequence_length)")
    if previous is None:
        return new
    previous = torch.as_tensor(previous, dtype=torch.long).cpu()
    if previous.ndim != 2 or previous.shape[0] != 3:
        raise ValueError(
            "previous MRoPE positions must have shape (3, sequence_length)"
        )
    return torch.cat([previous, new], dim=1)


def _append_grid_thw(
    previous: torch.Tensor | None,
    new: torch.Tensor,
) -> torch.LongTensor:
    new = torch.as_tensor(new, dtype=torch.long).cpu()
    if previous is None:
        return new
    previous = torch.as_tensor(previous, dtype=torch.long).cpu()
    if previous.ndim != 2 or previous.shape[1] != 3:
        raise ValueError("previous_grid_thw must have shape (frames, 3)")
    return torch.cat([previous, new], dim=0)


def _encoder_length(grid_thw: torch.Tensor, merge_size: int) -> int:
    merge_square = merge_size**2
    if bool(((grid_thw[:, 1:] % merge_size) != 0).any()):
        raise ValueError("frame grid must be divisible by merge_size")
    vision_tokens = torch.prod(grid_thw, dim=1) // merge_square
    separators = grid_thw[:, 0]
    return int((vision_tokens + separators).sum().item())


def _event_hash(event: FramePromptEvent) -> int:
    if event.frame_ref is None:
        raise ValueError("text-only events do not have a media hash")
    source = event.fingerprint or f"{event.frame_ref}\0{event.timestamp:.9f}"
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:16], "big")
