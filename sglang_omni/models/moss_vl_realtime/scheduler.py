"""Persistent-request scheduler for MOSS-VL realtime frame updates."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from PIL import Image
from sglang.srt.managers.schedule_batch import NextBatchPlan, ScheduleBatch
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.observability.metrics_collector import QueueCount

from sglang_omni.models.moss_vl_realtime.batch_adapter import (
    RUNTIME_STATE_ATTR,
    MossVLRealtimeScheduleBatch,
)
from sglang_omni.models.moss_vl_realtime.frame_store import resolve_shared_memory_frame
from sglang_omni.models.moss_vl_realtime.payload_types import FramePromptEvent
from sglang_omni.models.moss_vl_realtime.runtime_state import (
    MossVLRealtimePhase,
    MossVLRealtimeRuntimeState,
)
from sglang_omni.models.moss_vl_realtime.segment import (
    MossVLRealtimeSegment,
    MossVLRealtimeSegmentBuilder,
)
from sglang_omni.models.moss_vl_realtime.session_state import (
    MossVLRealtimeSessionController,
)
from sglang_omni.scheduling.omni_scheduler import _FAILED_BATCH_RESULT, OmniScheduler

logger = logging.getLogger(__name__)


class MossVLRealtimeScheduler(OmniScheduler):
    """Interrupt decode between steps to extend one or more frame events."""

    def __init__(
        self,
        *args: Any,
        segment_builder: MossVLRealtimeSegmentBuilder,
        frame_resolver: Callable[[FramePromptEvent], Any] | None = None,
        silence_token_ids: tuple[int, ...],
        parked_request_timeout_s: float = 300.0,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("enable_overlap", False):
            raise ValueError("MOSS-VL realtime requires overlap scheduling disabled")
        server_args = kwargs.get("server_args")
        if server_args is not None and int(server_args.page_size) != 1:
            raise ValueError("MOSS-VL realtime requires page_size == 1")
        self.segment_builder = segment_builder
        self.frame_resolver = frame_resolver or resolve_local_frame
        self.realtime_sessions = MossVLRealtimeSessionController()
        self._realtime_extend_batch: ScheduleBatch | None = None
        self.silence_token_ids = tuple(int(token_id) for token_id in silence_token_ids)
        if not self.silence_token_ids:
            raise ValueError("silence_token_ids must not be empty")
        self.parked_request_timeout_s = float(parked_request_timeout_s)
        if self.parked_request_timeout_s <= 0:
            raise ValueError("parked_request_timeout_s must be positive")
        self.parked_reqs: dict[str, Any] = {}
        self.parked_since: dict[str, float] = {}
        kwargs["request_update_handler"] = self._ingest_request_update
        super().__init__(*args, **kwargs)

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        state = _runtime_state(req_data)
        self.realtime_sessions.open(state.request_id, state.session_id)
        try:
            super()._enqueue_built_request(
                payload,
                pending_stream_done,
                req_data,
                request_admission_lock_held=request_admission_lock_held,
            )
        except Exception:
            self.realtime_sessions.close(state.request_id)
            raise

    def _ingest_request_update(self, req_data: Any, data: Any) -> None:
        state = _runtime_state(req_data)
        try:
            self.realtime_sessions.ingest(state.request_id, data)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            # A malformed/out-of-order client event must not crash the stage:
            # reject the update, surface the error, and abort only the
            # offending request.
            logger.exception("Rejected realtime update for %s", state.request_id)
            self._emit_request_error(state.request_id, exc)
            self.abort(state.request_id, defer_running_cleanup=False)

    def get_next_batch_to_run(self) -> Any | None:
        self._expire_parked_requests()
        if self._realtime_extend_batch is None:
            if self._async_pending is not None and self._has_pending_realtime_events():
                # Update barrier: resolve the in-flight lookahead step before
                # materializing events, so _build_segment's "exactly one
                # committed pending token" invariant holds at materialize time.
                self._resolve_pending_async()
            self._realtime_extend_batch = self._materialize_realtime_extensions()
        if self._realtime_extend_batch is None and self._decode_rate_limited():
            return None
        return super().get_next_batch_to_run()

    def _decode_rate_limited(self) -> bool:
        """Defer ordinary decode without delaying frame or prompt ingestion.

        Realtime deployment currently admits one live request. Keeping the
        request in ``running_batch`` lets the scheduler continue draining
        control messages while preserving its KV ownership.
        """
        reqs = tuple(getattr(self.running_batch, "reqs", ()))
        if not reqs:
            return False
        if len(reqs) != 1:
            raise AssertionError("MOSS-VL realtime admitted multiple live requests")
        state = getattr(reqs[0], RUNTIME_STATE_ATTR, None)
        return (
            isinstance(state, MossVLRealtimeRuntimeState)
            and time.monotonic() < state.next_decode_not_before
        )

    def _stamp_batch_launch(self, batch: Any) -> None:
        """Start the token-rate interval at forward launch, matching HF."""
        super()._stamp_batch_launch(batch)
        launch_time = float(batch.launch_ts)
        for req in batch.reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if isinstance(state, MossVLRealtimeRuntimeState):
                state.next_decode_not_before = (
                    launch_time + 1.0 / state.max_tokens_per_turn
                )

    def _has_pending_realtime_events(self) -> bool:
        running_reqs = tuple(getattr(self.running_batch, "reqs", ()))
        for req in running_reqs + tuple(self.parked_reqs.values()):
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState) or req.finished():
                continue
            session = self.realtime_sessions.get(req.rid)
            if session is not None and session.pending_events:
                return True
        return False

    def get_new_batch_prefill(self, running_batch: Any) -> NextBatchPlan:
        realtime_batch = self._realtime_extend_batch
        if realtime_batch is not None:
            self._realtime_extend_batch = None
            return NextBatchPlan(
                batch_to_run=realtime_batch,
                running_batch=running_batch,
            )
        return super().get_new_batch_prefill(running_batch)

    def _materialize_realtime_extensions(self) -> ScheduleBatch | None:
        running_batch = self.running_batch
        running_reqs = tuple(getattr(running_batch, "reqs", ()))
        candidate_reqs = running_reqs + tuple(self.parked_reqs.values())
        if not candidate_reqs:
            return None

        selected: list[tuple[Any, FramePromptEvent, MossVLRealtimeSegment]] = []
        for req in candidate_reqs:
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            if not isinstance(state, MossVLRealtimeRuntimeState):
                continue
            session = self.realtime_sessions.get(req.rid)
            if (
                session is None
                or not session.pending_events
                or state.phase
                not in (
                    MossVLRealtimePhase.DECODING,
                    MossVLRealtimePhase.WAITING_FOR_EVENT,
                )
                or req.finished()
            ):
                continue
            event = session.drain(max_events=1)[0]
            try:
                segment = self._build_segment(req, state, event)
                bind_realtime_page_row(
                    req,
                    self.req_to_token_pool.req_to_token,
                )
                _append_segment_to_request(req, state, segment)
            except Exception as exc:
                logger.exception("Failed to materialize realtime event for %s", req.rid)
                self._emit_request_error(req.rid, exc)
                self.abort(req.rid, defer_running_cleanup=False)
                continue
            selected.append((req, event, segment))

        if not selected:
            return None

        reqs = [req for req, _, _ in selected]
        extend_batch = MossVLRealtimeScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        try:
            extend_batch.prepare_for_extend()
        # This boundary must contain allocator, processor, and model-specific errors.
        except Exception as exc:  # noqa: BLE001
            for req, _, segment in selected:
                _undo_appended_segment(req, segment)
                self._emit_request_error(req.rid, exc)
                self.abort(req.rid, defer_running_cleanup=False)
            return None
        extend_batch.prefill_stats = PrefillStats(
            log_input_tokens=sum(extend_batch.extend_lens),
            log_hit_tokens=sum(extend_batch.prefix_lens),
            new_token_ratio=float(getattr(self, "new_token_ratio", 0.0)),
            num_running_reqs=QueueCount.from_reqs(
                list(running_reqs),
                self.enable_priority_scheduling,
            ),
            num_new_seqs=len(reqs),
        )

        selected_ids = {req.rid for req in reqs}
        for request_id in selected_ids:
            if self.parked_reqs.pop(request_id, None) is not None:
                logger.info("Realtime request %s woke for a frame event", request_id)
            self.parked_since.pop(request_id, None)
        keep_indices = [
            index
            for index, req in enumerate(running_reqs)
            if req.rid not in selected_ids
        ]
        if running_batch is not None:
            running_batch.filter_batch(keep_indices=keep_indices)
            running_batch.batch_is_full = False
        return extend_batch

    def _resolve_and_process(
        self, batch: Any, sched_output: Any, pending_step: Any
    ) -> None:
        """Extend the lookahead overrun drop with silence-parked requests.

        A request parked after step N is still present in step N+1's
        already-launched batch: its overrun token must not be emitted,
        appended, or counted, and its decode KV slot must be freed. The runner
        side skips the state advance via the WAITING_FOR_EVENT phase check in
        ``post_decode_resolve``. Mirrors the upstream finished/retracted
        pre-drop, adding ``rid in parked_reqs`` to the predicate.
        """
        pre_drop = [
            r.finished() or r.is_retracted or r.rid in self.parked_reqs
            for r in batch.reqs
        ]
        if not any(pre_drop):
            return super()._resolve_and_process(batch, sched_output, pending_step)
        skip_rids = {batch.reqs[i].rid for i, was in enumerate(pre_drop) if was}
        result = self._run_batch_resolve(
            batch, sched_output, pending_step, skip_rids=skip_rids
        )
        if result is _FAILED_BATCH_RESULT:
            return
        keep = [i for i, was in enumerate(pre_drop) if not was]
        # Free only the parked rows' overrun slots: finished/retracted rows
        # follow the upstream cleanup contract (released with the request).
        parked_drop_indices = [
            i
            for i, req in enumerate(batch.reqs)
            if pre_drop[i] and not req.finished() and not req.is_retracted
        ]
        self._free_overrun_step_slots(batch.out_cache_loc, parked_drop_indices)
        if result.next_token_ids is not None and keep:
            idx = torch.tensor(keep, device=result.next_token_ids.device)
            result.next_token_ids = result.next_token_ids[idx]
        batch.reqs = [batch.reqs[i] for i in keep]
        if batch.reqs:
            self.process_batch_result(batch, result)

    def process_batch_result(self, batch: Any, result: Any) -> None:
        for req in batch.reqs:
            if getattr(req, "_moss_vl_realtime_final_extend", False):
                if not getattr(req, "_moss_vl_realtime_keep_ignore_eos", False):
                    req.sampling_params.ignore_eos = False
                del req._moss_vl_realtime_final_extend
        super().process_batch_result(batch, result)
        for req in tuple(batch.reqs):
            if req.finished():
                self.realtime_sessions.close(req.rid)
                self.parked_reqs.pop(req.rid, None)
                self.parked_since.pop(req.rid, None)
        self._park_silent_requests(batch)

    def _park_silent_requests(self, batch: Any) -> None:
        keep_indices: list[int] = []
        for index, req in enumerate(tuple(batch.reqs)):
            state = getattr(req, RUNTIME_STATE_ATTR, None)
            session = self.realtime_sessions.get(req.rid)
            is_silence = bool(req.output_ids) and _ends_with_token_ids(
                req.output_ids,
                self.silence_token_ids,
            )
            if is_silence or req.finished():
                reason = getattr(req, "finished_reason", None)
                logger.info(
                    "Realtime request %s result token=%s silence=%s "
                    "ignore_eos=%s finished=%s",
                    req.rid,
                    int(req.output_ids[-1]) if req.output_ids else None,
                    is_silence,
                    getattr(getattr(req, "sampling_params", None), "ignore_eos", None),
                    reason.to_json() if reason is not None else None,
                )
            should_park = (
                isinstance(state, MossVLRealtimeRuntimeState)
                and session is not None
                and not session.pending_events
                and not session.final_received
                and not req.finished()
                and is_silence
            )
            if not should_park:
                keep_indices.append(index)
                continue
            state.mark_waiting()
            self.parked_reqs[req.rid] = req
            self.parked_since[req.rid] = time.monotonic()
            logger.info("Realtime request %s parked after silence", req.rid)
        if len(keep_indices) != len(batch.reqs):
            batch.filter_batch(keep_indices=keep_indices)
            batch.batch_is_full = False

    def _expire_parked_requests(self) -> None:
        now = time.monotonic()
        expired = [
            request_id
            for request_id, parked_at in self.parked_since.items()
            if now - parked_at >= self.parked_request_timeout_s
        ]
        for request_id in expired:
            self._emit_request_error(
                request_id,
                TimeoutError("realtime request exceeded parked idle timeout"),
            )
            self.abort(request_id, defer_running_cleanup=False)

    def _find_request_data(self, request_id: str) -> Any | None:
        data = super()._find_request_data(request_id)
        if data is not None:
            return data
        req = self.parked_reqs.get(request_id)
        return None if req is None else req._omni_data

    def _build_segment(
        self,
        req: Any,
        state: MossVLRealtimeRuntimeState,
        event: FramePromptEvent,
    ) -> MossVLRealtimeSegment:
        committed_total = state.encoder_length + state.decoder_length
        req._refresh_fill_ids()
        pending_count = len(req.full_untruncated_fill_ids) - committed_total
        if pending_count != 1:
            raise RuntimeError(
                "realtime event must coalesce with exactly one pending sampled token"
            )
        common = {
            "previous_mrope_positions": state.mrope_positions,
            "previous_visible_counts": state.visible_frame_counts,
            "previous_grid_thw": state.full_grid_thw,
            "next_mrope_position": state.next_mrope_position,
            "committed_encoder_length": state.encoder_length,
            "committed_decoder_length": state.decoder_length,
            "pending_text_tokens": 1,
        }
        if not event.has_frame:
            return self.segment_builder.build_prompt(event, **common)
        image = self.frame_resolver(event)
        return self.segment_builder.build(event, image, **common)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        pending = getattr(self, "_async_pending", None)
        if pending is not None and any(r.rid == request_id for r in pending[0].reqs):
            # Flush the in-flight lookahead step before touching this
            # request's state/KV; its row resolves (and is dropped when
            # already parked) through the normal path first.
            self._resolve_pending_async()
        parked_req = self.parked_reqs.get(request_id)
        data = self._find_request_data(request_id)
        if data is not None:
            state = getattr(data, "runtime_state", None)
            if (
                isinstance(state, MossVLRealtimeRuntimeState)
                and not state._append_inflight
            ):
                state.finish(aborted=True)
        self.realtime_sessions.close(request_id)
        super().abort(
            request_id,
            defer_running_cleanup=defer_running_cleanup,
        )
        self.parked_since.pop(request_id, None)
        parked_req = self.parked_reqs.pop(request_id, parked_req)
        if parked_req is not None:
            self._release_request_kv_cache(parked_req)
            parked_req._omni_data = None

    def stop(self) -> None:
        for request_id in tuple(self.parked_reqs):
            self.abort(request_id, defer_running_cleanup=False)
        super().stop()


def _append_segment_to_request(
    req: Any,
    state: MossVLRealtimeRuntimeState,
    segment: MossVLRealtimeSegment,
) -> None:
    committed_total = state.encoder_length + state.decoder_length
    req._refresh_fill_ids()
    if len(req.full_untruncated_fill_ids) != committed_total + 1:
        raise RuntimeError("request does not have exactly one pending sampled token")
    state.pending_token_id = int(req.full_untruncated_fill_ids[-1])
    page_prefix = req._moss_vl_realtime_page_row[:committed_total]
    invalid = torch.nonzero(page_prefix <= 0, as_tuple=False).flatten()
    if invalid.numel():
        first_invalid = invalid[:8].tolist()
        kv = getattr(req, "kv", None)
        raise RuntimeError(
            "realtime committed KV prefix contains empty slots "
            f"(indices={first_invalid}, encoder_length={state.encoder_length}, "
            f"decoder_length={state.decoder_length}, "
            f"req_kv_committed_len={getattr(req, 'kv_committed_len', None)}, "
            f"req_kv_allocated_len={getattr(kv, 'kv_allocated_len', None)})"
        )
    req._moss_vl_realtime_previous_mm_inputs = req.multimodal_inputs
    req._moss_vl_realtime_previous_extend_range = req.extend_range
    req._moss_vl_realtime_previous_prefix_indices = req.prefix_indices
    req._moss_vl_realtime_previous_skip_radix_cache_insert = req.skip_radix_cache_insert
    req.skip_radix_cache_insert = True
    req._moss_vl_realtime_staged_mrope_positions = (
        segment.multimodal_inputs.mrope_positions.clone()
    )
    req._moss_vl_realtime_staged_visible_frame_counts = (
        segment.multimodal_inputs.visible_frame_counts.clone()
    )
    req._moss_vl_realtime_staged_full_grid_thw = segment.full_grid_thw.clone()
    req._moss_vl_realtime_staged_event = segment.event.to_dict()
    if segment.event.prompt is not None:
        req._moss_vl_realtime_staged_turn_transition = {
            "interrupted_turn_id": state.turn_id,
            "turn_id": state.turn_id + 1,
        }
    if segment.event.final:
        req._moss_vl_realtime_final_extend = True
    req.output_ids.extend(segment.raw_append_ids)
    req.sampling_params.max_new_tokens += len(segment.raw_append_ids)
    req.multimodal_inputs = segment.multimodal_inputs
    req._refresh_fill_ids()
    req.prefix_indices = req._moss_vl_realtime_page_row[:committed_total].to(
        dtype=torch.int64
    )
    req.set_extend_range(committed_total, len(req.full_untruncated_fill_ids))


def _undo_appended_segment(req: Any, segment: MossVLRealtimeSegment) -> None:
    count = len(segment.raw_append_ids)
    if count:
        del req.output_ids[-count:]
        req.sampling_params.max_new_tokens -= count
    req.multimodal_inputs = req._moss_vl_realtime_previous_mm_inputs
    req.extend_range = req._moss_vl_realtime_previous_extend_range
    req.prefix_indices = req._moss_vl_realtime_previous_prefix_indices
    req.skip_radix_cache_insert = req._moss_vl_realtime_previous_skip_radix_cache_insert
    del req._moss_vl_realtime_previous_mm_inputs
    del req._moss_vl_realtime_previous_extend_range
    del req._moss_vl_realtime_previous_prefix_indices
    del req._moss_vl_realtime_previous_skip_radix_cache_insert
    del req._moss_vl_realtime_staged_mrope_positions
    del req._moss_vl_realtime_staged_visible_frame_counts
    del req._moss_vl_realtime_staged_full_grid_thw
    del req._moss_vl_realtime_staged_event
    if hasattr(req, "_moss_vl_realtime_staged_turn_transition"):
        del req._moss_vl_realtime_staged_turn_transition
    if hasattr(req, "_moss_vl_realtime_final_extend"):
        del req._moss_vl_realtime_final_extend
    req._refresh_fill_ids()
    state = getattr(req, RUNTIME_STATE_ATTR)
    state.pending_token_id = None


def bind_realtime_page_row(req: Any, req_to_token: torch.Tensor) -> None:
    """Bind the live page-table row used when converting decode to extend."""
    req_pool_index = int(req.req_pool_idx)
    req._moss_vl_realtime_page_row = req_to_token[req_pool_index]


def resolve_local_frame(event: FramePromptEvent) -> Image.Image:
    """Resolve local/file refs and one-consumer shared-memory frames."""
    if event.frame_ref is None:
        raise ValueError("text-only event has no frame to resolve")
    parsed = urlparse(event.frame_ref)
    if parsed.scheme == "shm":
        return resolve_shared_memory_frame(event.frame_ref)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif parsed.scheme == "":
        path = Path(event.frame_ref)
    else:
        raise ValueError(
            f"frame_ref scheme {parsed.scheme!r} needs a media-relay resolver"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return image.convert("RGB")


def _runtime_state(req_data: Any) -> MossVLRealtimeRuntimeState:
    state = getattr(req_data, "runtime_state", None)
    if not isinstance(state, MossVLRealtimeRuntimeState):
        raise TypeError("MOSS-VL realtime request data is missing runtime_state")
    return state


def _ends_with_token_ids(values: Any, suffix: tuple[int, ...]) -> bool:
    if len(values) < len(suffix):
        return False
    return tuple(int(value) for value in values[-len(suffix) :]) == suffix
