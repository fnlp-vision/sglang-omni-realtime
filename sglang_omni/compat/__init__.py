"""Compatibility shims for running sglang-omni on sglang 0.5.14 (NPU build).

The upstream sglang-omni-realtime sources target the 0.5.16 scheduler
contract (``NextBatchPlan``-returning prefill planning, split ModelRunner
phases). The NPU environment ships a 0.5.14 build with the Ascend hardware
backend instead. This module bridges the differences at runtime:

- ``NextBatchPlan`` dataclass missing from ``schedule_batch``
- ``Scheduler.get_next_batch_to_run`` / ``get_new_batch_prefill`` signatures
  (0.5.14 owns ``running_batch``/``last_batch`` on the scheduler itself and
  returns the batch directly)
- ModelRunner phase methods (``alloc_memory_pool``,
  ``init_attention_backends``) that do not exist pre-0.5.15; the 0.5.14
  constructor already performs those steps.

Everything is idempotent and a no-op on a true 0.5.16 runtime.
"""

from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_bridge_applied = False


@dataclass
class NextBatchPlan:
    """0.5.16 scheduler plan contract, reintroduced for 0.5.14."""

    batch_to_run: Optional[Any] = None
    running_batch: Optional[Any] = None


def _sglang_version_tuple() -> tuple[int, ...]:
    import sglang

    try:
        parts = []
        for piece in sglang.__version__.split(".")[:3]:
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)
    except Exception:
        return (0, 0, 0)


def needs_bridge() -> bool:
    """True when the installed sglang predates the 0.5.16 scheduler contract.

    Detected off ``ModelRunner.__init__`` (0.5.16 takes a composed ``ps``
    ParallelState; 0.5.14 takes flat rank arguments), which is unaffected by
    any shim this module installs.
    """
    try:
        import inspect

        from sglang.srt.model_executor.model_runner import ModelRunner

        return "ps" not in inspect.signature(ModelRunner.__init__).parameters
    except Exception:
        return False


def export_next_batch_plan() -> None:
    """Make ``NextBatchPlan`` importable from sglang.srt.managers.schedule_batch."""
    if not needs_bridge():
        return
    import sglang.srt.managers.schedule_batch as sb

    if not hasattr(sb, "NextBatchPlan"):
        sb.NextBatchPlan = NextBatchPlan


def apply_scheduler_bridge() -> None:
    """Adapt 0.5.14 Scheduler methods to the 0.5.16 calling contract.

    After this, callers can invoke
    ``scheduler.get_next_batch_to_run(running_batch, last_batch)`` and receive
    a ``NextBatchPlan``, and ``scheduler.get_new_batch_prefill(running_batch)``
    returns a ``NextBatchPlan`` — while the 0.5.14 internals (invoked through
    the saved originals) keep operating on their native signatures. The
    temporary instance-level shadowing inside ``patched_next`` keeps the 0.5.14
    planning path calling the original no-arg prefill method even when an
    OmniScheduler subclass overrides the 0.5.16-shaped one.
    """
    global _bridge_applied
    if _bridge_applied or not needs_bridge():
        return
    export_next_batch_plan()

    from sglang.srt.managers.scheduler import Scheduler as S

    orig_next = S.get_next_batch_to_run
    orig_prefill = S.get_new_batch_prefill

    def patched_next(self: Any, running_batch: Any = None, last_batch: Any = None):
        if running_batch is not None:
            self.running_batch = running_batch
        if last_batch is not None:
            self.last_batch = last_batch

        def _prefill_and_unwrap(*args, **kwargs):
            # Unshadow first so this (and any later call in the planning
            # path) reaches the Omni/OmniScheduler hook with its 0.5.16
            # NextBatchPlan contract; unwrap to the batch the 0.5.14
            # planning path expects.
            self.__dict__.pop("get_new_batch_prefill", None)
            plan = self.get_new_batch_prefill(self.running_batch)
            self.running_batch = plan.running_batch
            return plan.batch_to_run

        self.get_new_batch_prefill = _prefill_and_unwrap
        try:
            batch = orig_next(self)
        finally:
            self.__dict__.pop("get_new_batch_prefill", None)
        return NextBatchPlan(batch_to_run=batch, running_batch=self.running_batch)

    def patched_prefill(self: Any, running_batch: Any = None):
        if running_batch is not None:
            self.running_batch = running_batch
        batch = orig_prefill(self)
        return NextBatchPlan(batch_to_run=batch, running_batch=self.running_batch)

    def init_metrics_collector(self: Any, tp_rank: int, pp_rank: int, dp_rank: Any):
        from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector

        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

    def init_metrics_reporter(self: Any, tp_rank: int, pp_rank: int, dp_rank: Any):
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            SchedulerMetricsReporter,
        )

        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

    S.get_next_batch_to_run = patched_next
    S.get_new_batch_prefill = patched_prefill
    S.init_metrics_collector = init_metrics_collector
    S.init_metrics_reporter = init_metrics_reporter
    _bridge_applied = True
    logger.info(
        "sglang-omni: installed 0.5.14 scheduler protocol bridge "
        "(NextBatchPlan-compatible planning methods)"
    )


def model_runner_phase_guard(model_runner: Any) -> None:
    """No-op the split phase methods when the 0.5.14 runner did the work already."""
    if not hasattr(model_runner, "alloc_memory_pool"):
        model_runner.alloc_memory_pool = lambda: None
    if not hasattr(model_runner, "init_attention_backends"):
        model_runner.init_attention_backends = lambda: None


def install_kv_cache_configurator_shim() -> None:
    """Expose ``sglang.srt.mem_cache.kv_cache_configurator`` on 0.5.14.

    0.5.16 composes a ``KVCacheConfigurator`` into the ModelRunner; 0.5.14
    keeps ``_profile_available_bytes`` on the runner's own mixin. The shim
    wraps a runner reference, forwards attribute reads to it (``gpu_id``,
    ``server_args``, …), and delegates profiling to the 0.5.14 mixin method,
    so Omni's colocated-budget subclass works unchanged.
    """
    try:
        import sglang.srt.mem_cache.kv_cache_configurator as _kcc  # noqa: F401

        return
    except ImportError:
        pass

    import sys
    import types

    import sglang.srt.mem_cache as mem_cache_mod
    from sglang.srt.model_executor.model_runner import ModelRunner

    @dataclass
    class KVCacheConfigurator:
        model_runner: Any

        def __getattr__(self, name: str) -> Any:
            return getattr(self.model_runner, name)

        def _profile_available_bytes(self, pre_model_load_memory: float) -> int:
            return self.model_runner._profile_available_bytes(pre_model_load_memory)

    mod = types.ModuleType("sglang.srt.mem_cache.kv_cache_configurator")
    mod.KVCacheConfigurator = KVCacheConfigurator
    sys.modules["sglang.srt.mem_cache.kv_cache_configurator"] = mod
    setattr(mem_cache_mod, "kv_cache_configurator", mod)

    if not hasattr(ModelRunner, "init_kv_cache_configurator"):

        def init_kv_cache_configurator(self: Any) -> None:
            self.kv_cache_configurator = KVCacheConfigurator(model_runner=self)

        ModelRunner.init_kv_cache_configurator = init_kv_cache_configurator

    if not hasattr(ModelRunner, "effective_max_total_num_tokens"):

        @property
        def effective_max_total_num_tokens(self: Any) -> int:
            return self.max_total_num_tokens

        ModelRunner.effective_max_total_num_tokens = effective_max_total_num_tokens
    logger.info("sglang-omni: installed KVCacheConfigurator shim for 0.5.14")


def patch_server_args_fields() -> None:
    """Add 0.5.16-only ServerArgs fields that Omni reads, with safe defaults."""
    try:
        from sglang.srt.server_args import ServerArgs
    except ImportError:
        return

    defaults = {
        "dcp_size": 1,  # decode context parallel (0.5.16)
    }

    original_post_init = ServerArgs.__post_init__

    def post_init(self: Any) -> None:
        original_post_init(self)
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    if getattr(original_post_init, "_omni_compat_patched", False):
        return
    post_init._omni_compat_patched = True
    ServerArgs.__post_init__ = post_init


def install_runtime_context_shim() -> None:
    """Expose ``get_flags`` (0.5.16 structured runtime accessors) on 0.5.14.

    Only Omni-side writers need it; the 0.5.14 srt code paths never call it.
    The flags object mirrors the small subset Omni reads and writes
    (``capture.enable_torch_compile``) and accepts arbitrary writes so other
    optional fields don't break assignments.
    """
    try:
        from sglang.srt.runtime_context import get_flags  # noqa: F401

        return
    except ImportError:
        pass

    import sglang.srt.runtime_context as rc

    class _LooseFlags:
        capture: Any = types.SimpleNamespace(enable_torch_compile=False)

        def __getattr__(self, name: str) -> Any:
            value = types.SimpleNamespace()
            object.__setattr__(self, name, value)
            return value

    _FLAGS = _LooseFlags()

    def get_flags() -> Any:
        return _FLAGS

    rc.get_flags = get_flags


def patch_envs_fields() -> None:
    """Register 0.5.16 env fields that Omni reads on the 0.5.14 Envs registry."""
    try:
        from sglang.srt.environ import envs

        if hasattr(envs, "SGLANG_MAX_NEW_TOKENS_LIMIT"):
            return
        from sglang.srt.environ import EnvInt

        # Runtime setattr skips __set_name__, so name it manually.
        field = EnvInt(65536)
        field.name = "SGLANG_MAX_NEW_TOKENS_LIMIT"
        envs.SGLANG_MAX_NEW_TOKENS_LIMIT = field
    except Exception:
        logger.warning("sglang-omni: could not patch env fields", exc_info=True)


def install_forward_batch_shim() -> None:
    """Bridge ForwardBatch.init_new's per-forward override kwargs.

    Omni (0.5.16 contract) passes ``capture_hidden_mode`` /
    ``return_hidden_states_before_norm`` explicitly; 0.5.14 reads
    ``capture_hidden_mode`` off the ScheduleBatch. Translate the kwargs into
    batch-field writes so the native constructor keeps its contract.
    """
    import inspect

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    params = inspect.signature(ForwardBatch.init_new).parameters
    if "capture_hidden_mode" in params and "return_hidden_states_before_norm" in params:
        return

    original = ForwardBatch.init_new.__func__

    @classmethod
    def init_new(cls, batch, model_runner, capture_hidden_mode=None, **kwargs):
        if capture_hidden_mode is not None and hasattr(batch, "capture_hidden_mode"):
            batch.capture_hidden_mode = capture_hidden_mode
        return original(cls, batch, model_runner)

    ForwardBatch.init_new = init_new


def install_req_extend_range_shim() -> None:
    """Expose the 0.5.16 ``Req.extend_range`` slot on 0.5.14.

    0.5.16 tracks the pending extend window explicitly; 0.5.14 derives it
    from ``len(fill_ids) - len(prefix_indices)``, which Omni's realtime
    segment append already keeps consistent. The shim only provides the
    storage slot so Omni's save/restore symmetry works.
    """
    try:
        from sglang.srt.managers.schedule_batch import Req
    except ImportError:
        return

    if hasattr(Req, "set_extend_range"):
        return

    def set_extend_range(self: Any, start: int, end: int) -> None:
        self.extend_range = (int(start), int(end))

    Req.set_extend_range = set_extend_range
    if not hasattr(Req, "extend_range"):
        Req.extend_range = None


def apply_all() -> None:
    export_next_batch_plan()
    apply_scheduler_bridge()
    install_kv_cache_configurator_shim()
    patch_server_args_fields()
    install_runtime_context_shim()
    install_forward_batch_shim()
    install_req_extend_range_shim()
    patch_envs_fields()


__all__ = [
    "NextBatchPlan",
    "apply_all",
    "apply_scheduler_bridge",
    "export_next_batch_plan",
    "model_runner_phase_guard",
    "needs_bridge",
]
