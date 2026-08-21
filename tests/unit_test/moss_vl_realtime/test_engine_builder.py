from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.models.moss_vl_realtime.engine_builder import (
    DECODE_GRAPH_ENCODER_LEN_FILL_VALUE,
    MossVLRealtimeEngineBuilder,
)


def _make_builder(**overrides) -> MossVLRealtimeEngineBuilder:
    kwargs = {
        "max_running_requests": 1,
        "max_new_tokens": 64,
        "context_length": 32768,
        "mem_fraction_static": None,
    }
    kwargs.update(overrides)
    return MossVLRealtimeEngineBuilder(**kwargs)


def _fake_model_worker() -> SimpleNamespace:
    hf_config = SimpleNamespace()
    return SimpleNamespace(
        model_runner=SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    )


def test_generation_defaults_keep_cuda_graph_disabled_by_default() -> None:
    defaults = _make_builder().generation_defaults(dtype="bfloat16")

    assert defaults["disable_cuda_graph"] is True
    assert "disable_prefill_cuda_graph" not in defaults
    # FlashInfer is the project's decode backend in all modes; setting any
    # backend dimension requires pinning prefill explicitly as well.
    assert defaults["decode_attention_backend"] == "flashinfer"
    assert defaults["prefill_attention_backend"] == "flashinfer"


def test_generation_defaults_enable_decode_graph_only() -> None:
    defaults = _make_builder(disable_cuda_graph=False).generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is False
    # Frame extend keeps dynamic shapes and must stay eager.
    assert defaults["disable_prefill_cuda_graph"] is True
    assert defaults["page_size"] == 1
    assert defaults["decode_attention_backend"] == "flashinfer"


def test_generation_defaults_pass_through_page_size() -> None:
    assert _make_builder().generation_defaults(dtype="bfloat16")["page_size"] == 1
    assert (
        _make_builder(page_size=16).generation_defaults(dtype="bfloat16")["page_size"]
        == 16
    )


def test_page_size_must_be_positive_and_divide_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="page_size"):
        _make_builder(page_size=0)
    # 4096 (chunked_prefill_size / max_prefill_tokens) % 48 != 0
    with pytest.raises(ValueError, match="page_size"):
        _make_builder(page_size=48)


def test_async_decode_defaults_off_and_requires_page_size_one() -> None:
    builder = _make_builder()
    assert builder.enable_async_decode is False

    with pytest.raises(ValueError, match="page_size"):
        _make_builder(enable_async_decode=True, page_size=16)


def test_extra_scheduler_kwargs_pass_async_decode_through() -> None:
    builder = _make_builder(enable_async_decode=True)
    builder.processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
    kwargs = builder.extra_scheduler_kwargs()

    assert kwargs["enable_overlap"] is False
    assert kwargs["enable_async_decode"] is True
    assert kwargs["async_decode_min_batch_size"] == 1

    default_builder = _make_builder()
    default_builder.processor = SimpleNamespace(
        tokenizer=SimpleNamespace(eos_token_id=0)
    )
    default_kwargs = default_builder.extra_scheduler_kwargs()
    assert default_kwargs["enable_async_decode"] is False


def test_setup_model_sets_encoder_len_fill_value_when_graph_enabled() -> None:
    model_worker = _fake_model_worker()

    _make_builder(disable_cuda_graph=False).setup_model(
        model_worker=model_worker,
        checkpoint_dir="",
        device="cuda:0",
        gpu_id=0,
        server_args=None,
    )

    hf_config = model_worker.model_runner.model_config.hf_config
    assert hf_config.max_source_positions == DECODE_GRAPH_ENCODER_LEN_FILL_VALUE


def test_setup_model_leaves_hf_config_untouched_when_graph_disabled() -> None:
    model_worker = _fake_model_worker()

    _make_builder().setup_model(
        model_worker=model_worker,
        checkpoint_dir="",
        device="cuda:0",
        gpu_id=0,
        server_args=None,
    )

    hf_config = model_worker.model_runner.model_config.hf_config
    assert not hasattr(hf_config, "max_source_positions")
