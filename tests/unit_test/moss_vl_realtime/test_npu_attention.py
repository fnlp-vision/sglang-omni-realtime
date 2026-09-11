"""Opt-in checks of patched SGLang attention; run on CPU and on an Ascend card."""
import importlib.util
import inspect
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get('MOSSVL_TEST_NPU_PATCHES') != '1',
    reason='requires the installed Ascend environment patches',
)


def backend_and_device():
    device = os.environ.get('NPU_TEST_DEVICE', 'cpu')
    if device.startswith('npu'):
        import torch_npu  # noqa: F401
    spec = importlib.util.find_spec('sglang')
    path = Path(spec.origin).parent / 'srt/hardware_backend/npu/attention/ascend_torch_native_backend.py'
    spec = importlib.util.spec_from_file_location('moss_test_ascend_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.AscendTorchNativeAttnBackend()
    assert 'cross_attention_custom_mask' in inspect.signature(backend.run_sdpa_forward_extend).parameters
    return backend, device


@pytest.mark.parametrize('logit_cap', [0.0, 20.0])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_mixed_requests_visibility_and_fully_masked_rows(logit_cap, dtype):
    backend, device = backend_and_device()
    query = torch.zeros((5, 2, 4), device=device, dtype=dtype)
    keys = torch.zeros((5, 1, 4), device=device, dtype=dtype)
    values = torch.tensor([1, 3, 5, 7, 9], device=device, dtype=dtype)[:, None, None].expand(5, 1, 4).contiguous()
    mask = torch.tensor([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0], device=device, dtype=torch.uint8)
    actual = backend.run_sdpa_forward_extend(
        query, torch.empty_like(query), keys, values,
        torch.tensor([[0, 1, 0], [2, 3, 4]], device=device),
        torch.tensor([0, 1], device=device), torch.tensor([7, 2], device=device),
        torch.tensor([4, 0], device=device), torch.tensor([3, 2], device=device),
        torch.tensor([2, 3], device=device), is_cross_attention=True, enable_gqa=True,
        logit_cap=logit_cap, cross_attention_custom_mask=mask)
    expected = torch.tensor([1, 0, 2, 7, 6], dtype=torch.float32)[:, None, None].expand(5, 2, 4)
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.05, rtol=0.01)


def test_empty_encoder_produces_zero_output():
    backend, device = backend_and_device()
    query = torch.zeros((2, 1, 4), device=device)
    actual = backend.run_sdpa_forward_extend(
        query, torch.empty_like(query), torch.empty((0, 1, 4), device=device),
        torch.empty((0, 1, 4), device=device), torch.empty((1, 0), dtype=torch.long, device=device),
        torch.tensor([0], device=device), torch.tensor([2], device=device),
        torch.tensor([0], device=device), torch.tensor([2], device=device),
        torch.tensor([0], device=device), is_cross_attention=True,
        cross_attention_custom_mask=torch.empty(0, dtype=torch.uint8, device=device))
    torch.testing.assert_close(actual, torch.zeros_like(query))


def test_mask_length_is_validated():
    backend, device = backend_and_device()
    query = torch.zeros((2, 1, 4), device=device)
    with pytest.raises(ValueError, match='Q/KV lengths'):
        backend.run_sdpa_forward_extend(
            query, torch.empty_like(query), query, query,
            torch.tensor([[0, 1]], device=device), torch.tensor([0], device=device),
            torch.tensor([2], device=device), torch.tensor([0], device=device),
            torch.tensor([2], device=device), torch.tensor([2], device=device),
            is_cross_attention=True, cross_attention_custom_mask=torch.ones(1, device=device))
