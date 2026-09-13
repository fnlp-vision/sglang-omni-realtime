"""Installer preflight and idempotence checks using private source fixtures."""
import difflib
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
MOSS = 'sglang/srt/models/moss_vl.py'
NATIVE = 'sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py'
BACKEND = 'sglang/srt/hardware_backend/npu/attention/ascend_backend.py'
RUNNER = 'sglang/srt/model_executor/model_runner.py'


@pytest.fixture
def installer(tmp_path, monkeypatch):
    if shutil.which('patch') is None:
        pytest.skip('GNU patch is required')
    spec = importlib.util.spec_from_file_location('patch_installer', ROOT/'patches/npu/apply_npu_patches.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    site = tmp_path/'site-packages'
    content = 'base = 0\nmask = 0\ntorch_path = 0\n'
    for rel in (MOSS, NATIVE, BACKEND, RUNNER):
        path = site/rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (site/RUNNER).write_text('class ModelRunner:\n    def __init__(self, ps): pass\n')
    patches = tmp_path/'patches'
    patches.mkdir()
    monkeypatch.setattr(module, '__file__', str(patches/'apply.py'))

    def patch(name, rel, old, new):
        text = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                          fromfile='a/'+rel, tofile='b/'+rel, n=0))
        (patches/name).write_text(text)

    patch('0001-fix-vision-rope-for-transformers-5-and-npu-inv-freq.patch', MOSS,
          content, content.replace('base = 0', 'base = 1'))
    patch('0002-fix-cross-attention-extend-sdpa-alignment.patch', NATIVE,
          content, content.replace('base = 0', 'base = 1'))
    patch('0003-preserve-frame-visibility.patch', BACKEND,
          content, content.replace('mask = 0', 'mask = 1'))
    patch('0004-use-torch-cross-attention.patch', NATIVE,
          content, content.replace('torch_path = 0', 'torch_path = 1'))
    monkeypatch.setattr(sys, 'argv', ['apply.py', str(site)])
    return module, site


def snapshot(site):
    return {str(p.relative_to(site)):p.read_bytes() for p in site.rglob('*') if p.is_file()}


def test_explicit_target_is_patched_and_repeat_is_noop(installer):
    module, site = installer
    module.main()
    assert 'torch_path = 1' in (site/NATIVE).read_text()
    assert 'mask = 1' in (site/BACKEND).read_text()
    assert (site/(NATIVE+'.moss-npu.bak')).is_file()
    first = snapshot(site)
    module.main()
    assert snapshot(site) == first


def test_incompatible_later_patch_leaves_all_targets_unchanged(installer):
    module, site = installer
    (site/NATIVE).write_text('unsupported_source = True\n')
    before = snapshot(site)
    with pytest.raises(RuntimeError, match='incompatible'):
        module.main()
    assert snapshot(site) == before


@pytest.mark.parametrize('signature,expected', [('ps', '0.5.16'), ('tp_rank, tp_size', '0.5.14')])
def test_api_detection_uses_selected_target(installer, signature, expected):
    module, site = installer
    (site/RUNNER).write_text(f'class ModelRunner:\n    def __init__(self, {signature}): pass\n')
    assert module.detect_patch_set(site) == expected


def test_unknown_api_is_rejected(installer):
    module, site = installer
    (site/RUNNER).write_text('class ModelRunner:\n    def __init__(self, unknown): pass\n')
    before = snapshot(site)
    with pytest.raises(RuntimeError, match='cannot identify'):
        module.main()
    assert snapshot(site) == before
