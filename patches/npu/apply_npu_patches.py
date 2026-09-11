"""Apply known Ascend environment patches, rejecting incompatible sources."""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile


def detect_patch_set(site: Path) -> str:
    path = site / 'sglang/srt/model_executor/model_runner.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ModelRunner':
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                    names = {arg.arg for arg in method.args.args + method.args.kwonlyargs}
                    if 'ps' in names:
                        return '0.5.16'
                    if {'tp_rank', 'tp_size'} <= names:
                        return '0.5.14'
    raise RuntimeError('cannot identify the installed SGLang ModelRunner API')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site_packages', nargs='?', type=Path)
    parser.add_argument('--patch-set', choices=('auto', '0.5.14', '0.5.16'), default='auto',
                        help='select source layout explicitly for vendor backports')
    args = parser.parse_args()
    if args.site_packages is None:
        spec = importlib.util.find_spec('sglang')
        if spec is None or not spec.origin:
            parser.error('sglang is not installed in this interpreter')
        site = Path(spec.origin).resolve().parent.parent
    else:
        site = args.site_packages.resolve()
    if not (site / 'sglang').is_dir():
        parser.error(f'expected a site-packages directory containing sglang: {site}')
    if shutil.which('patch') is None:
        parser.error('GNU patch is required')

    here = Path(__file__).resolve().parent
    native = 'sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py'
    backend = 'sglang/srt/hardware_backend/npu/attention/ascend_backend.py'
    moss = 'sglang/srt/models/moss_vl.py'
    operations = [
        ('0001-fix-vision-rope-for-transformers-5-and-npu-inv-freq.patch', moss),
        ('0002-fix-cross-attention-extend-sdpa-alignment.patch', native),
        ('0003-preserve-frame-visibility.patch', None),
    ]
    patch_set = detect_patch_set(site) if args.patch_set == 'auto' else args.patch_set
    if patch_set == '0.5.14':
        operations = [('sglang-0.5.14.patch', None)]
    print(f'SGLang patch set: {patch_set}')
    originals = {path: (site / path).read_bytes() for path in (moss, native, backend)}
    # Resolve all patches on private copies before writing any installed file.
    with tempfile.TemporaryDirectory(prefix='moss-npu-patches-') as temp:
        staging = Path(temp)
        for path, data in originals.items():
            destination = staging / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        for name, target in operations:
            patch = (here / name).read_bytes()
            command = ['patch', '--batch', '--silent', '-p1']
            if target:
                command.append(target)
            def run(*options):
                return subprocess.run(command + list(options), input=patch,
                                      cwd=staging, capture_output=True)
            if run('--forward', '--dry-run').returncode == 0:
                applied = run('--forward')
                if applied.returncode:
                    raise RuntimeError(f'failed applying {name}: {applied.stderr.decode()}')
                print(f'Applied: {name}')
            elif run('--reverse', '--dry-run').returncode == 0:
                print(f'Already applied: {name}')
            else:
                raise RuntimeError(
                    f'{name} is incompatible with the installed sources. '
                    'No installed files were changed; use the documented SGLang build.'
                )
        updated = {path: (staging / path).read_bytes() for path in originals}
        for path, data in updated.items():
            ast.parse(data, filename=path)
        for path, before in originals.items():
            if (site / path).read_bytes() != before:
                raise RuntimeError(f'installed source changed concurrently: {path}')
        for path, data in updated.items():
            if data != originals[path]:
                destination = site / path
                backup = destination.with_suffix(destination.suffix + '.moss-npu.bak')
                if not backup.exists():
                    backup.write_bytes(originals[path])
                destination.write_bytes(data)
    print(f'Ascend patches installed under {site}; restart serving processes.')


if __name__ == '__main__':
    main()
