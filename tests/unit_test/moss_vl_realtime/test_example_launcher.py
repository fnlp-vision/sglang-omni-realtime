from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[3]
_LAUNCHER_PATH = _REPO_ROOT / "examples" / "run_moss_vl_realtime_server.py"
_SPEC = importlib.util.spec_from_file_location(
    "moss_vl_realtime_launcher", _LAUNCHER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


def test_launcher_exposes_venv_console_scripts_without_activation(monkeypatch) -> None:
    python = "/project/.venv-main/bin/python"
    monkeypatch.setattr(_LAUNCHER.sys, "executable", python)
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin")

    _LAUNCHER._ensure_python_bin_on_path()
    _LAUNCHER._ensure_python_bin_on_path()

    assert os.environ["PATH"].split(os.pathsep) == [
        "/project/.venv-main/bin",
        "/usr/local/bin",
        "/usr/bin",
    ]


def _parse(monkeypatch, *argv: str):
    monkeypatch.setattr(
        _LAUNCHER.sys, "argv", ["run_moss_vl_realtime_server.py", *argv]
    )
    return _LAUNCHER.parse_args()


def test_launcher_async_decode_flag_defaults_off(monkeypatch) -> None:
    args = _parse(monkeypatch, "--model-path", "/models/x")

    assert args.enable_async_decode is False


def test_launcher_async_decode_flag_opt_in(monkeypatch) -> None:
    args = _parse(monkeypatch, "--model-path", "/models/x", "--enable-async-decode")

    assert args.enable_async_decode is True


def test_launcher_async_decode_requires_page_size_one(monkeypatch) -> None:
    import pytest

    with pytest.raises(SystemExit):
        _parse(
            monkeypatch,
            "--model-path",
            "/models/x",
            "--enable-async-decode",
            "--kv-page-size",
            "16",
        )
