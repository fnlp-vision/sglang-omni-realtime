"""Portable delivery entrypoints without loading a model or discovering GPUs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[3] / "deployment/moss_vl_realtime"


@pytest.fixture
def modules(monkeypatch):
    spec = importlib.util.spec_from_file_location("common", HERE / "common.py")
    common = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "common", common)
    spec.loader.exec_module(common)
    entry = common.load_module("delivery_test_entry", HERE / "entry.py")
    return common, entry


def inventory():
    return [
        dict(
            index=str(i),
            uuid=f"GPU-{i}",
            name="fixture",
            total_mib=81920,
            used_mib=40000 if i == 0 else 735,
        )
        for i in range(4)
    ]


def test_single_gpu_default_and_explicit_parallel_waves(modules):
    common, entry = modules
    picked = common.select_gpus(None, inventory())
    assert [g["index"] for g in picked] == ["1"]
    assert [[name for name, _ in wave] for wave in entry.execution_waves(picked)] == [
        ["tf"],
        ["sglang"],
    ]
    picked = common.select_gpus("2,3", inventory())
    assert len(entry.execution_waves(picked)) == 1
    assert [g["index"] for _, g in entry.execution_waves(picked)[0]] == ["2", "3"]


@pytest.mark.parametrize(
    "requested,visible",
    [("0", None), ("2,2", None), ("2,GPU-2", None), ("2", "1"), (None, "")],
)
def test_reject_busy_duplicate_or_invisible_gpu(modules, requested, visible):
    with pytest.raises(ValueError):
        modules[0].select_gpus(requested, inventory(), visible)


def test_visibility_and_low_memory(modules):
    common, _ = modules
    assert common.select_gpus(None, inventory(), "GPU-3")[0]["index"] == "3"
    devices = inventory()[1:2]
    devices[0]["total_mib"] = 24000
    with pytest.raises(ValueError):
        common.select_gpus("1", devices)


def test_server_is_single_gpu_four_sessions_and_environment_is_direct(
    modules, monkeypatch
):
    common, _ = modules
    command = common.server_command("/model with spaces", 18500)
    assert command[command.index("--gpu") + 1] == "0"
    assert command[command.index("--max-running-requests") + 1] == "4"
    assert "--tp-size" not in command
    assert "/model with spaces" in command
    monkeypatch.setenv("HTTPS_PROXY", "http://unused")
    monkeypatch.setenv("REALTIME_FRAME_POOLING_ENABLED", "1")
    env = common.environment("GPU-2")
    assert "HTTPS_PROXY" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-2"
    assert env["REALTIME_FRAME_POOLING_ENABLED"] == "0"


def test_case_schedule_and_manifest_groups(modules):
    common, _ = modules
    case = dict(case_id="cars", frames=["a.jpg", "b.jpg"], question="Question?")
    events = common.events_for(case, 12)
    assert [e["seq_no"] for e in events] == list(range(13))
    assert events[-1]["type"] == "prompt" and events[-1]["final"]
    assert all(not e["final"] for e in events[:-1])
    assert len(common.groups([case, case], 1)) == 2
    assert len(common.groups([case, case], 4)[0]) == 4


def test_model_validation_and_dry_run(modules, tmp_path):
    _, entry = modules
    with pytest.raises(ValueError):
        entry.validate_model(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MossVLForConditionalGeneration"]})
    )
    assert entry.main(["test", str(tmp_path), "--dry-run"]) == 0


def test_missing_worker_is_not_reported_as_pass(modules, tmp_path):
    common, entry = modules
    common.write_json(
        tmp_path / "tf.json",
        [
            dict(
                backend="TF",
                phase="comparison",
                sessions=1,
                case="cars",
                status="PASS",
                text="cars",
                frames=12,
                expected_frames=12,
            )
        ],
    )
    assert not entry.report(
        tmp_path, {"tf": 0}, {"gpus": inventory()[1:2], "revision": "fixture"}
    )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert not result["passed"]
    assert any(row["phase"] == "worker" for row in result["results"])


def test_existing_failed_case_is_not_called_an_incomplete_worker(modules, tmp_path):
    common, entry = modules
    for backend in ("tf", "sglang"):
        common.write_json(
            tmp_path / f"{backend}.json",
            [
                dict(
                    backend=backend,
                    phase="comparison",
                    sessions=4,
                    case="drawing",
                    status="FAIL",
                    checks={"visible_output": False},
                )
            ],
        )
    assert not entry.report(
        tmp_path,
        {"tf": 1, "sglang": 1},
        {"gpus": inventory()[1:2], "revision": "fixture"},
    )
    result = json.loads((tmp_path / "summary.json").read_text())
    assert len(result["results"]) == 2
    assert "visible_output" in (tmp_path / "report.md").read_text()


def test_occupied_port_is_not_reused(modules):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        with pytest.raises(OSError):
            modules[0].free_port(port=sock.getsockname()[1])
