"""Configuration, process ownership and bundled validation inputs."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def environment(gpu):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.lower().endswith("_proxy")
        and not k.startswith(("REALTIME_FRAME_", "MOSS_DELIVERY_"))
    }
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        NO_PROXY="*",
        no_proxy="*",
        PYTHONDONTWRITEBYTECODE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPATH=os.pathsep.join([str(ROOT), str(HERE)]),
        REALTIME_FRAME_WINDOW_ENABLED="1",
        REALTIME_FRAME_WINDOW_RAW_S=str(CONFIG["raw_window_seconds"]),
        REALTIME_FRAME_POOLING_ENABLED="0",
        SGLANG_OMNI_STRICT_PORT="1",
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def gpu_inventory():
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    )
    return [
        dict(
            index=r[0].strip(),
            uuid=r[1].strip(),
            name=r[2].strip(),
            total_mib=int(r[3]),
            used_mib=int(r[4]),
        )
        for r in csv.reader(output.splitlines())
    ]


def select_gpus(requested, inventory, visible=None):
    ids = (
        None
        if visible is None
        else [v.strip() for v in visible.split(",") if v.strip()]
    )
    available = [
        r for r in inventory if ids is None or r["index"] in ids or r["uuid"] in ids
    ]
    if requested:
        wanted = [v.strip() for v in requested.split(",")]
        if len(wanted) != len(set(wanted)) or not all(wanted):
            raise ValueError("--gpus requires distinct GPU indices or UUIDs")
        picked = []
        for gpu in wanted:
            matches = [r for r in available if gpu in (r["index"], r["uuid"])]
            if len(matches) != 1:
                raise ValueError(f"GPU {gpu} is not visible")
            picked.append(matches[0])
        if len({r["uuid"] for r in picked}) != len(picked):
            raise ValueError("--gpus resolves to duplicate physical GPUs")
    else:
        picked = [
            r
            for r in available
            if r["used_mib"] <= CONFIG["maximum_existing_memory_mib"]
            and (r["total_mib"] - r["used_mib"]) / 1024 >= CONFIG["minimum_free_gib"]
        ][:1]
    if not picked:
        raise ValueError(
            "No idle GPU has sufficient free VRAM; release a GPU or specify --gpus"
        )
    for r in picked:
        if r["used_mib"] > CONFIG["maximum_existing_memory_mib"]:
            raise ValueError(
                f"GPU {r['index']} is occupied; existing processes are not stopped"
            )
        if (r["total_mib"] - r["used_mib"]) / 1024 < CONFIG["minimum_free_gib"]:
            raise ValueError(
                f"GPU {r['index']} has insufficient free VRAM for the 128K profile"
            )
    return picked


def free_port(host="127.0.0.1", port=0):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return sock.getsockname()[1]


def server_command(model, port, host="127.0.0.1", probe=False):
    entry = (
        HERE / "server_probe.py"
        if probe
        else ROOT / "examples/run_moss_vl_realtime_server.py"
    )
    return [
        sys.executable,
        "-u",
        str(entry),
        "--model-path",
        str(model),
        "--gpu",
        "0",
        "--host",
        host,
        "--port",
        str(port),
        "--context-length",
        str(CONFIG["context_length"]),
        "--mem-fraction-static",
        str(CONFIG["mem_fraction_static"]),
        "--max-running-requests",
        str(CONFIG["max_sessions"]),
        "--parked-request-timeout",
        str(CONFIG["parked_timeout_seconds"]),
    ]


class Child:
    def __init__(self, command, env, log, *, new_group=True):
        self.log = Path(log).open("w")
        self.new_group = new_group
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=new_group,
            )
        except BaseException:
            self.log.close()
            raise

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.process.poll() is None:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Child {self.process.pid} exceeded {timeout}s")
            time.sleep(0.2)
        return self.process.returncode

    def stop(self):
        def send(sig):
            try:
                if self.new_group:
                    os.killpg(self.process.pid, sig)
                elif self.process.poll() is None:
                    self.process.send_signal(sig)
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            send(signal.SIGKILL)
            self.process.wait(10)
        finally:
            self.log.close()


def prepare_cases(output):
    import av
    from PIL import Image

    frames_dir = output / "frames"
    frames_dir.mkdir()

    def save(image, name):
        image = image.convert("RGB")
        image.thumbnail((CONFIG["image_max_edge"], CONFIG["image_max_edge"]))
        path = frames_dir / name
        image.save(path, quality=CONFIG["jpeg_quality"])
        return str(path)

    with Image.open(ROOT / "tests/data/cars.jpg") as image:
        cars = save(image, "cars.jpg")
    draw = []
    with av.open(str(ROOT / "tests/data/draw.mp4")) as video:
        for frame in video.decode(video=0):
            if float(frame.time or 0) >= len(draw) / CONFIG["fps"]:
                draw.append(save(frame.to_image(), f"draw_{len(draw):03d}.jpg"))
            if len(draw) >= CONFIG["comparison_frames"]:
                break
    if not draw:
        raise ValueError("Bundled draw.mp4 has no decodable frames")
    return [
        dict(
            case_id="cars",
            frames=[cars],
            question="What vehicles are visible? Answer briefly.",
        ),
        dict(
            case_id="drawing",
            frames=draw,
            question="Describe the drawing in the video. Answer briefly.",
        ),
    ]


def events_for(case, count):
    events = [
        dict(
            type="frame",
            seq_no=i,
            timestamp=i / CONFIG["fps"],
            frame_path=case["frames"][i % len(case["frames"])],
            final=False,
        )
        for i in range(count)
    ]
    events.append(
        dict(
            type="prompt",
            seq_no=count,
            timestamp=count / CONFIG["fps"],
            prompt=case["question"],
            final=True,
        )
    )
    return events


def initial_prompt(case):
    return "Watch the video stream. " + case["question"]


def groups(cases, count):
    return (
        [[case] for case in cases]
        if count == 1
        else [[cases[i % len(cases)] for i in range(count)]]
    )
