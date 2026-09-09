# SGLang-Omni Realtime for MOSS-VL

English | [简体中文](./README_zh.md)

A realtime video-understanding backend built on [SGLang-Omni](https://github.com/sgl-project/sglang-omni) and [SGLang](https://github.com/sgl-project/sglang). Clients stream timestamped frames and questions over WebSocket and receive incremental text and input-processing events.

## Related Projects

- [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG): model weights and Transformers 5.12.1-compatible model/processor code.
- [MOSS-VL-Realtime Demo](https://github.com/fnlp-vision/MOSS-VL-Realtime_Demo): browser UI, ASR/TTS, memory, and REST gateway. Use its installer for the complete application instead of installing this backend separately.

## Features

- Incremental visual features and KV cache for JPEG, PNG, and WebP frames.
- Stateful interaction, question interruption, and wake-up after silence.
- Dynamic multi-session scheduling, single-GPU and tensor-parallel inference.
- Decode CUDA Graph, visual KV windows, and bounded input queues.

## Installation

Use Linux x86_64, Python 3.12, a CUDA 13-compatible NVIDIA driver, Git, a C/C++ compiler, CMake, and FFmpeg. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. Keep this environment separate from the Demo and other inference engines.

From the repository root:

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip sync --require-hashes \
  --build-constraint deployment/repro/build-constraints.txt \
  --index-url https://pypi.org/simple deployment/repro/requirements.lock
uv pip install --no-deps --no-build-isolation -e .
uv pip check
```

The dependency file selects SGLang 0.5.16, Transformers 5.12.1, Torch 2.11.0, and a CUDA 13.0 toolchain. See the [installation guide](./docs/get_started/installation.md) (Chinese) for system packages and troubleshooting.

## Download and Start

```bash
export MODEL_PATH="$HOME/models/MOSS-VL-Realtime-SGLANG"
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --revision bcfd9ccf1e9db2896ad852301cc8dde4a6349c78 \
  --local-dir "$MODEL_PATH"

export CUDA_HOME="$(python deployment/repro/cuda_toolkit.py)"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
python deployment/moss_vl_realtime/check_env.py "$MODEL_PATH"
bash deployment/moss_vl_realtime/start.sh "$MODEL_PATH"
```

The model is public. Keep its configuration, custom code, tokenizer, processor, and all weight shards together. In a new terminal, reactivate `.venv` and set the same model/toolchain variables before starting.

The launcher selects an idle GPU and warms up the model. Use `--gpus 0` to choose a GPU, `--port 18510` to change the port, or `--dry-run` to inspect configuration.

| Default | Value |
| --- | --- |
| Address | `http://127.0.0.1:18500` |
| Sessions / context | 4 / 131072 |
| Memory fraction | 0.5 |
| Visual KV window | 60 seconds, enabled |
| Pooling / async decode | Disabled |

These defaults apply to this `start.sh` and come from [config.json](./deployment/moss_vl_realtime/config.json). The Python launcher and Demo-managed deployment have their own defaults. Adjust memory and concurrency for your GPU.

```bash
curl --fail http://127.0.0.1:18500/health
curl --fail http://127.0.0.1:18500/v1/models
```

## Client Example

The following uses frames included in the repository:

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:18500/v1/video/realtime \
  --prompt "Describe the visible scene." \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0000.png --timestamp 0.0 \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0001.png --timestamp 1.0
```

The client marks the last frame as `final`. Use the Demo for continuous camera interaction. Frames and prompts share increasing `seq_no` values; wait for `input.frame.ready` before sending image bytes. `session.abort` ends the session.

`max_tokens_per_turn` is a per-session tokens/s target, not response length. Protocol fields and advanced deployment options are in the [Realtime Cookbook](./docs/cookbook/moss_vl_realtime.md).

## Connect an Existing Demo

Set these values in the Demo's `.env.deploy` when connecting it manually:

```dotenv
VLM_DEPLOY=sglang_omni
SGLANG_OMNI_URLS=http://127.0.0.1:18500
SGLANG_OMNI_SESSIONS_PER_REPLICA=4
SGLANG_OMNI_CONTEXT_LENGTH=131072
MODEL_PATH=/absolute/path/to/MOSS-VL-Realtime-SGLANG
```

The URL, session capacity, and context must match the backend. The model/tokenizer path must be readable by the Demo. Visual KV eviction does not stop text context from growing; cross-context continuation uses Demo memory rollover.

Services bind to loopback by default. Public access requires authentication, TLS, and access controls.

## Tests and Documentation

- [Accuracy and latency tests](./deployment/moss_vl_realtime/README.md).
- [Protocol and advanced settings](./docs/cookbook/moss_vl_realtime.md).
- [Capacity planning](./docs/cookbook/moss_vl_realtime_capacity.md).

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime*.py
```

## License

This project retains the upstream [Apache License 2.0](./LICENSE). Thanks to the SGLang-Omni, SGLang, and MOSS-VL teams. Report issues in [this repository](https://github.com/fnlp-vision/sglang-omni-realtime/issues).
