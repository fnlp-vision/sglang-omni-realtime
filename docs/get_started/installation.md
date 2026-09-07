# Installation

Install this checkout to use the MOSS-VL realtime backend. Its dependencies are
defined in [pyproject.toml](../../pyproject.toml).

## Environment

Use Linux, an NVIDIA GPU and Python 3.12. The current stack pins:

| Component | Version |
| --- | --- |
| SGLang | 0.5.16 |
| Transformers | 5.12.1 |
| PyTorch / torchvision / torchaudio | 2.11.0 / 0.26.0 / 2.11.0 |
| FlashInfer | 0.6.14, CUDA 13 extra |
| torchcodec | 0.11.1 |

Prepare a compatible NVIDIA driver and CUDA toolchain. The dependency set
includes CUDA 13 NIXL and Mooncake wheels; native builds may also require UCX,
compiler tools and FFmpeg libraries. The [Dockerfile](../../docker/Dockerfile)
records the upstream environment's build steps.

## Install from source

With `uv` installed:

```bash
git clone https://github.com/fnlp-vision/sglang-omni-realtime.git
cd sglang-omni-realtime
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e .
```

The Python distribution name remains `sglang-omni`. Installing this local
checkout selects the specialized implementation. Use separate environments
for the Demo and optional TTS engines.

## Download the checkpoint

Use [OpenMOSS-Team/MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG),
which contains the Transformers 5.12.1-compatible model and processor code.
For private-repository access, sign in with an authorized account using
`hf auth login`.

```bash
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --local-dir /path/to/MOSS-VL-Realtime-SGLANG
```

Continue with the [realtime serving guide](../cookbook/moss_vl_realtime.md).
For installation of the general upstream framework and its other platforms,
refer to the [official SGLang-Omni documentation](https://sgl-project.github.io/sglang-omni/).
