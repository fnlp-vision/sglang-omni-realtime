# Ascend Integration

**English** | [简体中文](./README_zh.md)

Native MOSS-VL serving on Ascend, with the shared [legacy VL adapter](../../vl_legacy_adapter/README.md). The native backend serves `/v1/video/realtime`; the separate adapter serves `/v1/realtime`. There is no second legacy protocol implementation inside the backend.

## Environment

Target stack: Ascend 910B2C, CANN 9.0.0, PyTorch/torch_npu 2.11, SGLang 0.5.16 with Ascend support, Transformers 5.12.1, and Python 3.12 or 3.13. Use the NPU maintainer's matching environment and a local [MOSS-VL-Realtime-SGLANG](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG) checkpoint. Do not apply the CUDA dependency lock to this environment. SGLang 0.5.14 shims are retained from the incoming branch but are outside this integration's acceptance target.

Run from the repository root using that environment's Python:

```bash
python -m pip install --no-deps -e .
bash patches/npu/apply_npu_patches.sh
export ASCEND_USE_FA=false
export ASCEND_USE_FIA=false
```

The patch installer operates on the active interpreter's SGLang package; an optional argument selects its `site-packages` directory. Stop serving processes before patching. All patches are staged before installation; already-applied patches are accepted, incompatible sources fail rather than being skipped, and modified files get `.moss-npu.bak` backups. Restart after installation. The environment must already contain application dependencies, including FastAPI, websockets, Pillow, psutil, pytest and pytest-asyncio.

Frame visibility uses the native SDPA extend path with a per-request mask. FA/FIA, speculative and context-parallel paths are not accepted for masked cross-attention. Missing patches prevent MOSS-VL NPU startup. Decode retains its existing all-visible encoder behavior.

## Start

```bash
MODEL_PATH=/path/to/model bash deploy.sh start
# Two TP2 groups, logical devices 0,1 and 2,3:
MODEL_PATH=/path/to/model bash deploy.sh start4
# Two TP4 groups, logical devices 0-3 and 4-7:
MODEL_PATH=/path/to/model bash deploy.sh start8
```

Each command runs in the foreground. Stop with Ctrl-C or SIGTERM from a service manager; cleanup is restricted to owned processes. The former process-name-based `stop`/`restart` commands are not used. Logs are kept in a printed, unique `/tmp/moss-npu-*` directory. Any child failure stops the supervised deployment.

| Setting | Default |
| --- | --- |
| Native backend ports | `PORT=8000`, then 8001 for the second group |
| Legacy adapter ports | `ADAPTER_PORT=18600`, then 18601 |
| Single group device | `GPU=0`, or `GPUS=0,1` for TP |
| Group override | `NPU_GROUPS='0,1;2,3'` |
| Backend session capacity | Device count in each group; override `MAX_RUNNING_REQUESTS` |
| Adapter capacity | `MAX_INFLIGHT=1`; set independently for concurrent callers |
| Context / memory fraction | TP2: 8192 / 0.80; otherwise 32768 / 0.70 |

`CONTEXT_LENGTH`, `MEM_FRACTION` and `HOST` override the defaults. Device IDs are logical indices in the visible-device list. Verify HCCS planes on the actual host; the example grouping is not a portable hardware topology guarantee. Each adapter connects to its own backend, not to a load balancer. Capacity and memory defaults need NPU acceptance testing.

## Acceptance

**Status: implementation handoff, not validated. No tests or inference were run for these integration fixes.** The recipient must record the commit, environment, commands, raw outputs and results before merging into main. Historical CUDA measurements elsewhere are not results for this NPU revision.

```bash
python -m pytest tests/unit_test/vendor/test_sglang_server_args.py \
  vl_legacy_adapter/tests/test_lifecycle.py \
  tests/unit_test/moss_vl_realtime/test_legacy_perf_probe.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=cpu python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
MOSSVL_TEST_NPU_PATCHES=1 NPU_TEST_DEVICE=npu:0 python -m pytest \
  tests/unit_test/moss_vl_realtime/test_npu_attention.py -q
python vl_legacy_adapter/tests/smoke_client.py \
  --url ws://127.0.0.1:18600/v1/realtime \
  --testdata deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122
bash perf.sh
```

`perf.sh` defaults to five rounds of four distinct repository frames at 160 tokens/s, with a real 10 s deadline per round. It requires all ACKs and visible text, preserves output interleaved with ACKs, and measures TTFT from connection start to the first visible text. `FRAMES_DIR`, `NFRAMES`, `ROUNDS`, `TOKEN_RATE`, `TIMEOUT_S` and `VL_MODEL_WS_URL` configure it. These are protocol/latency checks, not HF equivalence proofs.

Required before main: NPU HF/SGLang comparison for multi-frame extend, questions before/after frame boundaries, retained visual KV, visual-window transitions, no-visible-frame rows and mixed-length 1/2/4 sessions. Use matching inputs and sampling in the maintainer's HF harness; retain output and first-divergence evidence. Also cover delayed/missing binaries, malformed images, setup failure, cancellation/reconnect and memory reclamation, plus CUDA regressions. The existing CUDA accuracy launcher is not an NPU evaluation script.
