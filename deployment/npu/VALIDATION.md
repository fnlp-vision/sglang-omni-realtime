# Ascend Integration Validation

**English** | [简体中文](./VALIDATION_zh.md)

## Scope

Date: 2026-09-13. Integration base: `c0bacdd`; PR #1 head: `fd921c2`. This record covers their integration, not subsequent main changes or end-to-end model acceptance.

The integration adopts the PR's explicit Torch cross-attention path and deployment-default centralization. It retains platform-specific startup locks, configuration error handling, instance-local version compatibility, fail-closed patch installation and owned-process cleanup. Mask naming, shape validation, empty-encoder handling and zero output for fully masked rows remain consistent across both patch sets. CUDA production defaults are unchanged.

## Results

| Check | Result |
| --- | --- |
| ServerArgs, version compatibility, NPU locks, deployment defaults, installer and platform guards | 48 passed |
| Stage process environment, GPU memory helpers, execution bridge and legacy performance probe | 56 passed |
| 0.5.16 patch installation and repeat installation on private sources | Passed; repeat leaves files unchanged |
| 0.5.14 patch installation and repeat installation on pinned upstream private sources | Passed; repeat leaves files unchanged |
| Patched 0.5.16 attention on CPU | 9 passed |
| Patched 0.5.14 attention on CPU | 9 passed |

Attention checks cover mixed Q/KV lengths, grouped-query attention, FP32/BF16, softcapping, fully masked rows, empty encoder input, malformed masks, random-input reference comparison and unchanged self-attention dispatch. Cross-attention tests fail if fused SDPA is called. Installer tests cover explicit targets, API detection, idempotence and leaving originals unchanged on incompatible patches.

Source fixtures: installed SGLang 0.5.16 from the established backend environment, and the official SGLang `v0.5.14` source tag. Tests only patched private copies; installed model environments were not changed. These checks validate source integration and CPU arithmetic, not vendor NPU binaries.

## Pending Acceptance

Real CUDA/NPU inference, CUDA TP startup, NPU HF alignment with original fixtures, mixed-length 1/2/4 sessions, visual-window transitions and device-memory recovery remain required. Run both supported NPU runtime variants. The GPU SSH tunnel was unavailable during this integration, and no NPU device was used.

Commands and environment requirements are in the [deployment guide](./README.md). Merge into current main only after separately integrating its later changes and recording hardware results.
