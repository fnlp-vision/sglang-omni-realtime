# VL API v2 Validation

**English** | [简体中文](./VALIDATION_zh.md)

## Original Accuracy Suite

On 2026-09-11, the unchanged repository accuracy entry point was run on original main `68a0eef` and the feature branch with strict token/event checks:

```bash
bash deployment/moss_vl_realtime/test_accuracy.sh /path/to/model --strict-tokens
```

Result: **PASS**. Scripts, comparison rules, manifest and task contracts were unchanged. Original system/initial prompts, complete PNG sequences, timestamps, prompt events and generation settings were preserved, advancing each event after silence. HF and SGLang each used one H200; all concurrent sessions within a backend shared one GPU.

| Sessions | Original main HF/SGLang token and event matches | Feature HF/SGLang token and event matches | main/v1/v2 per-event protocol text matches | Each protocol endpoint matches its single-session output |
| --- | --- | --- | --- | --- |
| 1 | 4/4 | 4/4 | 4/4 | 4/4 |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 |

Task checks and single/concurrent alignment passed on both branches. Direct cross-branch comparison also matched raw tokens and per-event text in 12/12 HF and 12/12 SGLang runs.

The same manifest was additionally replayed through original main, feature v1 and feature v2 WebSocket endpoints: 36 sessions, checksum-verified original PNG bytes, no image transcoding or prompt changes. Full and per-input-event text matched the original engine suite, and all task checks passed. The protocol does not expose raw token IDs, so WebSocket text equality is not described as raw-token equality. Per-response v2 done and native request-terminal done are checked under their respective contracts, not required to be identical.

The original accuracy suite's all-pass result was reproduced, with no regression on these cases. The following experiments use different inputs and supplement, rather than replace, that suite.

## Supplemental Environment

2026-09-11: H200, one GPU per service (TP=1), BF16, SGLang 0.5.16, Transformers 5.12.1, local MOSS-VL Realtime SGLANG checkpoint. Production configuration: `mem_fraction_static=0.5`, context 131072, four sessions, 60-second visual KV window. No memory/ASR/TTS components.

Compared original main `68a0eef`, native v1 on the feature branch, and v2 on the feature branch. Four repository image fixtures, temperature 0, per-frame processed acknowledgements, visible text joined by `turn_id`. Native and v2 done events intentionally differ. Generation settings: `max_new_tokens=512`, `max_tokens_per_turn=160`. This is neither a latency benchmark nor an HF model accuracy evaluation.

## Results

| Check | Result |
| --- | --- |
| CPU regressions | 499 passed, 2 skipped |
| Four single-session fixtures | Identical visible text across main/v1/v2; matching final vision/text/total usage |
| Subsequent frames and questions | Identical v1/v2 text and final usage |
| Same-turn responses | Three distinct response IDs in one turn; concatenated text matches v1; silence alone does not settle twice |
| Interruption during output | Identical v1/v2 text; interrupted segment has no normal done and its usage carries forward |
| Visual window | 16 frames spanning 150 seconds; 3536 historical and 1547 resident visual positions; matching v1/v2 text and final usage |
| Input validation | Unknown fields, sequence gaps and corrupt images rejected; valid same-sequence retry succeeds |
| Abort and timeout | Idle, output and in-flight-input aborts and missing-binary timeout terminate correctly; telemetry disabled still yields final usage |
| Context limit | Oversized initial/extended input returns context_exhausted; extension error includes seq_no; uncommitted input is not billed |
| Capacity and recovery | v1/v2 share four slots; excess connection rejected; new sessions work after disconnect; no outstanding requests |

Recorded v2 sessions were checked for settlement arithmetic, monotonic totals, one terminal event, no post-terminal output and telemetry frequency. Isolated test services exited cleanly without owned child processes remaining.

### Open-Description Concurrent Exact Text

This experiment uses open-description prompts and the first four frames transcoded to JPEG, unlike the original manifest suite above. Counts show exact matches against the corresponding single-session text, not semantic accuracy. Sessions ran independently without enforcing identical batch composition.

| Version | 2 sessions | 4 sessions |
| --- | --- | --- |
| Original main | 1/2 | 3/4 |
| Current v1 | 0/2 | 3/4 |
| Current v2 | 0/2 | 2/4 |

No cross-session output, missing terminal events or usage arithmetic errors were observed. Concurrent exact-text alignment is not a pass. Original main also changes wording; these results do not establish v2 as the cause, and token/logit-level root-cause analysis was not performed. Descriptions refer to the input subjects, but matching text does not establish factual correctness of every detail.

## Regression Command

```bash
python -m pytest tests/unit_test/moss_vl_realtime \
  vl_api_adapter/tests/test_vl_api_v2.py \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests -q
```

This is the CPU regression command, not the GPU comparison runner. Skips are the model-step comparison requiring CUDA and the processor comparison requiring an explicit model path. Multi-GPU TP/NPU, full Demo/memory/ASR/TTS integration, long-running stress and actual worker crashes were not tested. CPU tests cover failure paths such as unavailable final snapshots.
