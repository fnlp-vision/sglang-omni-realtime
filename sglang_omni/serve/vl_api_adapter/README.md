# VL API v2

**English** | [简体中文](./README_zh.md)

[Complete API Reference](./API.md) | [Validation Results](../../../deployment/vl_api_v2/VALIDATION.md)

An opt-in protocol adapter for MOSS-VL realtime. A visible text segment followed by a **model-origin silence event** produces one `response.done`. Silence without new visible text does not create an empty response. The backend request continues until `final`, abort, a limit or a failure ends it.

Native v1 and `vl_legacy_adapter` retain their existing semantics. Do not point the current Demo or thin gateway at v2: they treat `response.done` as request-terminal. `turn_id` still advances on new questions; `response_id` and `response_seq` distinguish segments within a turn. No network-silence timer, generation-budget change, model reload or KV reset is used to split responses.

## Start

Install the repository's backend environment, then run from the repository root:

```bash
bash deployment/vl_api_v2/start.sh /path/to/MOSS-VL-Realtime-SGLANG
```

This uses the existing single-GPU production profile and idle-device selection. Native v1 remains on port 18500; v2 uses port 18610 and `/v1/video/realtime`, sharing the loaded model and total session capacity. `--gpus`, `--host` and `--port` configure the original launcher; `VL_API_V2_PORT` selects the v2 port. Defaults do not enable v2 when using the original start script.

For custom/TP settings, add `--vl-api-v2-port 18610` to `examples/run_moss_vl_realtime_server.py`. `VL_API_V2_MODEL_VERSION` optionally supplies deployment revision metadata; absent values are null/unknown. `VL_API_V2_API_KEY` optionally requires `Authorization: Bearer ...` on the v2 WebSocket. With no key, use a trusted network/authenticated gateway. The v2 app exposes no administrative routes.

```bash
python examples/vl_api_v2_client.py \
  --frame deployment/moss_vl_realtime/cases/cd067_sbpro_L2_stream_000122/frame_0003.png
```

`--frame` and `--prompt` can repeat. The client keeps reading past per-response done and stops only at session done.

## Inputs and Limits

Field definitions and the three-phase frame handshake follow the [native input reference](../../../docs/cookbook/moss_vl_realtime.md), with the v2 output differences described below.

The native configure/frame/prompt field set is retained: initial-user `prompt`, separate `system_prompt`, per-extend `max_new_tokens`, tokens/s `max_tokens_per_turn`, input queue capacity and optional `include_usage`. Images may be JPEG, PNG or WebP; frame-attached prompt/final remain supported. Unknown fields, non-finite values, invalid sequences and mismatched image MIME are rejected. Frame/prompt events share continuous `seq_no`; only a clearly rejected, unaccepted input can reuse its sequence.

`session.created` advertises `response.done.per_response`, usage, response IDs and error-correlation capabilities. `session.configured` includes the effective backend `context_limit`. An input frame requires metadata -> ready -> binary -> accepted; processed confirms model consumption. The binary deadline is 10 seconds after ready. Configuration timeout is inherited from the native manager and does not include prefill. Actual parked timeout and limits are advertised, not hard-coded from past measurements.

## Responses and Accounting

```json
{"type":"response.done","response_id":"response_...","response_seq":1,"turn_id":1,"finish_reason":"stop","boundary":"silence","usage":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233,"cumulative":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233}}}
```

- `response.text.delta` carries the active response ID; leading whitespace before the first visible text can have null response identifiers.
- Same-turn continuation creates a new response ID. A new question emits interruption and does not settle an unfinished segment; that consumption stays in the next delta or terminal residual.
- Response sequences include interrupted segments, so gaps between done events are allowed. They are independent of the strictly continuous input seq_no. `boundary` distinguishes model silence from native request completion; silence settlement uses `finish_reason=stop`.
- Natural request finish settles any still-active segment before `session.done`. An already settled segment is not emitted twice. Abort/failure does not fabricate a normally completed response.
- Counters represent committed logical model positions, not FLOPs, characters or KV residency. Vision positions include encoder separators; text input includes actual templates/frame scaffolding/questions. Accepted sampled output IDs include silence/EOS/control tokens. Re-feeding an output token is not another input. Failed/uncommitted appends and discarded lookahead are not billed.
- Each response delta is cumulative minus the last settlement. `text_tokens = text_input_tokens + text_output_tokens`; `total_tokens = vision_tokens + text_tokens`. Eviction never subtracts historical totals. No clamping hides counter regressions.
- `session.usage` remains optional, at most 1 Hz on v2 only, with native context/residency meanings. Its frequency does not determine settlement accuracy. Native v1 telemetry timing is unchanged.
- `session.done.usage` is the frozen backend total, including residual consumption after the last response. Reset/rollover creates a new request ledger; cross-request aggregation belongs to the caller.

## Errors and Finalization

```json
{"type":"session.done","session_id":"video_sess_...","request_id":"video_req_...","reason":"completed","aborted":false,"usage":{"vision_tokens":145,"text_input_tokens":78,"text_output_tokens":10,"text_tokens":88,"total_tokens":233}}
```

Recoverable errors emit `invalid_request`, with `seq_no` when it is a valid identifiable integer. Rejected frames also emit `input.frame.rejected`; malformed/missing IDs cannot be invented and may be null there. A known pending frame is not destroyed by rejection of a different metadata event.

Ordered fatal paths emit error then one `session.done`: context exhaustion maps to `context_exhausted`; configuration/binary/parked timeout maps to `session_timeout`; other backend faults map to `response_failed`. Session reasons are completed, aborted, context_exhausted or error. No output follows session done.

Finalization requests an internal scheduler-owned acknowledgement after the target's in-flight step is resolved. It does not assume `Client.abort()` alone is a GPU barrier. The backend retains small completed ledgers for up to five minutes, pruned on admission with a 4096-record target bound; no tensors are retained. The leader's TP result is used once, never multiplied by rank count.

Transport/process failure cannot guarantee a terminal event. If authoritative finalization is unavailable, the adapter emits a best-effort error and closes abnormally (1011), without inventing a zero or complete final total. This is not an orderly, successfully settled session. Backend cleanup still runs. Capacity rejection before model admission returns zero usage and close 1013.

## Validation Handoff

The original accuracy suite passes with 1/2/4 sessions, and per-event native/v2 text matches when replaying the same manifest. Additional checks cover CPU regressions, same-turn responses, interruption, sliding windows, accounting and failure cleanup. See [validation results](../../../deployment/vl_api_v2/VALIDATION.md) for scope and supplemental open-description experiments. Passing fixed fixtures does not guarantee exact concurrent output for arbitrary inputs.

```bash
python -m pytest tests/unit_test/serve/test_vl_api_v2.py \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/serve/test_video_realtime_lifecycle.py \
  vl_legacy_adapter/tests/test_lifecycle.py -q
```

The new segment semantics are deliberate, not a transparent in-place upgrade of v1. AGW must use the v2 capability/endpoint and unique response ID for per-response settlement.
