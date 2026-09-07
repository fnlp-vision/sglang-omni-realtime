# MOSS-VL Realtime Video Understanding

MOSS-VL Realtime serves one persistent video-understanding session over the
`/v1/video/realtime` WebSocket endpoint. A client appends timestamped image
frames and optional prompts while the same SGLang request remains alive. The
model can emit text, return `<|silence|>` and park, then wake when new input
arrives.

One instance can serve **multiple concurrent sessions**: the cap is
`--max-running-requests` (default 1), kept equal to the WebSocket session limit,
and parked sessions still occupy their slot. Once the cap is reached, a new
connection receives `session_capacity_exceeded` and is closed with code 1013.
The launch examples place all TP ranks on one host.

## Prerequisites

Install this repository following the [installation guide](../get_started/installation.md).
Use the [MOSS-VL-Realtime-SGLANG checkpoint](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime-SGLANG),
which includes Transformers 5.12.1-compatible configuration and processor code:

```bash
uv pip install -e .
hf download OpenMOSS-Team/MOSS-VL-Realtime-SGLANG \
  --local-dir /path/to/MOSS-VL-Realtime-SGLANG
export MODEL_PATH=/path/to/MOSS-VL-Realtime-SGLANG
```

Private checkpoint access requires an authorized Hugging Face account
(`hf auth login`). Keep the full downloaded directory together and pass its
local path with `--model-path`. The original
[MOSS-VL-Realtime](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Realtime)
repository contains the 4.57-series reference implementation; this backend
uses the 5.12.1 compatibility package linked above.

## Start the server

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --gpu 0 \
  --host 127.0.0.1 --port 8000 \
  --context-length 131072 --mem-fraction-static 0.60 \
  --max-running-requests 1
```

For tensor parallel deployment, provide one distinct GPU per rank:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --tp-size 2 \
  --gpus 0,1 \
  --host 127.0.0.1 --port 8000 \
  --context-length 131072 --mem-fraction-static 0.60 \
  --max-running-requests 2
```

`--gpus` must contain exactly `--tp-size` unique GPU ids. Omni starts one
process per rank and exposes only that rank's GPU as local `cuda:0`. This
topology uses NCCL collectives; SGLang custom all-reduce is disabled because
its rendezvous requires distinct visible device ordinals in every rank.

Under TP>1, frame references (for example `shm://`) are resolved by rank 0
only, and the pixels are broadcast to every rank over the TP CPU group; the
single-consumer shared-memory frame lifecycle stays intact. A custom
`frame_resolver` (the scheduler's Python parameter) is likewise executed only
on rank 0 under TP.

The examples explicitly choose 128K context and a 0.60 memory fraction.
The launcher's code defaults are listed below. Size context and concurrency
against the available KV pool; startup reports an error if one full context
cannot fit.

| Setting | Default | Notes |
|---|---:|---|
| Context length | 262144 (256K) | Matches the checkpoint's `max_position_embeddings`; startup fails fast if the KV pool cannot hold one full session context (raise `--mem-fraction-static` or lower `--context-length`) |
| Static memory fraction | 0.40 | Adjust to available device memory together with `--context-length` |
| Concurrent sessions | 1 | `--max-running-requests N`; also the WebSocket session cap; parked sessions still occupy a slot |
| Decode CUDA Graph | On | FlashInfer decode; dynamic frame extend remains eager |
| KV page size | 1 | Fixed; realtime does not patch SGLang's paged allocator |
| Async decode | Off | Optional `--enable-async-decode` |
| Tensor parallelism | 1 | Use `--tp-size N --gpus g0,...,gN-1` |
| Overlap scheduling | Off | Not supported by the realtime update invariants |

Use `--disable-decode-cuda-graph` to run eager decode. When decode Graph is on,
the launcher only accepts the FlashInfer decode backend.

Before Uvicorn starts listening, the launcher opens an internal temporary
request and processes one generated 640 x 352 RGB frame through the normal
realtime update path. Startup only continues after `input.frame.processed`, so
the first external WebSocket does not pay the one-time vision/JIT setup cost.
The warmup request is then aborted and its shared-memory frame is cleaned up.
It is excluded from radix-prefix insertion and does not consume the single
user-session slot after startup, so its internal KV cannot cross into a real
streaming session.
Use `--disable-startup-warmup` only to isolate startup problems; warmup remains
enabled by default.

## Send timestamped frames

The example client defaults to **1 FPS**:

```bash
python examples/moss_vl_realtime_client.py \
  --url ws://127.0.0.1:8000/v1/video/realtime \
  --prompt "Describe relevant changes." \
  --frame /path/to/frame_000.png --timestamp 0.0 \
  --frame /path/to/frame_001.png --timestamp 1.0 \
  --frame /path/to/frame_002.png --timestamp 2.0
```

The example client schedules events with explicit timestamps relative to the
start of replay. `--fps` and `--frame-interval` are mutually exclusive and
provide fallback spacing for events without an explicit timestamp; explicit
timestamps take precedence. The example above therefore sends frames one
second apart. The existing semantic baseline uses 1 FPS.

The explicit timestamp is part of the model input. Each frame is lowered to the
same structure used by offline singleton video segments:

```text
<|vision_start|><|time_start|>1.0 seconds<|time_end|><|image|><|vision_end|>
```

## WebSocket protocol

`session.configure` accepts `prompt` (the initial user prompt), whose default
is an empty string, matching the Transformers realtime reference. Production
deployments are expected to pass the SFT prompt explicitly; the system prompt
default is byte-identical to the reference implementation.

`session.configure` accepts `max_tokens_per_turn`, matching the Transformers
realtime API. Despite its historical name, this is a generation-rate cap in
tokens per second, not a per-turn token-count limit. Its default `86400` is
effectively unlimited. `max_new_tokens` is the decode allowance re-anchored
after every input extend, rather than a lifetime session budget.

Rate limiting happens in the scheduler before model execution. The request
keeps its KV allocation while ordinary decode waits, and the event loop
continues accepting frames and prompt interrupts. Frame/prompt extend always
takes priority over the decode-rate deadline, so a low output rate does not add
the same delay to user input handling. All TP ranks receive the same request
configuration and rank 0 broadcasts scheduler inputs to keep their forward
sequence aligned.

A normal frame follows this sequence:

1. Server sends `session.created`.
2. Client sends `session.configure`.
3. Server sends `session.configured`, then `session.ready` after initial prefill.
4. Client sends `input.frame` JSON metadata.
5. Server sends `input.frame.ready` when the bounded queue has capacity.
6. Client sends the binary image payload.
7. Server sends `input.frame.accepted` after reliable submission.
8. Server sends `input.frame.processed` after incremental forward and KV commit.
9. Text arrives through `response.text.delta`; terminal requests end with
   `response.done` and `session.done`.

`session.configured` also advertises two operational limits:
`max_frame_bytes` (the largest binary frame the server accepts; the transport
is configured to allow it end to end) and `parked_request_timeout_s` (the idle
timeout after which a silence-parked request is aborted and the session ends).

Frame metadata contains `seq_no`, video `timestamp`, optional `prompt`, `final`,
and `mime_type`. Prompt-only updates use `input.prompt` and the same ordered
sequence number space.

Event ordering is validated at the WebSocket edge: `seq_no` must be dense,
starting at 0 for the session, and frame timestamps must be
non-decreasing. A violating event is rejected with a per-event
`invalid_request` error and the session stays alive. Input submission failures
and engine failures terminate the session; explicit abort, disconnect and
parked timeout also end it.

### Context usage and completion

`session.created.capabilities` includes `session.usage`. Set `include_usage`
to `true` in `session.configure` to receive these resource updates:

| Field | Meaning |
| --- | --- |
| `decoder_tokens` | Decoder positions, including a sampled token awaiting its next forward |
| `encoder_tokens` | Historical encoder positions appended to the request |
| `encoder_kv_tokens` | Encoder positions currently retained in the KV cache |
| `token_space_used` | Historical encoder positions plus decoder positions |
| `context_limit` / `context_remaining` | Configured context and remaining positions |

`response.done` and `session.done` end the persistent session. A `turn_id` can
contain several text segments separated by `response.turn.silence`; it is not
a separate ID for every proactive answer. A final input closes the input
stream and lets generation finish. `session.abort` explicitly stops the whole
session. Handle errors and use a completion timeout in client applications.

### Text barge-in and turns

There is no separate turn-interrupt input. Any `input.prompt`, or an
`input.frame` carrying a non-empty `prompt`, atomically closes the current
assistant turn and opens the next user/assistant turn in the same persistent
request. The appended training-format text is:

```text
<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

The scheduler applies the update at the next safe token boundary, retains the
existing text and vision KV, and commits the prompt extend before confirming
the transition. It does not abort the request or recreate the WebSocket.
Already-running CUDA work is cooperative rather than forcibly cancelled.

All events queued at one scheduler step are drained into a single segment,
matching the Transformers reference drain: prompts are spliced first in
arrival order and every prompt is followed by `<|silence|>` when the same
drain cycle also appends frames (the trained assistant-turn opener); frames
follow, sorted by timestamp. Multiple prompts in one cycle therefore transition
through consecutive turn ids, and a frame that races with a prompt is appended
after the prompt text.

The observable event order is:

```text
input.prompt.accepted (or input.frame.accepted)
response.turn.interrupted  turn_id=N, next_turn_id=N+1
input.prompt.processed (or input.frame.processed)  turn_id=N+1
response.text.delta  turn_id=N+1
```

`session.created`, `session.ready`, every text delta, and `response.done` carry
a `turn_id`. Clients should stop rendering or speaking the old turn as soon as
they submit the new prompt, and use the server-confirmed
`response.turn.interrupted` boundary to reject any stale turn output.

### Backpressure

The input queue defaults to 4 events and is configurable from 1 to 256. When it
is full, the server delays `input.frame.ready`; a conforming client keeps the
next frame on the producer side until capacity is released. The server does not
silently drop frames and does not create an unbounded overflow queue.

If a live camera permanently produces faster than the model consumes, the
producer must slow down or provide its own bounded buffering/storage policy.
Finite memory, constant capture FPS, and unlimited no-drop retention cannot all
be guaranteed simultaneously.

## Concurrent sessions and KV pressure

With `--max-running-requests N` (N > 1) the instance serves N concurrent
sessions, each an independent persistent request; parked sessions still occupy
their slot, so the WebSocket session cap is kept equal to N. Sessions do not
share prefix KV (radix insertion is skipped for realtime requests), so each
session pays its own prefill.

Capacity ordering is:

- **Admission**: when all slots are busy, a new WebSocket receives
  `error(session_capacity_exceeded)` and is closed with code 1013.
- **Startup precheck**: boot fails fast when the KV pool cannot hold one full
  session context; oversubscribing sessions (`N * context_len > pool`) logs a
  warning, because the runtime then relies on the degradation below.
- **Runtime degradation**: when the KV pool cannot fit the next extend/decode,
  the scheduler aborts the currently heaviest session first (its client
  receives `error(response_failed)` and a closed connection) until the batch
  fits again. A session aborted this way must reconnect as a new session and
  re-push its stream from scratch.

Under concurrent load the decode-rate cap (`max_tokens_per_turn`) becomes a
soft target: a step runs as soon as any session in the batch is due, letting
not-yet-due sessions ride along.

## Benchmark mode

`benchmark_ignore_eos` exists only to collect a fixed number of decode tokens.
Production servers reject it. A benchmark server must be started explicitly:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path "$MODEL_PATH" \
  --context-length 131072 --mem-fraction-static 0.60 \
  --enable-benchmark-mode
```

Do not enable benchmark mode for production traffic.

## Vision KV sliding window (opt-in)

The window is disabled by default. To retain recent raw frames and evict older
visual KV, set these variables before starting the server:

```bash
export REALTIME_FRAME_WINDOW_ENABLED=1
export REALTIME_FRAME_WINDOW_RAW_S=60
export REALTIME_FRAME_POOLING_ENABLED=0
```

Pooling is a separate experimental option, also disabled by default. Setting
`REALTIME_FRAME_POOLING_ENABLED=1` allows full groups of aged raw frames to
become mean-pooled virtual frames. `REALTIME_FRAME_POOL_RATIO` controls group
size, and `REALTIME_FRAME_POOL_WINDOW_S` controls virtual-frame retention.

Window behavior:

- **Visual context:** The raw-only window removes aged visual context. Pooling additionally averages
  RoPE-rotated keys and values from different frames. Both change the visual
  history available to attention and can change the output.
- **Context accounting:** Token space keeps one pad placeholder per historical
  encoder slot. The context guard therefore still counts these positions after
  physical KV slots have been reclaimed.
- **Pooling under pressure:** If a pooled copy cannot be allocated, the round
  evicts aged frames directly and logs "degraded to eviction".

## Validation

Run the model-specific and serving tests:

```bash
python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/client/test_control_event.py
```

Use GPU model-step and processor tests for reference comparisons, and fix the
model revision, prompts, timestamps and token boundaries when comparing eager,
CUDA Graph and async decode. Live input arrival can place new frames at
different generation boundaries.

The 2026-09-06 CPU regression run on `e1b5fcf` passed 268 tests with five skips
when also including `tests/unit_test/pipeline/test_async_decode.py` (without
the extra client tests in the command above). Four skips required CUDA and one
required a model path. Existing GPU checks cover single-device and TP2 paths.

An earlier accelerated mixed long/short-session workload produced short-session
completion timeouts after all inputs had been processed. Recheck this workload
when selecting deployment concurrency, and distinguish an input being processed
from the entire session completing.
