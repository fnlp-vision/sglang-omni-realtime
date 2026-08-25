# MOSS-VL Realtime Video Understanding

MOSS-VL Realtime serves one persistent video-understanding session over the
`/v1/video/realtime` WebSocket endpoint. A client appends timestamped image
frames and optional prompts while the same SGLang request remains alive. The
model can emit text, return `<|silence|>` and park, then wake when new input
arrives.

This integration currently targets **one WebSocket, one video stream, one
model, and one active session** on a single host. Multi-session scheduling and
cross-host media relay are not supported.

## Prerequisites

Install this repository and use a MOSS-VL streaming checkpoint whose processor
and model code implement the realtime timestamp, incremental vision KV, and
silence semantics:

```bash
uv pip install -e .
```

The validated environment uses SGLang 0.5.16 and Transformers 5.12.1. Keep the
checkpoint outside the source tree and pass its path with `--model-path`.
The project acceptance checkpoint is:

```text
/inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/mossvl_streaming_tf_5.12.1
```

Do not substitute `MOSS-VL-0708-Instruct-sglang` for realtime semantic
validation. That checkpoint can load through the same SGLang model class, but
on the 1 FPS SB-Pro contract it produces ordinary video descriptions instead
of the trained `<|silence|>` / event-trigger behavior. A successful model load
therefore proves framework compatibility, not streaming-model correctness.

## Start the server

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path /inspire/qb-ilm/project/video-understanding/public/train/moss_vl_streaming/8B/final_release/mossvl_streaming_tf_5.12.1 \
  --gpu 0 \
  --host 0.0.0.0 \
  --port 8000
```

The default server configuration is the validated single-stream setup:

| Setting | Default | Notes |
|---|---:|---|
| Context length | 131072 (128K) | Shared by the launcher, pipeline config, stage factory, and engine builder |
| Static memory fraction | 0.40 | Leaves enough H200 KV capacity for the 128K request row |
| Decode CUDA Graph | On | FlashInfer decode; dynamic frame extend remains eager |
| KV page size | 1 | Fixed; realtime does not patch SGLang's paged allocator |
| Async decode | Off | Optional `--enable-async-decode` |
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

`--fps` and `--frame-interval` expose producer pacing controls and are mutually
exclusive. The model was trained and semantically validated at **1 FPS**. Other
positive FPS values are accepted by the interface, but their answer quality is
not qualified and should not be treated as a framework correctness criterion.
No higher-FPS model-quality experiments are required for acceptance.

The explicit timestamp is part of the model input. Each frame is lowered to the
same structure used by offline singleton video segments:

```text
<|vision_start|><|time_start|>1.0 seconds<|time_end|><|image|><|vision_end|>
```

## WebSocket protocol

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

Frame metadata contains `seq_no`, video `timestamp`, optional `prompt`, `final`,
and `mime_type`. Prompt-only updates use `input.prompt` and the same ordered
sequence number space.

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

## Benchmark mode

`benchmark_ignore_eos` exists only to collect a fixed number of decode tokens.
Production servers reject it. A benchmark server must be started explicitly:

```bash
python examples/run_moss_vl_realtime_server.py \
  --model-path /path/to/moss-vl-realtime-checkpoint \
  --enable-benchmark-mode
```

Do not enable benchmark mode for production traffic.

## Validation

Run the model-specific and serving tests:

```bash
python -m pytest -q \
  tests/unit_test/moss_vl_realtime \
  tests/unit_test/serve/test_video_realtime.py \
  tests/unit_test/client/test_control_event.py
```

The semantic acceptance baseline uses real 1 FPS streams. Decode Graph, eager
decode, and optional async decode must preserve greedy output against
that baseline. Higher FPS is an exposed producer-control capability, not a
model-accuracy acceptance matrix.
