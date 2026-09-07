"""Shared-model Transformers reference with independent round-robin states."""

import gc
import time

from common import CONFIG, events_for, groups, initial_prompt, write_json


def run(model_path, cases, output):
    import torch
    import transformers
    from PIL import Image

    from sglang_omni.models.moss_vl_realtime.model_step import MossVLRealtimeStepper
    from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

    torch.set_num_threads(4)
    processor = transformers.AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    ).eval()
    stepper = MossVLRealtimeStepper(model, processor)
    silence = processor.tokenizer.convert_tokens_to_ids("<|silence|>")
    rows = []

    def drive(state, tokens, limit):
        if state.pending_text_length:
            stepper.commit_pending_tokens(state)
        for i in range(limit):
            token = int(stepper.sample_next_token(state).item())
            tokens.append(token)
            if token == silence:
                break
            if i + 1 < limit:
                stepper.commit_pending_tokens(state)

    for count in CONFIG["session_counts"]:
        for group_index, group in enumerate(groups(cases, count)):
            started = time.monotonic()
            states, tokens, timings = [], [], []
            for case in group:
                encoded = processor.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful visual assistant.",
                        },
                        {"role": "user", "content": initial_prompt(case)},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                ids = (
                    encoded["input_ids"] if hasattr(encoded, "keys") else encoded
                ).to("cuda")
                states.append(stepper.initial_prefill(ids))
                tokens.append([])
                timings.append([])
            torch.cuda.reset_peak_memory_stats()
            traces = [events_for(case, CONFIG["comparison_frames"]) for case in group]
            stream_start = time.monotonic()
            peak_memory = None
            processed = [0] * count
            for tick in range(CONFIG["comparison_frames"] + 1):
                time.sleep(
                    max(0, stream_start + tick / CONFIG["fps"] - time.monotonic())
                )
                for lane, (state, trace) in enumerate(zip(states, traces)):
                    event = trace[tick]
                    frames = []
                    if event["type"] == "frame":
                        with Image.open(event["frame_path"]) as image:
                            frames = [(image.convert("RGB"), event["timestamp"])]
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    stepper.apply_event_and_extend(
                        state, prompt=event.get("prompt"), frames=frames
                    )
                    torch.cuda.synchronize()
                    timings[lane].append(time.monotonic() - begin)
                    processed[lane] += int(event["type"] == "frame")
                    drive(
                        state,
                        tokens[lane],
                        (
                            CONFIG["tf_final_tokens"]
                            if event["final"]
                            else CONFIG["tf_tokens_per_frame"]
                        ),
                    )
                    measured = get_process_gpu_memory_bytes(0)
                    if measured is not None:
                        peak_memory = max(peak_memory or 0, measured)
            for lane, case in enumerate(group):
                text = processor.tokenizer.decode(
                    tokens[lane], skip_special_tokens=True
                )
                rows.append(
                    dict(
                        backend="TF",
                        scheduling="round_robin",
                        phase="comparison",
                        sessions=count,
                        group=group_index,
                        lane=lane,
                        case=case["case_id"],
                        status=(
                            "PASS"
                            if text.strip()
                            and processed[lane] == CONFIG["comparison_frames"]
                            else "FAIL"
                        ),
                        frames=processed[lane],
                        expected_frames=CONFIG["comparison_frames"],
                        elapsed_seconds=time.monotonic() - started,
                        text=text,
                        token_ids=tokens[lane],
                        frame_step_seconds=timings[lane][:-1],
                        process_peak_gib=(
                            None if peak_memory is None else peak_memory / 2**30
                        ),
                        torch_peak_reserved_gib=torch.cuda.max_memory_reserved()
                        / 2**30,
                        kv_peak_gib=None,
                        cleanup="request states released",
                    )
                )
            write_json(output / "tf.json", rows)
            print(f"TF sessions={count} group={group_index} completed", flush=True)
            del states, state
            gc.collect()
            torch.cuda.empty_cache()
    return rows
