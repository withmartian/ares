#!/usr/bin/env python

# # Collect 20Q Episode Data with Activations
#
# Runs Twenty Questions episodes using a HookedTransformer model across all
# available GPUs, captures middle-layer residual stream activations, and saves
# everything to disk for downstream probing / steering / early-detection
# experiments.
#
# ## Requirements
#   uv sync --group transformer-lens
#   # CHAT_COMPLETION_API_KEY must be set in .env (for the oracle)
#
# ## Run
#   uv run --no-sync python examples/20q_case_study/collect_20q_data.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import pathlib
import queue
import threading

import ares
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 64
ENV_NAME = "20q"
N_EPISODES = 50
MAX_STEPS_PER_EPISODE = 25
OUTPUT_DIR = pathlib.Path("outputs/20q_data")

# Serialize HookedTransformer.from_pretrained across threads to avoid the
# meta-tensor race condition where 7/8 GPUs get uninitialized weights.
_MODEL_LOAD_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_devices() -> list[str]:
    """Detect available CUDA GPUs, falling back to CPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


def _is_invalid_answer(oracle_answer: str) -> bool:
    return "invalid question" in oracle_answer.lower()


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-device worker
# ---------------------------------------------------------------------------


async def _collect_episodes_for_device(
    device: str,
    work_queue: queue.Queue[int],
    max_steps_per_episode: int,
) -> tuple[list[dict], int, int, int]:
    """Load model on *device*, pull episodes from *work_queue*, save each to disk.

    Returns (episode_summaries, n_layers, d_model, middle_layer).
    """
    print(f"[{device}] Loading {MODEL_NAME}...")
    torch.cuda.set_device(device)

    with _MODEL_LOAD_LOCK:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.bfloat16)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    client = create_hooked_transformer_client_with_chat_template(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=False,
    )

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    middle_layer = n_layers // 2
    hook_name = f"blocks.{middle_layer}.hook_resid_post"
    print(f"[{device}] n_layers={n_layers}, d_model={d_model}, middle_layer={middle_layer}")

    episode_summaries: list[dict] = []

    # Simple dict-based activation capture: a hook writes to this dict, we
    # read from it after each forward pass.  Avoids the overhead of
    # ActivationCapture (redundant GPU->CPU copies, O(n^2) trajectory copies).
    captured: dict[str, torch.Tensor] = {}

    def _capture_hook(activation: torch.Tensor, hook) -> torch.Tensor:  # noqa: ARG001
        captured["resid_post"] = activation.detach().cpu().float()
        return activation

    # Register the hook once for the whole worker lifetime.
    hook_point = model.hook_dict[hook_name]
    hook_point.add_hook(_capture_hook)

    try:
        while True:
            try:
                ep = work_queue.get_nowait()
            except queue.Empty:
                break

            steps: list[dict] = []
            async with ares.make(ENV_NAME) as env:
                ts = await env.reset()
                step_idx = 0
                n_invalid = 0

                while (not ts.last()) and (step_idx < max_steps_per_episode):
                    captured.clear()

                    assert ts.observation is not None
                    action = await client(ts.observation)

                    activation = captured.get("resid_post")

                    # Record conversation history length before stepping.
                    prev_history_len = len(env._conversation_history)

                    ts = await env.step(action)

                    # Extract oracle answer from conversation history.
                    oracle_answer = ""
                    if len(env._conversation_history) > prev_history_len:
                        last_entry = env._conversation_history[-1]
                        if last_entry.startswith("A:"):
                            oracle_answer = last_entry

                    is_invalid = int(_is_invalid_answer(oracle_answer))
                    n_invalid += is_invalid

                    step_record: dict = {
                        "step_idx": step_idx,
                        "is_invalid": is_invalid,
                        "oracle_response": oracle_answer,
                    }
                    if activation is not None:
                        step_record["activation"] = activation  # [1, seq_len, d_model]

                    steps.append(step_record)
                    step_idx += 1
                    tqdm.write(
                        f"    [{device}] ep={ep} step={step_idx}/{max_steps_per_episode}  oracle={oracle_answer[:60]}"
                    )

            # Save episode to disk.
            episode_data = {
                "episode_idx": ep,
                "steps": steps,
            }
            ep_path = OUTPUT_DIR / f"episode_{ep:04d}.pt"
            torch.save(episode_data, ep_path)

            summary = {"episode_idx": ep, "n_steps": step_idx, "n_invalid": n_invalid}
            episode_summaries.append(summary)

            invalid_pct = (n_invalid / step_idx * 100) if step_idx > 0 else 0.0
            tqdm.write(
                f"  [{device}] episode={ep:03d}  steps={step_idx:2d}  invalid={n_invalid:2d} ({invalid_pct:.0f}%)"
            )

    finally:
        hook_point.remove_hooks("fwd")

    del model, tokenizer, client
    _free_gpu_memory()

    return episode_summaries, n_layers, d_model, middle_layer


def _run_device_worker(
    device: str,
    work_queue: queue.Queue[int],
    max_steps_per_episode: int,
) -> tuple[list[dict], int, int, int]:
    """Run episode collection in a dedicated event loop (for thread-based parallelism)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_collect_episodes_for_device(device, work_queue, max_steps_per_episode))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    devices = _get_devices()
    devices = devices[:N_EPISODES]  # Don't use more GPUs than episodes.
    n_devices = len(devices)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"DEVICES: {devices}")
    print(f"N_EPISODES: {N_EPISODES}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    work_queue: queue.Queue[int] = queue.Queue()
    for ep_idx in range(N_EPISODES):
        work_queue.put(ep_idx)

    print(f"\nCollecting {N_EPISODES} episodes across {n_devices} device(s)...")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = [
            loop.run_in_executor(executor, _run_device_worker, device, work_queue, MAX_STEPS_PER_EPISODE)
            for device in devices
        ]
        worker_results = await asyncio.gather(*futures, return_exceptions=True)

    # Separate successes from failures.
    all_summaries: list[dict] = []
    n_layers = d_model = middle_layer = 0
    failed_devices: list[str] = []

    for device, result in zip(devices, worker_results, strict=True):
        if isinstance(result, BaseException):
            print(f"\n[{device}] FAILED: {result}")
            failed_devices.append(device)
            continue
        summaries, nl, dm, ml = result
        all_summaries.extend(summaries)
        n_layers, d_model, middle_layer = nl, dm, ml

    # Write metadata.
    metadata = {
        "model_name": MODEL_NAME,
        "n_layers": n_layers,
        "d_model": d_model,
        "middle_layer": middle_layer,
        "n_episodes": len(all_summaries),
        "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
        "n_devices": n_devices,
        "failed_devices": failed_devices,
        "episodes": all_summaries,
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    total_steps = sum(s["n_steps"] for s in all_summaries)
    total_invalid = sum(s["n_invalid"] for s in all_summaries)
    print(f"\nDone. {len(all_summaries)} episodes, {total_steps} steps, {total_invalid} invalid.")
    print(f"Saved to {OUTPUT_DIR}/")
    if failed_devices:
        print(f"WARNING: {len(failed_devices)} device(s) failed: {failed_devices}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
