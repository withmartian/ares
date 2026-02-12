#!/usr/bin/env python

# # Probing for "Invalid Question" in Twenty Questions
#
# This script trains a linear probe on middle-layer residual stream activations
# to predict whether the oracle will respond with "Invalid Question." at each
# step of a Twenty Questions episode.
#
# Unlike the episode-success probe (mech_interp_20q_multimodel_probing.py),
# this probe has a label for *every* step, giving substantially more training
# data per episode.
#
# ## Requirements
#   uv sync --group transformer-lens   # installs transformer-lens + scikit-learn
#   # CHAT_COMPLETION_API_KEY must be set in .env (for the oracle)
#
# ## Run
#   uv run -m examples.probe_invalid_question

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
import gc
import json
import pathlib
import queue

import ares
from ares.contrib.mech_interp import ActivationCapture
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

MAX_NEW_TOKENS = 64  # 20Q questions are short; 256 was needlessly slow.

ENV_NAME = "20q"
N_EPISODES = 50
MAX_STEPS_PER_EPISODE = 25  # >20 so episodes can reach the natural limit.

# Minimum samples at a given step to attempt a probe.
MIN_SAMPLES_FOR_PROBE = 8

OUTPUT_DIR = pathlib.Path("outputs/20q_invalid_question_probing_50episodes")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepSample:
    """A single step's activation and oracle-invalidity label."""

    feature: np.ndarray  # shape [d_model]
    is_invalid: int  # 1 = oracle responded "Invalid Question.", 0 = valid (Yes/No)
    episode_idx: int
    step_idx: int


@dataclass
class EpisodeRecord:
    """Summary of one episode for logging."""

    episode_idx: int
    n_steps: int
    n_invalid: int
    label_success: int


@dataclass
class ProbeResult:
    """Probe results for one model."""

    model_name: str
    n_layers: int
    d_model: int
    middle_layer: int
    n_episodes: int
    n_total_steps: int
    n_invalid_steps: int
    invalid_rate: float
    # Per-step probe accuracy: step_idx -> accuracy
    step_accuracies: dict[int, float] = field(default_factory=dict)
    step_sample_counts: dict[int, int] = field(default_factory=dict)
    # Global (pooled) probe metrics
    global_accuracy: float = 0.0
    global_report: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_reward(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])


def _to_numpy_mean_over_tokens(tensor_like) -> np.ndarray:
    arr = tensor_like.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 resid_post, got shape={arr.shape}")
    return arr[0].mean(axis=0)


def _is_invalid_answer(oracle_answer: str) -> bool:
    """Check if the oracle's answer is 'Invalid Question.'."""
    return "invalid question" in oracle_answer.lower()


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------


def _get_devices() -> list[str]:
    """Detect available CUDA GPUs, falling back to CPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return ["cpu"]


# ---------------------------------------------------------------------------
# Data collection (per-device worker)
# ---------------------------------------------------------------------------


async def _collect_episodes_for_device(
    device: str,
    work_queue: queue.Queue[int],
    max_steps_per_episode: int,
) -> tuple[list[StepSample], list[EpisodeRecord], int, int, int]:
    """Load model on *device* and pull episodes from *work_queue* until empty.

    Returns (samples, records, n_layers, d_model, middle_layer).
    """
    print(f"[{device}] Loading {MODEL_NAME}...")
    torch.cuda.set_device(device)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
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
    print(f"[{device}] n_layers={n_layers}, d_model={d_model}, middle_layer={middle_layer}")

    all_samples: list[StepSample] = []
    episode_records: list[EpisodeRecord] = []

    while True:
        try:
            ep = work_queue.get_nowait()
        except queue.Empty:
            break
        async with ares.make(ENV_NAME) as env:
            ts = await env.reset()

            step_idx = 0
            n_invalid = 0

            hook_name = f"blocks.{middle_layer}.hook_resid_post"
            with ActivationCapture(model, hook_filter=lambda name: name == hook_name) as capture:
                while (not ts.last()) and (step_idx < max_steps_per_episode):
                    capture.start_step()
                    assert ts.observation is not None
                    action = await client(ts.observation)
                    capture.end_step()

                    resid_post = capture.get_trajectory().get_activation(step_idx, hook_name)
                    feature = _to_numpy_mean_over_tokens(resid_post)

                    # Record conversation history length before step
                    prev_history_len = len(env._conversation_history)

                    # Step the environment (this calls the oracle internally)
                    ts = await env.step(action)

                    # Extract oracle answer from the conversation history
                    oracle_answer = ""
                    if len(env._conversation_history) > prev_history_len:
                        last_entry = env._conversation_history[-1]
                        if last_entry.startswith("A:"):
                            oracle_answer = last_entry

                    is_invalid = int(_is_invalid_answer(oracle_answer))
                    n_invalid += is_invalid

                    all_samples.append(
                        StepSample(
                            feature=feature,
                            is_invalid=is_invalid,
                            episode_idx=ep,
                            step_idx=step_idx,
                        )
                    )

                    step_idx += 1
                    tqdm.write(
                        f"    [{device}] ep={ep} step={step_idx}/{max_steps_per_episode}  oracle={oracle_answer[:40]}"
                    )

            final_reward = _scalar_reward(ts.reward)
            label_success = int(final_reward == 0.0 and ts.last())

            episode_records.append(
                EpisodeRecord(
                    episode_idx=ep,
                    n_steps=step_idx,
                    n_invalid=n_invalid,
                    label_success=label_success,
                )
            )

            invalid_pct = (n_invalid / step_idx * 100) if step_idx > 0 else 0.0
            tqdm.write(
                f"  [{device}] episode={ep:03d}  steps={step_idx:2d}  invalid={n_invalid:2d} ({invalid_pct:.0f}%)  "
                f"reward={final_reward:+.1f}  success={label_success}"
            )

    del model, tokenizer, client
    _free_gpu_memory()

    return all_samples, episode_records, n_layers, d_model, middle_layer


def _run_device_worker(
    device: str,
    work_queue: queue.Queue[int],
    max_steps_per_episode: int,
) -> tuple[list[StepSample], list[EpisodeRecord], int, int, int]:
    """Run episode collection in a dedicated event loop (for thread-based parallelism)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_collect_episodes_for_device(device, work_queue, max_steps_per_episode))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _cv_accuracy(x_data: np.ndarray, y: np.ndarray) -> float:
    """3-fold stratified cross-validated accuracy of a logistic regression probe."""
    n_folds = min(3, int(y.sum()), int((1 - y).sum()))
    if n_folds < 2:
        probe = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=SEED)
        probe.fit(x_data, y)
        return float(np.mean(probe.predict(x_data) == y))

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    accs = []
    for train_idx, test_idx in cv.split(x_data, y):
        probe = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=SEED)
        probe.fit(x_data[train_idx], y[train_idx])
        accs.append(float(np.mean(probe.predict(x_data[test_idx]) == y[test_idx])))
    return float(np.mean(accs))


def train_probes(
    samples: list[StepSample],
    model_name: str,
    n_layers: int,
    d_model: int,
    middle_layer: int,
    n_episodes: int,
) -> ProbeResult:
    """Train per-step and global probes to predict oracle invalidity."""
    n_total = len(samples)
    n_invalid = sum(s.is_invalid for s in samples)
    invalid_rate = n_invalid / n_total if n_total else 0.0

    print(f"\n  Total steps: {n_total},  Invalid: {n_invalid}/{n_total} ({invalid_rate:.1%})")

    result = ProbeResult(
        model_name=model_name,
        n_layers=n_layers,
        d_model=d_model,
        middle_layer=middle_layer,
        n_episodes=n_episodes,
        n_total_steps=n_total,
        n_invalid_steps=n_invalid,
        invalid_rate=invalid_rate,
    )

    # --- Per-step probes ---
    max_step = max(s.step_idx for s in samples) + 1
    print(f"  Max step index: {max_step - 1}")

    for step_idx in range(max_step):
        step_samples = [s for s in samples if s.step_idx == step_idx]
        result.step_sample_counts[step_idx] = len(step_samples)

        if len(step_samples) < MIN_SAMPLES_FOR_PROBE:
            print(f"  step {step_idx:2d}: n={len(step_samples):3d}  -- skipped (too few samples)")
            continue

        x_arr = np.stack([s.feature for s in step_samples])
        y_arr = np.array([s.is_invalid for s in step_samples], dtype=np.int64)
        n_pos = int(y_arr.sum())
        n_neg = len(y_arr) - n_pos

        if n_pos == 0 or n_neg == 0:
            print(f"  step {step_idx:2d}: n={len(y_arr):3d} (invalid={n_pos})  -- skipped (no class diversity)")
            continue

        acc = _cv_accuracy(x_arr, y_arr)
        result.step_accuracies[step_idx] = acc
        print(f"  step {step_idx:2d}: n={len(y_arr):3d} (invalid={n_pos})  acc={acc:.3f}")

    # --- Global probe (pooled across all steps) ---
    x_all = np.stack([s.feature for s in samples])
    y_all = np.array([s.is_invalid for s in samples], dtype=np.int64)
    n_pos_all = int(y_all.sum())
    n_neg_all = len(y_all) - n_pos_all

    if n_pos_all >= 2 and n_neg_all >= 2:
        result.global_accuracy = _cv_accuracy(x_all, y_all)

        # Train a final probe on all data for the classification report
        probe = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=SEED)
        probe.fit(x_all, y_all)
        y_pred = probe.predict(x_all)
        result.global_report = classification_report(
            y_all, y_pred, target_names=["Valid (Yes/No)", "Invalid Question"], zero_division=0
        )

        print(f"\n  Global probe CV accuracy: {result.global_accuracy:.3f}")
        print(f"\n  Classification report (train set):\n{result.global_report}")
    else:
        print("\n  Global probe: skipped (insufficient class diversity)")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results: list[ProbeResult], output_path: pathlib.Path) -> None:
    """Plot per-step probe accuracy and global accuracy for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Left: per-step accuracy
    ax = axes[0]
    for i, r in enumerate(all_results):
        short = r.model_name.split("/")[-1]
        steps = sorted(r.step_accuracies.keys())
        if not steps:
            continue
        accs = [r.step_accuracies[s] for s in steps]
        ax.plot(
            steps,
            accs,
            marker="o",
            markersize=4,
            label=f"{short} ({r.invalid_rate:.0%} invalid)",
            color=colors[i % len(colors)],
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Step (question number)")
    ax.set_ylabel("Probe accuracy (CV)")
    ax.set_title("Can residual stream predict 'Invalid Question' per step?")
    ax.legend()
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

    # Right: global accuracy bar chart
    ax2 = axes[1]
    names = [r.model_name.split("/")[-1] for r in all_results]
    global_accs = [r.global_accuracy for r in all_results]
    invalid_rates = [r.invalid_rate for r in all_results]

    x_pos = np.arange(len(names))
    bars = ax2.bar(x_pos, global_accs, color=[colors[i % len(colors)] for i in range(len(names))], alpha=0.8)
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    # Annotate with invalid rate
    for bar, rate in zip(bars, invalid_rates, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.0%} inv",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=15, ha="right")
    ax2.set_ylabel("Global probe accuracy (CV)")
    ax2.set_title("Pooled probe: predict Invalid Question")
    ax2.set_ylim(0.35, 1.05)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run() -> ProbeResult:
    devices = _get_devices()
    # Don't use more GPUs than episodes.
    devices = devices[:N_EPISODES]
    n_devices = len(devices)

    print(f"\n{'=' * 70}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"DEVICES: {devices}")
    print(f"{'=' * 70}")

    work_queue: queue.Queue[int] = queue.Queue()
    for ep_idx in range(N_EPISODES):
        work_queue.put(ep_idx)

    print(f"\nCollecting {N_EPISODES} episodes across {n_devices} device(s) (work-queue)...")

    # Each device gets its own thread with a private event loop, model, and
    # activation capture state.  CUDA ops release the GIL, so the GPUs run
    # truly in parallel.  Workers pull episodes from a shared queue so that
    # fast-finishing GPUs pick up more work automatically.
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _run_device_worker,
                device,
                work_queue,
                MAX_STEPS_PER_EPISODE,
            )
            for device in devices
        ]
        worker_results = await asyncio.gather(*futures)

    # Merge results from all workers.
    all_samples: list[StepSample] = []
    all_records: list[EpisodeRecord] = []
    for samples, records, *_ in worker_results:
        all_samples.extend(samples)
        all_records.extend(records)

    # Model config is identical across workers; take from the first.
    _, _, n_layers, d_model, middle_layer = worker_results[0]

    print(f"\nCollected {len(all_samples)} step samples from {len(all_records)} episodes across {n_devices} device(s)")

    result = train_probes(
        samples=all_samples,
        model_name=MODEL_NAME,
        n_layers=n_layers,
        d_model=d_model,
        middle_layer=middle_layer,
        n_episodes=len(all_records),
    )

    return result


def save_results_json(results: list[ProbeResult], path: pathlib.Path) -> None:
    """Persist numeric results so we can re-plot without re-running."""
    data = []
    for r in results:
        data.append(
            {
                "model_name": r.model_name,
                "n_layers": r.n_layers,
                "d_model": r.d_model,
                "middle_layer": r.middle_layer,
                "n_episodes": r.n_episodes,
                "n_total_steps": r.n_total_steps,
                "n_invalid_steps": r.n_invalid_steps,
                "invalid_rate": r.invalid_rate,
                "global_accuracy": r.global_accuracy,
                "step_accuracies": {str(k): v for k, v in r.step_accuracies.items()},
                "step_sample_counts": {str(k): v for k, v in r.step_sample_counts.items()},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Results JSON saved to {path}")


def print_summary(results: list[ProbeResult]) -> None:
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    header = f"{'Model':<35} {'Layers':>6} {'d_model':>7} {'Steps':>6} {'Inv%':>6} {'GlobalAcc':>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.model_name:<35} {r.n_layers:>6} {r.d_model:>7} "
            f"{r.n_total_steps:>6} {r.invalid_rate:>5.1%} {r.global_accuracy:>9.4f}"
        )
    print()


def main() -> None:
    result = asyncio.run(run())
    results = [result]
    print_summary(results)
    save_results_json(results, OUTPUT_DIR / "results.json")
    plot_results(results, OUTPUT_DIR / "probe_invalid_question_vs_step.png")


if __name__ == "__main__":
    main()
