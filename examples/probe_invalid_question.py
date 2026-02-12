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
from dataclasses import dataclass
from dataclasses import field
import gc
import json
import pathlib

import ares
from ares.contrib.mech_interp import ActivationCapture
from ares.contrib.mech_interp.hooked_transformer_client import HookedTransformerLLMClient
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DEVICE = "mps"  # Apple Silicon GPU. Use "cpu" if you hit MPS issues.
MAX_NEW_TOKENS = 256

ENV_NAME = "20q"
N_EPISODES = 25
MAX_STEPS_PER_EPISODE = 25  # >20 so episodes can reach the natural limit.

# Minimum samples at a given step to attempt a probe.
MIN_SAMPLES_FOR_PROBE = 8

OUTPUT_DIR = pathlib.Path("outputs/20q_invalid_question_probing")

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
# Data collection
# ---------------------------------------------------------------------------


async def collect_episodes(
    model: HookedTransformer,
    client: HookedTransformerLLMClient,
    n_episodes: int,
    max_steps_per_episode: int,
    middle_layer: int,
) -> tuple[list[StepSample], list[EpisodeRecord]]:
    """Run episodes and collect per-step activations with oracle-invalidity labels."""
    all_samples: list[StepSample] = []
    episode_records: list[EpisodeRecord] = []

    for ep in tqdm(range(n_episodes), desc="Episodes", unit="ep"):
        async with ares.make(ENV_NAME) as env:
            ts = await env.reset()

            step_idx = 0
            n_invalid = 0

            with ActivationCapture(model) as capture:
                while (not ts.last()) and (step_idx < max_steps_per_episode):
                    capture.start_step()
                    assert ts.observation is not None
                    action = await client(ts.observation)
                    capture.end_step()

                    # Capture the activation before stepping the environment
                    hook_name = f"blocks.{middle_layer}.hook_resid_post"
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
            print(
                f"  episode={ep:03d}  steps={step_idx:2d}  invalid={n_invalid:2d} ({invalid_pct:.0f}%)  "
                f"reward={final_reward:+.1f}  success={label_success}"
            )

    return all_samples, episode_records


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
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


async def run() -> ProbeResult:
    print(f"\n{'=' * 70}")
    print(f"MODEL: {MODEL_NAME}")
    print(f"{'=' * 70}")

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
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
    print(f"n_layers={n_layers}, d_model={d_model}, middle_layer={middle_layer}")

    print(f"\nCollecting {N_EPISODES} episodes...")
    samples, episode_records = await collect_episodes(
        model=model,
        client=client,
        n_episodes=N_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        middle_layer=middle_layer,
    )
    print(f"\nCollected {len(samples)} step samples from {len(episode_records)} episodes")

    result = train_probes(
        samples=samples,
        model_name=MODEL_NAME,
        n_layers=n_layers,
        d_model=d_model,
        middle_layer=middle_layer,
        n_episodes=len(episode_records),
    )

    del model, tokenizer, client
    _free_gpu_memory()

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
