#!/usr/bin/env python

# # Multi-Model Per-Step Linear Probe on Twenty Questions
#
# For each of three Qwen2.5-Instruct sizes (0.5B, 1.5B, 3B), this script:
#   1. Runs N episodes on the Twenty Questions environment
#   2. At every step, captures the middle-layer residual stream activation
#   3. After all episodes, trains a separate logistic regression probe at each
#      step index to predict whether the episode will end in success
#   4. Plots probe accuracy vs step for all three models on one graph
#
# ## Requirements
#   uv sync --group transformer-lens   # installs transformer-lens + scikit-learn
#   # CHAT_COMPLETION_API_KEY must be set in .env (for the gpt-4o-mini oracle)
#
# ## Run
#   uv run -m examples.mech_interp_20q_multimodel_probing

import asyncio
from dataclasses import dataclass
from dataclasses import field
import gc
import json
import pathlib

import ares
from ares.contrib.mech_interp import ActivationCapture
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
np.random.seed(SEED)

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

DEVICE = "mps"  # Apple Silicon GPU. Use "cpu" if you hit MPS issues.
MAX_NEW_TOKENS = 128

ENV_NAME = "20q"
N_EPISODES = 25
MAX_STEPS_PER_EPISODE = 25  # >20 so episodes can reach the natural limit.

# Minimum episodes at a given step to attempt a probe.
MIN_EPISODES_FOR_PROBE = 8

OUTPUT_DIR = pathlib.Path("outputs/20q_probing")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeData:
    """Per-step middle-layer activations and outcome label for one episode."""

    step_features: list[np.ndarray]  # step_features[t] has shape [d_model]
    label_success: int  # 1 = agent guessed correctly, 0 = failed
    episode_idx: int
    n_steps: int


@dataclass
class PerStepProbeResult:
    """Probe accuracy at each step for one model."""

    model_name: str
    n_layers: int
    d_model: int
    middle_layer: int
    n_episodes: int
    n_successes: int
    success_rate: float
    # step -> (accuracy, n_samples, n_pos)
    step_accuracies: dict[int, float] = field(default_factory=dict)
    step_sample_counts: dict[int, int] = field(default_factory=dict)


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


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


async def collect_episodes(
    model: HookedTransformer,
    client,
    n_episodes: int,
    max_steps_per_episode: int,
    middle_layer: int,
) -> list[EpisodeData]:
    """Run episodes and store per-step middle-layer activations."""
    episodes: list[EpisodeData] = []

    for ep in range(n_episodes):
        async with ares.make(ENV_NAME) as env:
            ts = await env.reset()

            step_features: list[np.ndarray] = []
            n_steps = 0

            with ActivationCapture(model) as capture:
                while (not ts.last()) and (n_steps < max_steps_per_episode):
                    capture.start_step()
                    assert ts.observation is not None
                    action = await client(ts.observation)
                    capture.end_step()

                    hook_name = f"blocks.{middle_layer}.hook_resid_post"
                    resid_post = capture.get_trajectory().get_activation(n_steps, hook_name)
                    step_features.append(_to_numpy_mean_over_tokens(resid_post))

                    ts = await env.step(action)
                    n_steps += 1

            final_reward = _scalar_reward(ts.reward)
            # In 20q, reward == 0.0 at a terminal step means the agent guessed correctly.
            label = int(final_reward == 0.0 and ts.last())

            if not step_features:
                continue

            episodes.append(
                EpisodeData(
                    step_features=step_features,
                    label_success=label,
                    episode_idx=ep,
                    n_steps=n_steps,
                )
            )

            print(f"  episode={ep:03d}  steps={n_steps:2d}  reward={final_reward:+.1f}  success={label}")

    return episodes


# ---------------------------------------------------------------------------
# Per-step probing
# ---------------------------------------------------------------------------


def _cv_accuracy(x_data: np.ndarray, y: np.ndarray) -> float:
    """3-fold stratified cross-validated accuracy of a logistic regression probe."""
    n_folds = min(3, int(y.sum()), int((1 - y).sum()))
    if n_folds < 2:
        # Not enough of one class for CV — train on all and report train accuracy.
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


def train_per_step_probes(
    episodes: list[EpisodeData],
    model_name: str,
    n_layers: int,
    d_model: int,
    middle_layer: int,
) -> PerStepProbeResult:
    """Train a probe at each step index and return per-step accuracy."""
    n_episodes = len(episodes)
    n_successes = sum(e.label_success for e in episodes)
    success_rate = n_successes / n_episodes if n_episodes else 0.0

    max_step = max(e.n_steps for e in episodes)
    print(f"\n  Episodes: {n_episodes},  Successes: {n_successes}/{n_episodes} ({success_rate:.1%})")
    print(f"  Max steps in any episode: {max_step}")

    result = PerStepProbeResult(
        model_name=model_name,
        n_layers=n_layers,
        d_model=d_model,
        middle_layer=middle_layer,
        n_episodes=n_episodes,
        n_successes=n_successes,
        success_rate=success_rate,
    )

    for step_idx in range(max_step):
        # Gather episodes that have at least step_idx+1 steps.
        xs = []
        ys = []
        for ep in episodes:
            if step_idx < ep.n_steps:
                xs.append(ep.step_features[step_idx])
                ys.append(ep.label_success)

        x_arr = np.stack(xs)
        y_arr = np.array(ys, dtype=np.int64)
        n_pos = int(y_arr.sum())
        n_neg = len(y_arr) - n_pos

        result.step_sample_counts[step_idx] = len(y_arr)

        if len(y_arr) < MIN_EPISODES_FOR_PROBE or n_pos == 0 or n_neg == 0:
            print(f"  step {step_idx:2d}: n={len(y_arr):3d} (pos={n_pos})  — skipped (insufficient diversity)")
            continue

        acc = _cv_accuracy(x_arr, y_arr)
        result.step_accuracies[step_idx] = acc
        print(f"  step {step_idx:2d}: n={len(y_arr):3d} (pos={n_pos})  acc={acc:.3f}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(all_results: list[PerStepProbeResult], output_path: pathlib.Path) -> None:
    """Plot probe accuracy vs step for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    short_names = []
    for i, r in enumerate(all_results):
        short = r.model_name.split("/")[-1]
        short_names.append(short)
        steps = sorted(r.step_accuracies.keys())
        if not steps:
            continue
        accs = [r.step_accuracies[s] for s in steps]
        ax.plot(steps, accs, marker="o", markersize=4, label=f"{short} ({r.success_rate:.0%} win)", color=colors[i])

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Step (question number)")
    ax.set_ylabel("Probe accuracy (CV)")
    ax.set_title("Can the residual stream predict episode success at each step?")
    ax.legend()
    ax.set_ylim(0.35, 1.05)
    ax.grid(True, alpha=0.3)

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


async def run_all_models() -> list[PerStepProbeResult]:
    all_results: list[PerStepProbeResult] = []

    for model_name in MODELS:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        print(f"Loading {model_name} on {DEVICE}...")
        model = HookedTransformer.from_pretrained(model_name, device=DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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
        episodes = await collect_episodes(
            model=model,
            client=client,
            n_episodes=N_EPISODES,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
            middle_layer=middle_layer,
        )
        print(f"\nCollected {len(episodes)} usable episodes")

        result = train_per_step_probes(
            episodes=episodes,
            model_name=model_name,
            n_layers=n_layers,
            d_model=d_model,
            middle_layer=middle_layer,
        )
        all_results.append(result)

        del model, tokenizer, client
        _free_gpu_memory()

    return all_results


def save_results_json(results: list[PerStepProbeResult], path: pathlib.Path) -> None:
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
                "n_successes": r.n_successes,
                "success_rate": r.success_rate,
                "step_accuracies": {str(k): v for k, v in r.step_accuracies.items()},
                "step_sample_counts": {str(k): v for k, v in r.step_sample_counts.items()},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Results JSON saved to {path}")


def print_comparison_table(results: list[PerStepProbeResult]) -> None:
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    header = f"{'Model':<35} {'Layers':>6} {'d_model':>7} {'Episodes':>8} {'Success%':>8} {'MeanAcc':>8}"
    print(header)
    print("-" * len(header))

    for r in results:
        if r.step_accuracies:
            mean_acc = np.mean(list(r.step_accuracies.values()))
            acc_str = f"{mean_acc:.4f}"
        else:
            acc_str = "   N/A"
        print(
            f"{r.model_name:<35} {r.n_layers:>6} {r.d_model:>7} "
            f"{r.n_episodes:>8} {r.success_rate:>7.1%} {acc_str:>8}"
        )
    print()


def main() -> None:
    results = asyncio.run(run_all_models())
    print_comparison_table(results)
    save_results_json(results, OUTPUT_DIR / "results.json")
    plot_results(results, OUTPUT_DIR / "probe_accuracy_vs_step.png")


if __name__ == "__main__":
    main()
