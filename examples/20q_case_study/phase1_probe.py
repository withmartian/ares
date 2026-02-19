#!/usr/bin/env python
"""Probing for "Invalid Question" in Twenty Questions.

Loads pre-collected 20Q episode data (from collect_20q_data.py), trains
linear probes on middle-layer residual stream activations, and evaluates
on a held-out test set.

No GPU required -- all computation is numpy / sklearn.

Requirements:
    pip install scikit-learn matplotlib numpy torch   (torch only for loading .pt files)

Run:
    uv run --no-sync python examples/20q_case_study/phase1_probe.py
"""

import dataclasses
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42

DATA_DIR = pathlib.Path("outputs/20q_data")
OUTPUT_DIR = pathlib.Path("outputs/20q_probing_results")

# Train/test split ratio (by episode).
TRAIN_RATIO = 0.8

# Feature pooling over the token dimension: "mean" or "last".
FEATURE_POOLING = "mean"

# Minimum samples at a given step to attempt a per-step probe.
MIN_SAMPLES_FOR_PROBE = 8

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepSample:
    """A single step's activation vector and oracle-invalidity label."""

    feature: np.ndarray  # shape [d_model]
    is_invalid: int  # 1 = oracle responded "Invalid Question.", 0 = valid
    episode_idx: int
    step_idx: int


@dataclasses.dataclass
class ProbeResult:
    """Aggregated probe results."""

    model_name: str
    n_layers: int
    d_model: int
    middle_layer: int
    n_episodes: int
    n_train_episodes: int
    n_test_episodes: int
    n_total_steps: int
    n_invalid_steps: int
    invalid_rate: float
    feature_pooling: str
    # Per-step probe test accuracy: step_idx -> accuracy
    step_accuracies: dict[int, float] = dataclasses.field(default_factory=dict)
    step_sample_counts: dict[int, int] = dataclasses.field(default_factory=dict)
    # Global (pooled) probe metrics
    global_accuracy: float = 0.0
    global_report: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _pool_activation(activation: torch.Tensor, method: str) -> np.ndarray:
    """Convert a [1, seq_len, d_model] tensor to a [d_model] numpy vector."""
    arr = activation.numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 activation, got shape={arr.shape}")
    if method == "mean":
        return arr[0].mean(axis=0)
    if method == "last":
        return arr[0, -1, :]
    raise ValueError(f"Unknown pooling method: {method}")


def load_episodes(data_dir: pathlib.Path) -> tuple[list[dict], dict]:
    """Load all episode .pt files and metadata from *data_dir*.

    Returns (episodes, metadata) where each episode is the dict saved by
    collect_20q_data.py.
    """
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json in {data_dir}. Run collect_20q_data.py first.")

    metadata = json.loads(metadata_path.read_text())

    ep_files = sorted(data_dir.glob("episode_*.pt"))
    if not ep_files:
        raise FileNotFoundError(f"No episode_*.pt files in {data_dir}. Run collect_20q_data.py first.")

    episodes = []
    for ep_file in ep_files:
        episodes.append(torch.load(ep_file, weights_only=False))

    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes, metadata


def episodes_to_samples(episodes: list[dict], pooling: str) -> list[StepSample]:
    """Convert loaded episodes into a flat list of StepSamples."""
    samples: list[StepSample] = []
    for ep in episodes:
        ep_idx = ep["episode_idx"]
        for step in ep["steps"]:
            activation = step.get("activation")
            if activation is None:
                continue
            feature = _pool_activation(activation, pooling)
            samples.append(
                StepSample(
                    feature=feature,
                    is_invalid=step["is_invalid"],
                    episode_idx=ep_idx,
                    step_idx=step["step_idx"],
                )
            )
    return samples


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def train_test_split_by_episode(
    samples: list[StepSample],
    train_ratio: float,
) -> tuple[list[StepSample], list[StepSample]]:
    """Split samples into train/test by episode index (no data leakage)."""
    episode_indices = sorted({s.episode_idx for s in samples})
    n_train = max(1, int(len(episode_indices) * train_ratio))
    rng = np.random.RandomState(SEED)
    rng.shuffle(episode_indices)  # type: ignore[arg-type]
    train_eps = set(episode_indices[:n_train])
    test_eps = set(episode_indices[n_train:])

    train = [s for s in samples if s.episode_idx in train_eps]
    test = [s for s in samples if s.episode_idx in test_eps]
    print(
        f"Split: {len(train_eps)} train episodes ({len(train)} steps), "
        f"{len(test_eps)} test episodes ({len(test)} steps)"
    )
    return train, test


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _fit_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Fit logistic regression on train, return accuracy on test."""
    probe = linear_model.LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=SEED)
    probe.fit(x_train, y_train)
    return float(np.mean(probe.predict(x_test) == y_test))


def train_probes(
    train_samples: list[StepSample],
    test_samples: list[StepSample],
    metadata: dict,
) -> ProbeResult:
    """Train per-step and global probes on train set, evaluate on test set."""
    all_samples = train_samples + test_samples
    n_total = len(all_samples)
    n_invalid = sum(s.is_invalid for s in all_samples)
    invalid_rate = n_invalid / n_total if n_total else 0.0

    n_train_eps = len({s.episode_idx for s in train_samples})
    n_test_eps = len({s.episode_idx for s in test_samples})

    print(f"\n  Total steps: {n_total},  Invalid: {n_invalid}/{n_total} ({invalid_rate:.1%})")

    result = ProbeResult(
        model_name=metadata.get("model_name", "unknown"),
        n_layers=metadata.get("n_layers", 0),
        d_model=metadata.get("d_model", 0),
        middle_layer=metadata.get("middle_layer", 0),
        n_episodes=n_train_eps + n_test_eps,
        n_train_episodes=n_train_eps,
        n_test_episodes=n_test_eps,
        n_total_steps=n_total,
        n_invalid_steps=n_invalid,
        invalid_rate=invalid_rate,
        feature_pooling=FEATURE_POOLING,
    )

    # --- Global probe (pooled across all steps) ---
    x_train_all = np.stack([s.feature for s in train_samples])
    y_train_all = np.array([s.is_invalid for s in train_samples], dtype=np.int64)
    x_test_all = np.stack([s.feature for s in test_samples])
    y_test_all = np.array([s.is_invalid for s in test_samples], dtype=np.int64)

    n_pos_train = int(y_train_all.sum())
    n_neg_train = len(y_train_all) - n_pos_train
    n_pos_test = int(y_test_all.sum())
    n_neg_test = len(y_test_all) - n_pos_test

    if n_pos_train >= 2 and n_neg_train >= 2 and n_pos_test >= 1 and n_neg_test >= 1:
        result.global_accuracy = _fit_and_eval(x_train_all, y_train_all, x_test_all, y_test_all)

        # Full classification report on test set.
        probe = linear_model.LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=SEED
        )
        probe.fit(x_train_all, y_train_all)
        y_pred = probe.predict(x_test_all)
        result.global_report = metrics.classification_report(
            y_test_all, y_pred, target_names=["Valid (Yes/No)", "Invalid Question"], zero_division=0
        )
        print(f"\n  Global probe test accuracy: {result.global_accuracy:.3f}")
        print(f"\n  Classification report (test set):\n{result.global_report}")
    else:
        print("\n  Global probe: skipped (insufficient class diversity)")

    # --- Per-step probes ---
    max_step = max(s.step_idx for s in all_samples) + 1
    print(f"  Max step index: {max_step - 1}")

    for step_idx in range(max_step):
        train_step = [s for s in train_samples if s.step_idx == step_idx]
        test_step = [s for s in test_samples if s.step_idx == step_idx]
        result.step_sample_counts[step_idx] = len(train_step) + len(test_step)

        if len(train_step) < MIN_SAMPLES_FOR_PROBE or len(test_step) < 2:
            print(f"  step {step_idx:2d}: train={len(train_step):3d} test={len(test_step):3d}  -- skipped")
            continue

        x_tr = np.stack([s.feature for s in train_step])
        y_tr = np.array([s.is_invalid for s in train_step], dtype=np.int64)
        x_te = np.stack([s.feature for s in test_step])
        y_te = np.array([s.is_invalid for s in test_step], dtype=np.int64)

        n_pos_tr = int(y_tr.sum())
        n_neg_tr = len(y_tr) - n_pos_tr
        if n_pos_tr == 0 or n_neg_tr == 0:
            print(f"  step {step_idx:2d}: train={len(y_tr):3d} (inv={n_pos_tr})  -- skipped (no class diversity)")
            continue

        acc = _fit_and_eval(x_tr, y_tr, x_te, y_te)
        result.step_accuracies[step_idx] = acc
        print(f"  step {step_idx:2d}: train={len(y_tr):3d} (inv={n_pos_tr})  test={len(y_te):3d}  acc={acc:.3f}")

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(result: ProbeResult, output_path: pathlib.Path) -> None:
    """Plot per-step probe test accuracy with global probe as a reference line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = sorted(result.step_accuracies.keys())
    if not steps:
        print("No per-step results to plot.")
        plt.close(fig)
        return

    accs = [result.step_accuracies[s] for s in steps]
    short_name = result.model_name.split("/")[-1]

    ax.plot(steps, accs, marker="o", markersize=5, color="#1f77b4", label=f"Per-step probe ({short_name})")

    # Annotate each point with its sample count (train + test).
    for s, a in zip(steps, accs, strict=True):
        n = result.step_sample_counts.get(s, 0)
        ax.annotate(f"n={n}", (s, a), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7, alpha=0.7)

    ax.axhline(
        result.global_accuracy,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
        label=f"Global probe ({result.global_accuracy:.3f})",
    )
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")

    ax.set_xlabel("Step (question number)")
    ax.set_ylabel("Probe test accuracy")
    ax.set_title(
        f"Linear probe: predict 'Invalid Question' from resid_post (layer {result.middle_layer})\n"
        f"{result.n_episodes} episodes, {result.feature_pooling}-pooled, "
        f"{result.invalid_rate:.0%} invalid overall"
    )
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


def save_results_json(result: ProbeResult, path: pathlib.Path) -> None:
    """Persist numeric results for re-plotting without re-running probes."""
    data = {
        "model_name": result.model_name,
        "n_layers": result.n_layers,
        "d_model": result.d_model,
        "middle_layer": result.middle_layer,
        "n_episodes": result.n_episodes,
        "n_train_episodes": result.n_train_episodes,
        "n_test_episodes": result.n_test_episodes,
        "n_total_steps": result.n_total_steps,
        "n_invalid_steps": result.n_invalid_steps,
        "invalid_rate": result.invalid_rate,
        "feature_pooling": result.feature_pooling,
        "global_accuracy": result.global_accuracy,
        "global_report": result.global_report,
        "step_accuracies": {str(k): v for k, v in result.step_accuracies.items()},
        "step_sample_counts": {str(k): v for k, v in result.step_sample_counts.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Results JSON saved to {path}")


def main() -> None:
    np.random.seed(SEED)

    print(f"Loading data from {DATA_DIR}...")
    episodes, metadata = load_episodes(DATA_DIR)

    print(f"Pooling activations ({FEATURE_POOLING})...")
    samples = episodes_to_samples(episodes, FEATURE_POOLING)
    print(f"Total samples: {len(samples)}")

    if not samples:
        print("No samples with activations found. Check that collect_20q_data.py ran successfully.")
        return

    train_samples, test_samples = train_test_split_by_episode(samples, TRAIN_RATIO)

    if not test_samples:
        print("No test samples after split. Need more episodes.")
        return

    result = train_probes(train_samples, test_samples, metadata)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Model:           {result.model_name}")
    print(f"  Layer probed:    {result.middle_layer} (middle of {result.n_layers})")
    print(f"  Feature dim:     {result.d_model}")
    print(f"  Pooling:         {result.feature_pooling}")
    print(f"  Episodes:        {result.n_episodes} ({result.n_train_episodes} train / {result.n_test_episodes} test)")
    print(f"  Total steps:     {result.n_total_steps}")
    print(f"  Invalid rate:    {result.invalid_rate:.1%}")
    print(f"  Global test acc: {result.global_accuracy:.4f}")
    print()

    save_results_json(result, OUTPUT_DIR / "results.json")
    plot_results(result, OUTPUT_DIR / "probe_invalid_question_vs_step.png")


if __name__ == "__main__":
    main()
