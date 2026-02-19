#!/usr/bin/env python

# # Visualise Steering Vector Evolution Across Steps
#
# Shows that the linear direction separating "valid" from "invalid" questions
# is not static — it evolves as the conversation unfolds.
#
# Produces three plots:
#   1. Cosine similarity heatmap between steering vectors at each pair of steps
#   2. Steering vector norm per step (magnitude of the valid-invalid separation)
#   3. PCA projection of steering vectors showing trajectory through activation space
#
# ## Run
#   uv run --no-sync python examples/20q_case_study/visualise_steering_vectors.py

import json
import pathlib

from matplotlib import collections
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration (must match phase2_steer.py)
# ---------------------------------------------------------------------------

SEED = 42
TRAIN_RATIO = 0.8
POOLING = "last-prompt"
MIN_CLASS_SAMPLES = 5

DATA_DIR = pathlib.Path("outputs/20q_data")
OUTPUT_DIR = pathlib.Path("outputs/20q_steering_vector_evolution")

# ---------------------------------------------------------------------------
# Reuse helpers from the case study
# ---------------------------------------------------------------------------


def _pool_activation(activation: torch.Tensor, method: str, prompt_len: int = 0) -> np.ndarray:
    arr = activation.numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 activation, got shape={arr.shape}")
    if method == "last-prompt":
        if prompt_len <= 0:
            raise ValueError("prompt_len required for 'last-prompt' pooling")
        return arr[0, prompt_len - 1, :]
    if method == "mean-prompt":
        if prompt_len <= 0:
            raise ValueError("prompt_len required for 'mean-prompt' pooling")
        return arr[0, :prompt_len, :].mean(axis=0)
    if method == "mean":
        return arr[0].mean(axis=0)
    raise ValueError(f"Unknown pooling method: {method}")


def load_episodes(data_dir: pathlib.Path) -> tuple[list[dict], dict]:
    metadata = json.loads((data_dir / "metadata.json").read_text())
    ep_files = sorted(data_dir.glob("episode_*.pt"))
    episodes = [torch.load(f, weights_only=False) for f in ep_files]
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes, metadata


def train_test_split(episode_indices: list[int], train_ratio: float) -> tuple[set[int], set[int]]:
    sorted_indices = sorted(episode_indices)
    n_train = max(1, int(len(sorted_indices) * train_ratio))
    rng = np.random.RandomState(SEED)
    rng.shuffle(sorted_indices)
    return set(sorted_indices[:n_train]), set(sorted_indices[n_train:])


# ---------------------------------------------------------------------------
# Compute per-step steering vectors
# ---------------------------------------------------------------------------


def compute_per_step_vectors(
    episodes: list[dict],
    train_eps: set[int],
    pooling: str,
    min_samples: int,
) -> dict[int, np.ndarray]:
    step_valid: dict[int, list[np.ndarray]] = {}
    step_invalid: dict[int, list[np.ndarray]] = {}

    for ep in episodes:
        if ep["episode_idx"] not in train_eps:
            continue
        for step in ep["steps"]:
            activation = step.get("activation")
            if activation is None:
                continue
            s = step["step_idx"]
            prompt_len = step.get("prompt_len", 0)
            feature = _pool_activation(activation, pooling, prompt_len=prompt_len)
            if step["is_invalid"]:
                step_invalid.setdefault(s, []).append(feature)
            else:
                step_valid.setdefault(s, []).append(feature)

    all_steps = sorted(set(step_valid.keys()) | set(step_invalid.keys()))
    vectors: dict[int, np.ndarray] = {}
    for s in all_steps:
        n_v = len(step_valid.get(s, []))
        n_i = len(step_invalid.get(s, []))
        if n_v >= min_samples and n_i >= min_samples:
            v_valid = np.stack(step_valid[s]).mean(axis=0)
            v_invalid = np.stack(step_invalid[s]).mean(axis=0)
            vectors[s] = v_valid - v_invalid

    print(f"Computed steering vectors for {len(vectors)} steps: {sorted(vectors.keys())}")
    return vectors


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cosine_similarity_heatmap(vectors: dict[int, np.ndarray], output_path: pathlib.Path) -> None:
    steps = sorted(vectors.keys())
    n = len(steps)
    sim_matrix = np.zeros((n, n))

    for i, si in enumerate(steps):
        for j, sj in enumerate(steps):
            vi, vj = vectors[si], vectors[sj]
            cos = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-12)
            sim_matrix[i, j] = cos

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(steps)
    ax.set_yticks(range(n))
    ax.set_yticklabels(steps)
    ax.set_xlabel("Step")
    ax.set_ylabel("Step")
    ax.set_title(
        "Cosine similarity between per-step steering vectors\n(valid - invalid direction evolves across the trajectory)"
    )

    # Annotate cells.
    for i in range(n):
        for j in range(n):
            color = "white" if abs(sim_matrix[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_norm_per_step(vectors: dict[int, np.ndarray], output_path: pathlib.Path) -> None:
    steps = sorted(vectors.keys())
    norms = [np.linalg.norm(vectors[s]) for s in steps]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(steps)), norms, color="#3182bd", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps)
    ax.set_xlabel("Step (question number)")
    ax.set_ylabel("||v_valid - v_invalid||")
    ax.set_title("Steering vector norm per step\n(magnitude of the valid/invalid separation in activation space)")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, norm in zip(bars, norms, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{norm:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_pca_trajectory(vectors: dict[int, np.ndarray], output_path: pathlib.Path) -> None:
    steps = sorted(vectors.keys())
    matrix = np.stack([vectors[s] for s in steps])  # [n_steps, d_model]

    # Centre and PCA.
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)  # noqa: N806, RUF059
    coords = centred @ Vt[:2].T  # project onto first 2 PCs

    explained = S[:2] ** 2 / (S**2).sum()

    fig, ax = plt.subplots(figsize=(8, 7))

    # Draw trajectory line coloured by step index.
    points = coords.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(min(steps), max(steps))
    lc = collections.LineCollection(segments, cmap="viridis", norm=norm, linewidths=2, alpha=0.7)
    lc.set_array(np.array(steps[:-1]))
    ax.add_collection(lc)

    # Scatter with step labels.
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=steps,
        cmap="viridis",
        s=80,
        zorder=5,
        edgecolors="white",
        linewidths=1.2,
    )
    for i, s in enumerate(steps):
        ax.annotate(
            f"t={s}",
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.7, "ec": "none"},
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.0%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.0%} variance)")
    ax.set_title(
        "Steering vector trajectory across steps (PCA)\n"
        "The direction that separates valid from invalid questions drifts over time"
    )
    fig.colorbar(scatter, ax=ax, label="Step", shrink=0.8)
    ax.grid(True, alpha=0.2)

    # Auto-scale to data.
    pad = 0.15 * max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Visualising steering vector evolution across steps")
    print("=" * 70)

    episodes, _ = load_episodes(DATA_DIR)
    all_ep_indices = sorted({ep["episode_idx"] for ep in episodes})
    train_eps, _ = train_test_split(all_ep_indices, TRAIN_RATIO)

    vectors = compute_per_step_vectors(episodes, train_eps, POOLING, MIN_CLASS_SAMPLES)

    if len(vectors) < 2:
        print("Need at least 2 steps with steering vectors to visualise. Exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_cosine_similarity_heatmap(vectors, OUTPUT_DIR / "cosine_similarity_heatmap.png")
    plot_norm_per_step(vectors, OUTPUT_DIR / "steering_vector_norms.png")
    plot_pca_trajectory(vectors, OUTPUT_DIR / "pca_trajectory.png")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
