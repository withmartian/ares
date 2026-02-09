"""Mechanistic interpretability example: probing correctness beliefs in SWE-bench agents.

This example demonstrates how to:
1. Generate multi-turn agent trajectories with activations + confidence labels
2. Train a linear probe to predict agent's correctness belief from hidden states
3. Compare self-reported confidence vs probe predictions across timesteps
4. Evaluate probe accuracy and ROC-AUC on test data

The example is designed to explore whether smaller models (7B/8B) show calibrated
internal representations of correctness even when their self-reported confidence
might be miscalibrated. This is inspired by the SWE-bench Verified benchmark.

Prerequisites:
    - Install dependencies: `uv sync --group examples`
    - For mock mode: No additional setup needed (CPU-friendly)
    - For real mode: Local Docker + LLM model + ARES environment setup

Example usage:
    # Run with synthetic multi-turn traces (default, fast)
    uv run -m examples.07_mi_swebench_correctness_probe

    # Generate more episodes and save to cache
    uv run -m examples.07_mi_swebench_correctness_probe --n-episodes 50 --cache-dir ./mi_cache

    # Load from cache
    uv run -m examples.07_mi_swebench_correctness_probe --cache-dir ./mi_cache --load-cache

    # Real mode stub (shows integration pattern - requires full setup)
    uv run -m examples.07_mi_swebench_correctness_probe --real
"""

# ruff: noqa: N806, N999
# N806: Allow uppercase variable names (X, y) for ML matrices
# N999: Allow numbered module name for example files

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt

from . import mi_utils

_LOGGER = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def generate_synthetic_multi_turn_trace(
    n_turns: int = 10,
    hidden_dim: int = 512,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Generate a synthetic multi-turn agent trace with activations and confidence.

    Simulates an agent working on a task over multiple turns. The agent has:
    - A true internal "correctness" state that evolves over time
    - Hidden activations that correlate with this correctness state
    - Self-reported confidence that may be miscalibrated

    Args:
        n_turns: Number of timesteps in the trace
        hidden_dim: Dimensionality of hidden states
        seed: Random seed for reproducibility

    Returns:
        Tuple of (activations, correctness_labels, confidence_scores) where:
            activations: shape (n_turns, hidden_dim)
            correctness_labels: shape (n_turns,) with binary labels (0 or 1)
            confidence_scores: shape (n_turns,) with self-reported confidence [0, 1]
    """
    rng = np.random.RandomState(seed)

    # Generate base activations
    X = rng.randn(n_turns, hidden_dim).astype(np.float32)

    # Simulate agent correctness evolving over time
    # Start uncertain, gradually improve (or worsen for some traces)
    initial_correctness = rng.rand() > 0.5
    correctness_trajectory = [initial_correctness]

    for _t in range(1, n_turns):
        # 70% chance to maintain state, 30% to flip
        if rng.rand() < 0.7:
            correctness_trajectory.append(correctness_trajectory[-1])
        else:
            correctness_trajectory.append(not correctness_trajectory[-1])

    y = np.array(correctness_trajectory, dtype=np.int32)

    # Add signal to activations: correct states have higher activation in some dims
    signal_dims = 64
    signal_strength = 0.8
    for t in range(n_turns):
        if y[t] == 1:
            X[t, :signal_dims] += signal_strength
        # Add some noise
        X[t] += rng.randn(hidden_dim).astype(np.float32) * 0.1

    # Generate self-reported confidence (potentially miscalibrated)
    # Confidence is correlated with correctness but with noise
    confidence = np.zeros(n_turns, dtype=np.float32)
    for t in range(n_turns):
        if y[t] == 1:
            # Correct: confidence should be high but add noise
            confidence[t] = np.clip(0.7 + rng.randn() * 0.15, 0.0, 1.0)
        else:
            # Incorrect: confidence should be low but add noise (miscalibration)
            confidence[t] = np.clip(0.4 + rng.randn() * 0.2, 0.0, 1.0)

    return X, y, confidence


def visualize_confidence_vs_probe(
    confidence: npt.NDArray[np.float32],
    probe_proba: npt.NDArray[np.float32],
    true_labels: npt.NDArray[np.int32],
    title: str = "Sample Trajectory",
) -> None:
    """Print ASCII visualization comparing confidence vs probe predictions.

    Args:
        confidence: Self-reported confidence scores
        probe_proba: Probe predicted probabilities
        true_labels: Ground truth correctness labels
        title: Title for the visualization
    """
    print()
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)
    print("Legend: ✓ = correct, ✗ = incorrect | Self-conf | Probe-pred")
    print("-" * 80)

    for t in range(len(confidence)):
        label_str = "✓" if true_labels[t] == 1 else "✗"
        conf_val = confidence[t]
        prob_val = probe_proba[t]

        # Create bar visualization
        conf_bar = "█" * int(conf_val * 20)
        prob_bar = "█" * int(prob_val * 20)

        print(f"t={t:2d} {label_str} | {conf_val:.2f} {conf_bar:20s} | {prob_val:.2f} {prob_bar:20s}")

    print("-" * 80)
    print()


def print_results(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary with accuracy and AUC metrics
    """
    print()
    print("=" * 80)
    print("CORRECTNESS PROBE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:  {metrics['auc']:.4f}")
    print("=" * 80)
    print()


async def run_mock_mode(
    n_episodes: int = 20,
    n_turns_per_episode: int = 10,
    hidden_dim: int = 512,
    test_split: float = 0.2,
    cache_dir: Path | None = None,
    load_cache: bool = False,
) -> None:
    """Run the example in mock mode with synthetic data.

    Args:
        n_episodes: Number of agent episodes to generate
        n_turns_per_episode: Number of turns per episode
        hidden_dim: Hidden dimension size
        test_split: Fraction of data to use for testing
        cache_dir: Directory to save/load cached activations (as npz)
        load_cache: Whether to load from cache instead of generating
    """
    print()
    print("=" * 80)
    print("MECHANISTIC INTERPRETABILITY: SWE-BENCH CORRECTNESS PROBE (MOCK MODE)")
    print("=" * 80)
    print()

    cache_path = cache_dir / "swebench_correctness_cache.npz" if cache_dir else None

    # Generate or load data
    if load_cache and cache_path and cache_path.exists():
        print(f"Loading cached data from {cache_path}...")
        data = np.load(cache_path)
        X = data["activations"]
        y = data["labels"]
        confidence = data["confidence"]
        print(f"Loaded {len(y)} samples with hidden_dim={X.shape[1]}")
    else:
        print("Generating synthetic multi-turn trajectories...")
        print(f"  Episodes: {n_episodes}")
        print(f"  Turns per episode: {n_turns_per_episode}")
        print(f"  Hidden dim: {hidden_dim}")
        print()

        all_X = []
        all_y = []
        all_confidence = []

        for episode_idx in range(n_episodes):
            X_ep, y_ep, conf_ep = generate_synthetic_multi_turn_trace(
                n_turns=n_turns_per_episode,
                hidden_dim=hidden_dim,
                seed=42 + episode_idx,
            )
            all_X.append(X_ep)
            all_y.append(y_ep)
            all_confidence.append(conf_ep)

        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        confidence = np.concatenate(all_confidence)

        print(f"Generated {len(y)} total samples across {n_episodes} episodes")

        # Save to cache if requested
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            if cache_path:
                print(f"Saving data to cache: {cache_path}")
                np.savez(cache_path, activations=X, labels=y, confidence=confidence)
        print()

    # Split into train/test
    n_samples = len(y)
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_test

    # Shuffle indices
    rng = np.random.RandomState(42)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Split data: {n_train} train, {n_test} test samples")
    print()

    # Train the probe
    print("Training correctness probe with sklearn LogisticRegression...")
    probe = mi_utils.SklearnLogisticProbe(hidden_dim=hidden_dim)
    probe.train(X_train, y_train, max_iter=200)
    print("Training complete!")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = probe.evaluate(X_test, y_test)
    print_results(metrics)

    # Visualize one complete episode from test set
    # Find which episode the first test sample belongs to
    first_test_idx = test_indices[0]
    episode_idx = first_test_idx // n_turns_per_episode
    episode_start = episode_idx * n_turns_per_episode
    episode_end = episode_start + n_turns_per_episode

    # Get this complete episode (even if some samples were in train set)
    traj_X = X[episode_start:episode_end]
    traj_y = y[episode_start:episode_end]
    traj_conf = confidence[episode_start:episode_end]
    traj_probe = probe.predict_proba(traj_X)

    visualize_confidence_vs_probe(
        traj_conf,
        traj_probe,
        traj_y,
        title=f"Sample Trajectory: Episode {episode_idx}",
    )

    # Interpretation
    print("Top 10 most important dimensions (by absolute weight):")
    print("-" * 80)
    top_indices = np.argsort(np.abs(probe.weights))[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        weight = probe.weights[idx]
        print(f"  {i:2d}. Dimension {idx:4d}: weight = {weight:+.4f}")
    print("-" * 80)
    print()

    print("✅ SWE-bench correctness probe example complete!")
    print()
    print("Key findings:")
    print("  - The probe can predict agent correctness from hidden activations")
    print("  - Self-reported confidence may be miscalibrated vs internal belief")
    print("  - This technique could reveal if 7B/8B models 'know' more than they say")
    print()
    print("Next steps:")
    print("  - Extend to real SWE-bench runs with actual model activations")
    print("  - Compare calibration across different model sizes")
    print("  - Investigate which layers/dimensions encode correctness best")
    print("=" * 80)


async def run_real_mode() -> None:
    """Run in real mode with actual SWE-bench environment (stub).

    This is a stub showing how to integrate with real ARES environments.
    Requires full setup including API keys, containers, and model access.
    """
    print()
    print("=" * 80)
    print("REAL MODE (STUB)")
    print("=" * 80)
    print()
    print("To run with real SWE-bench data, you would:")
    print()
    print("1. Set up an ARES environment with SWE-bench Verified:")
    print("   ```python")
    print("   import ares")
    print("   async with ares.make('sbv-mswea:0') as env:")
    print("       ts = await env.reset()")
    print("   ```")
    print()
    print("2. Wrap your LLM client with ActivationRecordingLLMClient:")
    print("   ```python")
    print("   from examples import mi_utils")
    print("   recording_client = mi_utils.ActivationRecordingLLMClient(")
    print("       client=your_llm_client,")
    print("       layer_index=-1,  # Last layer")
    print("   )")
    print("   ```")
    print()
    print("3. Modify your LLM client to extract activations from the model")
    print("   - For transformers: use hooks to capture hidden states")
    print("   - Add activations to response metadata or record directly")
    print()
    print("4. Run episodes and collect (activation, correctness_label) pairs:")
    print("   - Correctness can be inferred from task success")
    print("   - Or from intermediate validation steps")
    print()
    print("5. Train the probe on collected data and analyze results")
    print()
    print("=" * 80)
    print()
    print("TODOs for full implementation:")
    print("  [ ] Add transformer hooks to extract activations from model")
    print("  [ ] Implement confidence parsing from agent responses")
    print("  [ ] Add episode runner that collects multi-turn trajectories")
    print("  [ ] Integrate with ARES environment for real SWE-bench tasks")
    print("=" * 80)


def main() -> None:
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="SWE-bench correctness probe with mechanistic interpretability")
    parser.add_argument(
        "--real",
        action="store_true",
        help="Run in real mode with actual environment (stub)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Number of episodes to generate in mock mode (default: 20)",
    )
    parser.add_argument(
        "--n-turns",
        type=int,
        default=10,
        help="Number of turns per episode in mock mode (default: 10)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension size (default: 512)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Directory to save/load cached activations (npz format)",
    )
    parser.add_argument(
        "--load-cache",
        action="store_true",
        help="Load from cache instead of generating new data",
    )

    args = parser.parse_args()

    if args.real:
        asyncio.run(run_real_mode())
    else:
        asyncio.run(
            run_mock_mode(
                n_episodes=args.n_episodes,
                n_turns_per_episode=args.n_turns,
                hidden_dim=args.hidden_dim,
                test_split=args.test_split,
                cache_dir=args.cache_dir,
                load_cache=args.load_cache,
            )
        )


if __name__ == "__main__":
    main()
