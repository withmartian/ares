"""Pedagogical mechanistic interpretability example with linear probing.

This example demonstrates how to:
1. Record hidden activations from an LLM during agent interactions
2. Define a probe target based on agent's self-reported confidence
3. Train a linear probe to predict the target from activations
4. Evaluate the probe and visualize results

The example uses synthetic data by default to be CPU-friendly and runnable
without heavy infrastructure. It can be extended to work with real model
activations by modifying the LLM client.

Prerequisites:
    - Install dependencies: `uv sync --group examples`
    - For running with cached data: No additional setup needed
    - For running with live models: Local Docker + LLM model

Example usage:
    # Run with synthetic data (default, fast)
    uv run -m examples.06_mi_linear_probe

    # Run with cached activations (if you have them)
    uv run -m examples.06_mi_linear_probe --load-cache cache.pkl

    # Generate and save activations for later use
    uv run -m examples.06_mi_linear_probe --save-cache cache.pkl --n-episodes 10
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


def generate_synthetic_activations(
    n_samples: int = 100,
    hidden_dim: int = 512,
    seed: int = 42,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Generate synthetic activation data for pedagogical purposes.

    Creates synthetic hidden states where positive examples have higher
    activations in certain dimensions, making the probe learning task
    tractable but non-trivial.

    Args:
        n_samples: Number of samples to generate
        hidden_dim: Dimensionality of hidden states
        seed: Random seed for reproducibility

    Returns:
        Tuple of (activations, labels) where:
            activations: shape (n_samples, hidden_dim)
            labels: shape (n_samples,) with binary labels (0 or 1)
    """
    rng = np.random.RandomState(seed)

    # Generate base activations from standard normal
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)

    # Create labels (50/50 split)
    y = rng.randint(0, 2, size=n_samples).astype(np.int32)

    # Add signal: increase activations in first 50 dimensions for positive class
    signal_dims = 50
    signal_strength = 0.5
    for i in range(n_samples):
        if y[i] == 1:
            X[i, :signal_dims] += signal_strength

    return X, y


def print_results(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary with accuracy and AUC metrics
    """
    print()
    print("=" * 80)
    print("PROBE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    print("=" * 80)
    print()


def plot_training_curve(losses: list[float]) -> None:
    """Print a simple ASCII training curve.

    Args:
        losses: List of loss values per epoch
    """
    print()
    print("Training Loss Curve:")
    print("-" * 60)

    # Downsample if too many epochs
    if len(losses) > 20:
        step = len(losses) // 20
        losses = losses[::step]

    min_loss = min(losses)
    max_loss = max(losses)
    range_loss = max_loss - min_loss

    if range_loss < 1e-6:
        print("Loss converged to constant value")
        return

    for i, loss in enumerate(losses):
        # Normalize to 0-50 range for display
        bar_len = int(50 * (loss - min_loss) / range_loss)
        bar = "#" * bar_len
        print(f"Epoch {i:3d}: {loss:.4f} |{bar}")

    print("-" * 60)
    print()


async def run_pedagogical_example(
    n_samples: int = 100,
    hidden_dim: int = 512,
    test_split: float = 0.2,
    cache_path: Path | None = None,
    load_cache: bool = False,
) -> None:
    """Run the pedagogical MI example.

    Args:
        n_samples: Number of samples to generate
        hidden_dim: Hidden dimension size
        test_split: Fraction of data to use for testing
        cache_path: Path to save/load cached activations
        load_cache: Whether to load from cache instead of generating
    """
    print()
    print("=" * 80)
    print("MECHANISTIC INTERPRETABILITY: LINEAR PROBE EXAMPLE")
    print("=" * 80)
    print()

    # Generate or load activations
    if load_cache and cache_path and cache_path.exists():
        print(f"Loading cached activations from {cache_path}...")
        recording_client = mi_utils.ActivationRecordingLLMClient(
            client=None,  # type: ignore
            layer_index=-1,
        )
        recording_client.load_records(cache_path)

        # Extract X and y from records
        X = np.array([record.activations for record in recording_client.records])
        y = np.array([record.label for record in recording_client.records])
        n_samples = len(y)
        hidden_dim = X.shape[1]
        print(f"Loaded {n_samples} samples with hidden_dim={hidden_dim}")
    else:
        print("Generating synthetic activations...")
        print(f"  Samples: {n_samples}")
        print(f"  Hidden dim: {hidden_dim}")
        print()

        X, y = generate_synthetic_activations(
            n_samples=n_samples,
            hidden_dim=hidden_dim,
        )

        # Optionally save to cache
        if cache_path:
            print(f"Saving activations to cache: {cache_path}")
            recording_client = mi_utils.ActivationRecordingLLMClient(
                client=None,  # type: ignore
                layer_index=-1,
            )
            for i in range(n_samples):
                recording_client.add_record(
                    activations=X[i],
                    label=int(y[i]),
                    metadata={"sample_id": i},
                )
            recording_client.save_records(cache_path)
            print()

    # Split into train/test
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

    # Initialize and train probe
    print("Training linear probe...")
    probe = mi_utils.LinearProbe(hidden_dim=hidden_dim)

    losses = probe.train(
        X_train,
        y_train,
        learning_rate=0.1,
        n_epochs=100,
        l2_reg=0.01,
    )

    print("Training complete!")
    plot_training_curve(losses)

    # Evaluate on test set
    print("Evaluating on test set...")
    metrics = probe.evaluate(X_test, y_test)
    print_results(metrics)

    # Interpretation: which dimensions are most important?
    print("Top 10 most important dimensions (by absolute weight):")
    print("-" * 60)
    top_indices = np.argsort(np.abs(probe.weights))[-10:][::-1]
    for i, idx in enumerate(top_indices, 1):
        weight = probe.weights[idx]
        print(f"  {i:2d}. Dimension {idx:4d}: weight = {weight:+.4f}")
    print("-" * 60)
    print()

    print("✅ Mechanistic interpretability example complete!")
    print()
    print("Key takeaways:")
    print("  - Linear probes can extract interpretable features from hidden states")
    print("  - The weights show which dimensions are predictive of the target")
    print("  - This technique scales to real models with transformer hooks")
    print()
    print("Next steps:")
    print("  - Extend this to record activations from real LLM calls")
    print("  - Try different probe targets (e.g., task success, agent strategy)")
    print("  - Visualize activation patterns across different agent behaviors")
    print("=" * 80)


def main() -> None:
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Mechanistic interpretability example with linear probing")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
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
        "--save-cache",
        type=Path,
        help="Path to save cached activations",
    )
    parser.add_argument(
        "--load-cache",
        type=Path,
        help="Path to load cached activations from",
    )

    args = parser.parse_args()

    # Determine cache behavior
    cache_path = args.save_cache or args.load_cache
    load_cache = args.load_cache is not None

    asyncio.run(
        run_pedagogical_example(
            n_samples=args.n_samples,
            hidden_dim=args.hidden_dim,
            test_split=args.test_split,
            cache_path=cache_path,
            load_cache=load_cache,
        )
    )


if __name__ == "__main__":
    main()
