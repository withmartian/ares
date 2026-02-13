#!/usr/bin/env python

# # Phase 2: CAA Steering for 20Q Invalid Questions
#
# Tests whether the linear direction identified by Phase 1 probing is *causal*:
# can we steer the model toward valid questions by adding a contrastive
# activation vector (CAA) at the target step?
#
# ## Offline stage (no GPU needed beyond loading saved data):
#   1. Load saved episodes and Phase 1 probe results
#   2. Reproduce the same train/test split as Phase 1
#   3. Auto-select target step t* (highest probe accuracy with both classes >= 5 in train)
#   4. Compute steering vectors from training activations at t* for each pooling strategy
#
# ## Online stage (requires GPU + oracle API):
#   5. For each pooling strategy, run new episodes with steering at step t*
#      for several alpha values
#   6. Record invalid question rates per (pooling, alpha) condition
#   7. Plot steering effectiveness across pooling strategies
#
# ## Requirements
#   uv sync --extra transformer-lens
#   # CHAT_COMPLETION_API_KEY must be set in .env (for the oracle)
#   # Phase 1 data must exist in outputs/20q_data/ and outputs/20q_probing_results/
#
# ## Run
#   uv run --no-sync python examples/20q_case_study/phase2_steer.py

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import pathlib
import queue
import threading

import ares
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template
import matplotlib.pyplot as plt
import numpy as np
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
N_EPISODES = 20  # Episodes per (pooling, alpha) condition
MAX_STEPS_PER_EPISODE = 25
ALPHAS = [0.5, 1.0, 2.0, 4.0]

# Token-position pooling strategies for computing the steering vector.
# "last-k" averages the last k token positions; "mean" averages all positions.
# Note: saved activations are [1, 1, d_model] (last generated token only),
# so only "last-1" is meaningful with current data.
POOLING_STRATEGIES = ["last-1"]

SEED = 42
TRAIN_RATIO = 0.8

DATA_DIR = pathlib.Path("outputs/20q_data")
PROBE_RESULTS_PATH = pathlib.Path("outputs/20q_probing_results/results.json")
OUTPUT_DIR = pathlib.Path("outputs/20q_steering_results")

# Minimum samples per class at target step in training set.
MIN_CLASS_SAMPLES = 5

# Serialize model loading across threads.
_MODEL_LOAD_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers (shared with Phase 1)
# ---------------------------------------------------------------------------


def _pool_activation(activation: torch.Tensor, method: str) -> np.ndarray:
    """Convert a [1, seq_len, d_model] tensor to a [d_model] numpy vector.

    Supported methods:
      - "mean":   mean over all token positions
      - "last-k": mean over the last k token positions (e.g. "last-1", "last-4")
    """
    arr = activation.numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 activation, got shape={arr.shape}")
    if method == "mean":
        return arr[0].mean(axis=0)
    if method.startswith("last-"):
        k = int(method.split("-", 1)[1])
        seq_len = arr.shape[1]
        k = min(k, seq_len)  # Don't exceed available tokens.
        return arr[0, -k:, :].mean(axis=0)
    if method == "last":
        return arr[0, -1, :]
    raise ValueError(f"Unknown pooling method: {method}")


def load_episodes(data_dir: pathlib.Path) -> tuple[list[dict], dict]:
    """Load all episode .pt files and metadata from *data_dir*."""
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json in {data_dir}. Run collect_20q_data.py first.")
    metadata = json.loads(metadata_path.read_text())
    ep_files = sorted(data_dir.glob("episode_*.pt"))
    if not ep_files:
        raise FileNotFoundError(f"No episode_*.pt files in {data_dir}. Run collect_20q_data.py first.")
    episodes = [torch.load(ep_file, weights_only=False) for ep_file in ep_files]
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes, metadata


def train_test_split_by_episode(
    episode_indices: list[int],
    train_ratio: float,
) -> tuple[set[int], set[int]]:
    """Split episode indices into train/test sets (same logic and seed as Phase 1)."""
    sorted_indices = sorted(episode_indices)
    n_train = max(1, int(len(sorted_indices) * train_ratio))
    rng = np.random.RandomState(SEED)
    rng.shuffle(sorted_indices)  # type: ignore[arg-type]
    train_eps = set(sorted_indices[:n_train])
    test_eps = set(sorted_indices[n_train:])
    print(f"Split: {len(train_eps)} train episodes, {len(test_eps)} test episodes")
    return train_eps, test_eps


# ---------------------------------------------------------------------------
# Offline stage: compute steering vectors
# ---------------------------------------------------------------------------


def _select_target_step(probe_results: dict) -> int:
    """Auto-select target step t*: highest probe accuracy step."""
    step_accuracies = probe_results["step_accuracies"]
    if not step_accuracies:
        raise ValueError("No step accuracies in probe results.")
    # Find step with highest accuracy (break ties by lowest step index).
    best_step = max(step_accuracies, key=lambda s: (step_accuracies[s], -int(s)))
    return int(best_step)


def compute_steering_vector(
    episodes: list[dict],
    train_eps: set[int],
    target_step: int,
    pooling: str,
    min_class_samples: int,
) -> np.ndarray:
    """Compute steering vector from training activations at *target_step*.

    Returns v_valid - v_invalid (shape [d_model]).
    """
    valid_features: list[np.ndarray] = []
    invalid_features: list[np.ndarray] = []

    for ep in episodes:
        if ep["episode_idx"] not in train_eps:
            continue
        for step in ep["steps"]:
            if step["step_idx"] != target_step:
                continue
            activation = step.get("activation")
            if activation is None:
                continue
            feature = _pool_activation(activation, pooling)
            if step["is_invalid"]:
                invalid_features.append(feature)
            else:
                valid_features.append(feature)

    n_valid = len(valid_features)
    n_invalid = len(invalid_features)
    print(f"  [{pooling:6s}] step {target_step}: {n_valid} valid, {n_invalid} invalid (train)")

    if n_valid < min_class_samples or n_invalid < min_class_samples:
        raise ValueError(
            f"Insufficient samples at step {target_step}: "
            f"{n_valid} valid, {n_invalid} invalid (need >= {min_class_samples} each)"
        )

    v_valid = np.stack(valid_features).mean(axis=0)
    v_invalid = np.stack(invalid_features).mean(axis=0)
    steering_vector = v_valid - v_invalid

    norm = np.linalg.norm(steering_vector)
    print(f"  [{pooling:6s}] steering vector norm: {norm:.4f}")
    return steering_vector


# ---------------------------------------------------------------------------
# Online stage: run steered episodes
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


def make_steer_hook(sv: torch.Tensor, alpha: float):
    """Create a TransformerLens hook that adds alpha * steering_vector to all positions."""

    def hook(activation, hook):  # noqa: ARG001
        activation[:, :, :] += alpha * sv
        return activation

    return hook


async def _run_steered_episodes_for_device(
    device: str,
    work_queue: "queue.Queue[tuple[str, float | None, int]]",
    target_step: int,
    middle_layer: int,
    steering_vectors: dict[str, np.ndarray],
    max_steps_per_episode: int,
) -> list[dict]:
    """Load model on *device*, run steered episodes from *work_queue*.

    Each work item is (pooling, alpha, episode_idx). alpha=None means baseline.
    Returns list of per-episode result dicts.
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

    hook_name = f"blocks.{middle_layer}.hook_resid_post"
    hook_point = model.hook_dict[hook_name]

    # Pre-convert all steering vectors to device tensors.
    sv_tensors: dict[str, torch.Tensor] = {
        pooling: torch.tensor(sv, dtype=torch.bfloat16, device=device) for pooling, sv in steering_vectors.items()
    }

    results: list[dict] = []

    try:
        while True:
            try:
                pooling, alpha, ep_idx = work_queue.get_nowait()
            except queue.Empty:
                break

            condition = f"{pooling}/alpha={alpha}" if alpha is not None else f"{pooling}/baseline"

            # Track whether the question at step t* was invalid.
            target_step_invalid: int | None = None
            target_step_oracle: str = ""

            async with ares.make(ENV_NAME) as env:
                ts = await env.reset()
                step_idx = 0

                while (not ts.last()) and (step_idx < max_steps_per_episode):
                    # Install steering hook only at target step.
                    steer_active = False
                    if step_idx == target_step and alpha is not None:
                        hook_fn = make_steer_hook(sv_tensors[pooling], alpha)
                        hook_point.add_hook(hook_fn)
                        steer_active = True

                    assert ts.observation is not None
                    action = await client(ts.observation)

                    # Remove steering hook after generation.
                    if steer_active:
                        hook_point.remove_hooks("fwd")

                    prev_history_len = len(env._conversation_history)
                    ts = await env.step(action)

                    # Extract oracle answer.
                    oracle_answer = ""
                    if len(env._conversation_history) > prev_history_len:
                        last_entry = env._conversation_history[-1]
                        if last_entry.startswith("A:"):
                            oracle_answer = last_entry

                    # Record result at target step.
                    if step_idx == target_step:
                        target_step_invalid = int(_is_invalid_answer(oracle_answer))
                        target_step_oracle = oracle_answer

                    steered = "*" if steer_active else " "
                    tqdm.write(
                        f"    [{device}] {condition:22s} ep={ep_idx:03d} "
                        f"step={step_idx:2d}/{max_steps_per_episode}{steered} "
                        f"oracle={oracle_answer[:60]}"
                    )
                    step_idx += 1

            result = {
                "pooling": pooling,
                "condition": condition,
                "alpha": alpha,
                "episode_idx": ep_idx,
                "target_step": target_step,
                "target_step_invalid": target_step_invalid,
                "target_step_oracle": target_step_oracle,
                "n_steps": step_idx,
            }
            results.append(result)

            tqdm.write(
                f"  [{device}] {condition:22s} ep={ep_idx:03d} DONE  "
                f"invalid@t*={target_step_invalid}  oracle={target_step_oracle[:60]}"
            )

    finally:
        hook_point.remove_hooks("fwd")

    del model, tokenizer, client
    _free_gpu_memory()

    return results


def _run_device_worker(
    device: str,
    work_queue: "queue.Queue[tuple[str, float | None, int]]",
    target_step: int,
    middle_layer: int,
    steering_vectors: dict[str, np.ndarray],
    max_steps_per_episode: int,
) -> list[dict]:
    """Run steered episodes in a dedicated event loop (for thread-based parallelism)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _run_steered_episodes_for_device(
                device, work_queue, target_step, middle_layer, steering_vectors, max_steps_per_episode
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_steering_effectiveness(
    all_condition_results: dict[str, dict[str, dict]],
    target_step: int,
    middle_layer: int,
    model_name: str,
    output_path: pathlib.Path,
) -> None:
    """Plot grouped bar chart of invalid question rate per condition, grouped by pooling."""
    pooling_strategies = list(all_condition_results.keys())
    # Conditions within each pooling group (baseline + alphas).
    sample_conditions = list(all_condition_results[pooling_strategies[0]].keys())
    n_poolings = len(pooling_strategies)
    n_conditions = len(sample_conditions)

    fig, ax = plt.subplots(figsize=(max(12, n_poolings * 3), 6))

    x = np.arange(n_conditions)
    bar_width = 0.8 / n_poolings
    colors = plt.cm.tab10(np.linspace(0, 1, n_poolings))  # type: ignore[attr-defined]

    for i, pooling in enumerate(pooling_strategies):
        cond_results = all_condition_results[pooling]
        rates = [cond_results[c]["invalid_rate"] * 100 for c in sample_conditions]
        counts = [f"{cond_results[c]['n_invalid']}/{cond_results[c]['n_episodes']}" for c in sample_conditions]
        offset = (i - n_poolings / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, rates, bar_width * 0.9, label=pooling, color=colors[i])

        for bar, count in zip(bars, counts, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                count,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(sample_conditions)

    short_name = model_name.split("/")[-1]
    ax.set_xlabel("Condition")
    ax.set_ylabel("Invalid question rate at step t* (%)")
    ax.set_title(
        f"Steering effectiveness by token pooling: CAA on resid_post (layer {middle_layer})\n"
        f"{short_name}, target step={target_step}"
    )
    ax.legend(title="Pooling", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run() -> None:
    # --- Offline stage ---
    print(f"\n{'=' * 70}")
    print("PHASE 2: CAA Steering for 20Q Invalid Questions")
    print(f"{'=' * 70}")

    # Load Phase 1 data and results.
    print(f"\nLoading episodes from {DATA_DIR}...")
    episodes, metadata = load_episodes(DATA_DIR)

    print(f"Loading probe results from {PROBE_RESULTS_PATH}...")
    probe_results = json.loads(PROBE_RESULTS_PATH.read_text())

    middle_layer = metadata["middle_layer"]
    model_name = metadata["model_name"]

    # Reproduce Phase 1 train/test split.
    all_ep_indices = sorted({ep["episode_idx"] for ep in episodes})
    train_eps, _test_eps = train_test_split_by_episode(all_ep_indices, TRAIN_RATIO)

    # Select target step.
    target_step = _select_target_step(probe_results)
    step_acc = probe_results["step_accuracies"][str(target_step)]
    print(f"Auto-selected target step t* = {target_step} (probe accuracy: {step_acc:.3f})")

    # Compute a steering vector for each pooling strategy.
    print(f"\nComputing steering vectors for {len(POOLING_STRATEGIES)} pooling strategies...")
    steering_vectors: dict[str, np.ndarray] = {}
    for pooling in POOLING_STRATEGIES:
        steering_vectors[pooling] = compute_steering_vector(
            episodes, train_eps, target_step, pooling, MIN_CLASS_SAMPLES
        )

    # --- Online stage ---
    devices = _get_devices()

    # Build work queue: (pooling, alpha, episode_idx) for all combinations.
    # Baseline episodes are shared across pooling strategies (no steering applied),
    # so we only need one set of baseline episodes.
    work_queue: queue.Queue[tuple[str, float | None, int]] = queue.Queue()
    baseline_pooling = POOLING_STRATEGIES[0]  # Baseline doesn't use steering, pick any.
    for ep_idx in range(N_EPISODES):
        work_queue.put((baseline_pooling, None, ep_idx))
    for pooling in POOLING_STRATEGIES:
        for alpha in ALPHAS:
            for ep_idx in range(N_EPISODES):
                work_queue.put((pooling, alpha, ep_idx))

    total_episodes = work_queue.qsize()
    n_devices = min(len(devices), total_episodes)
    devices = devices[:n_devices]

    n_steered = len(POOLING_STRATEGIES) * len(ALPHAS) * N_EPISODES
    print(f"\nDEVICES: {devices}")
    print(f"POOLING STRATEGIES: {POOLING_STRATEGIES}")
    print(f"ALPHAS: {ALPHAS}")
    print(f"EPISODES PER CONDITION: {N_EPISODES}")
    print(f"TOTAL EPISODES: {total_episodes} ({N_EPISODES} baseline + {n_steered} steered)")
    print(f"TARGET STEP: {target_step}")
    for pooling, sv in steering_vectors.items():
        print(f"  {pooling:6s} steering vector norm: {np.linalg.norm(sv):.4f}")
    print(f"{'=' * 70}\n")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = [
            loop.run_in_executor(
                executor,
                _run_device_worker,
                device,
                work_queue,
                target_step,
                middle_layer,
                steering_vectors,
                MAX_STEPS_PER_EPISODE,
            )
            for device in devices
        ]
        worker_results = await asyncio.gather(*futures, return_exceptions=True)

    # Collect results.
    all_results: list[dict] = []
    failed_devices: list[str] = []
    for device, result in zip(devices, worker_results, strict=True):
        if isinstance(result, BaseException):
            print(f"\n[{device}] FAILED: {result}")
            failed_devices.append(device)
            continue
        all_results.extend(result)

    # Extract baseline results (shared across all pooling strategies).
    baseline_episodes = [r for r in all_results if r["alpha"] is None]
    baseline_with_target = [r for r in baseline_episodes if r["target_step_invalid"] is not None]
    baseline_n_invalid = sum(r["target_step_invalid"] for r in baseline_with_target)
    baseline_n_total = len(baseline_with_target)
    baseline_stats = {
        "n_episodes": baseline_n_total,
        "n_invalid": baseline_n_invalid,
        "invalid_rate": baseline_n_invalid / baseline_n_total if baseline_n_total > 0 else 0.0,
    }

    # Aggregate by (pooling, condition).
    all_condition_results: dict[str, dict[str, dict]] = {}
    for pooling in POOLING_STRATEGIES:
        cond_results: dict[str, dict] = {"baseline": baseline_stats}
        for alpha in ALPHAS:
            label = f"alpha={alpha}"
            ep_results = [r for r in all_results if r["pooling"] == pooling and r["alpha"] == alpha]
            n_with_target = [r for r in ep_results if r["target_step_invalid"] is not None]
            n_invalid = sum(r["target_step_invalid"] for r in n_with_target)
            n_total = len(n_with_target)
            cond_results[label] = {
                "n_episodes": n_total,
                "n_invalid": n_invalid,
                "invalid_rate": n_invalid / n_total if n_total > 0 else 0.0,
            }
        all_condition_results[pooling] = cond_results

    # Print summary.
    print(f"\n{'=' * 70}")
    print("STEERING RESULTS")
    print(f"{'=' * 70}")
    print(
        f"  {'baseline':22s}:  invalid={baseline_stats['n_invalid']:2d}/{baseline_stats['n_episodes']:2d}  "
        f"rate={baseline_stats['invalid_rate']:.1%}"
    )
    for pooling in POOLING_STRATEGIES:
        for alpha in ALPHAS:
            label = f"alpha={alpha}"
            stats = all_condition_results[pooling][label]
            print(
                f"  {pooling + '/' + label:22s}:  invalid={stats['n_invalid']:2d}/{stats['n_episodes']:2d}  "
                f"rate={stats['invalid_rate']:.1%}"
            )

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model_name": model_name,
        "middle_layer": middle_layer,
        "target_step": target_step,
        "pooling_strategies": POOLING_STRATEGIES,
        "steering_vector_norms": {p: float(np.linalg.norm(sv)) for p, sv in steering_vectors.items()},
        "alphas": ALPHAS,
        "n_episodes_per_condition": N_EPISODES,
        "conditions": all_condition_results,
        "episodes": all_results,
        "failed_devices": failed_devices,
    }
    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_path}")

    # Plot.
    plot_steering_effectiveness(
        all_condition_results,
        target_step=target_step,
        middle_layer=middle_layer,
        model_name=model_name,
        output_path=OUTPUT_DIR / "steering_effectiveness.png",
    )

    if failed_devices:
        print(f"\nWARNING: {len(failed_devices)} device(s) failed: {failed_devices}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
