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
import random
import threading

import ares
from ares import presets
from ares.contrib import mech_interp
from ares.environments import twenty_questions
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import transformer_lens

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_NEW_TOKENS = 64
ENV_NAME = "20q-steered"
N_EPISODES = 20  # Episodes per (pooling, alpha) condition
MAX_STEPS_PER_EPISODE = 25
ALPHAS = [0.5, 1.0, 2.0, 4.0]

AGENT_SYSTEM_PROMPT = "Try to keep your response under 30 words."

# Register a custom 20q preset with the system prompt for the agent.
ares.register_preset(
    ENV_NAME,
    presets.TwentyQuestionsSpec(system_prompt=AGENT_SYSTEM_PROMPT),
)

# Token-position pooling strategies for computing the steering vector.
# Activations are now [1, prompt_len + n_generated, d_model] with prompt_len saved.
# "last-prompt": last prompt token (decision point before generation)
# "mean-prompt": mean over all prompt tokens
# "first-generated": first generated token (start of model's response)
POOLING_STRATEGIES = ["last-prompt"]

# Inspect mode: print model completions side-by-side with oracle responses at t*.
INSPECT_MODE = False

SEED = 42
TRAIN_RATIO = 0.8

DATA_DIR = pathlib.Path("outputs/20q_data")
PROBE_RESULTS_PATH = pathlib.Path("outputs/20q_probing_results/results.json")
OUTPUT_DIR = pathlib.Path("outputs/deterministic_20q_steering_results")

# Minimum samples per class at target step in training set.
MIN_CLASS_SAMPLES = 5

# Serialize model loading across threads.
_MODEL_LOAD_LOCK = threading.Lock()


def _precompute_secret_words(n_episodes: int) -> dict[int, str]:
    """Pre-compute a deterministic secret word for each episode index.

    Uses a local Random instance so this is safe to call from any thread
    without affecting global random state.
    """
    rng = random.Random(SEED)
    objects = twenty_questions.DEFAULT_OBJECT_LIST
    return {ep_idx: rng.choice(objects) for ep_idx in range(n_episodes)}


# ---------------------------------------------------------------------------
# Helpers (shared with Phase 1)
# ---------------------------------------------------------------------------


def _pool_activation(activation: torch.Tensor, method: str, prompt_len: int = 0) -> np.ndarray:
    """Convert a [1, seq_len, d_model] tensor to a [d_model] numpy vector.

    Supported methods:
      - "last-prompt":      last prompt token (decision point before generation)
      - "mean-prompt":      mean over all prompt tokens
      - "first-generated":  first generated token (start of model's response)
      - "mean":             mean over all token positions
      - "last-k":           mean over the last k token positions (e.g. "last-1")
    """
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
    if method == "first-generated":
        if prompt_len <= 0:
            raise ValueError("prompt_len required for 'first-generated' pooling")
        if prompt_len >= arr.shape[1]:
            return arr[0, -1, :]  # Fallback if no generated tokens.
        return arr[0, prompt_len, :]
    if method == "mean":
        return arr[0].mean(axis=0)
    if method.startswith("last-"):
        k = int(method.split("-", 1)[1])
        seq_len = arr.shape[1]
        k = min(k, seq_len)
        return arr[0, -k:, :].mean(axis=0)
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


def _select_target_step(
    probe_results: dict,
    episodes: list[dict],
    train_eps: set[int],
    min_class_samples: int = MIN_CLASS_SAMPLES,
) -> int:
    """Auto-select target step t*: high probe accuracy AND high invalid rate.

    Picks the step that maximizes (probe_accuracy * invalid_rate) among steps
    with probe accuracy >= 0.85 and at least *min_class_samples* of each class
    in the training set, ensuring we can compute a reliable steering vector.
    """
    step_accuracies = probe_results["step_accuracies"]
    if not step_accuracies:
        raise ValueError("No step accuracies in probe results.")

    # Compute per-step invalid rates from training episodes.
    step_counts: dict[int, int] = {}
    step_invalid: dict[int, int] = {}
    for ep in episodes:
        if ep["episode_idx"] not in train_eps:
            continue
        for step in ep["steps"]:
            s = step["step_idx"]
            step_counts[s] = step_counts.get(s, 0) + 1
            step_invalid[s] = step_invalid.get(s, 0) + step["is_invalid"]

    print("\n  Step selection (train set):")
    print(
        f"  {'step':>4s}  {'n_total':>7s}  {'n_inv':>5s}  {'n_val':>5s}  "
        f"{'inv_rate':>8s}  {'probe_acc':>9s}  {'score':>6s}"
    )
    candidates: list[tuple[float, int]] = []
    for s_str, acc in sorted(step_accuracies.items(), key=lambda x: int(x[0])):
        s = int(s_str)
        n = step_counts.get(s, 0)
        n_inv = step_invalid.get(s, 0)
        n_val = n - n_inv
        inv_rate = n_inv / n if n > 0 else 0.0
        score = acc * inv_rate
        marker = ""
        if acc >= 0.85 and inv_rate >= 0.3 and n_val >= min_class_samples and n_inv >= min_class_samples:
            candidates.append((score, s))
            marker = " <--"
        print(f"  {s:4d}  {n:7d}  {n_inv:5d}  {n_val:5d}  {inv_rate:8.3f}  {acc:9.3f}  {score:6.3f}{marker}")

    if not candidates:
        # Fallback: pick highest accuracy step.
        best_step = max(step_accuracies, key=lambda s: (step_accuracies[s], -int(s)))
        return int(best_step)

    # Pick highest score, break ties by lowest step index.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


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
            prompt_len = step.get("prompt_len", 0)
            feature = _pool_activation(activation, pooling, prompt_len=prompt_len)
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


def compute_per_step_steering_vectors(
    episodes: list[dict],
    train_eps: set[int],
    pooling: str,
    min_class_samples: int,
) -> dict[int, np.ndarray]:
    """Compute a steering vector for every step that has enough samples of both classes.

    Returns dict mapping step_idx -> steering vector (v_valid - v_invalid).
    """
    # Gather features per step.
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

    # Compute steering vector for each step with enough samples.
    all_steps = sorted(set(step_valid.keys()) | set(step_invalid.keys()))
    vectors: dict[int, np.ndarray] = {}

    print(f"\n  Per-step steering vectors ({pooling}):")
    print(f"  {'step':>4s}  {'n_valid':>7s}  {'n_inv':>5s}  {'norm':>8s}  {'status'}")
    for s in all_steps:
        n_valid = len(step_valid.get(s, []))
        n_invalid = len(step_invalid.get(s, []))
        if n_valid >= min_class_samples and n_invalid >= min_class_samples:
            v_valid = np.stack(step_valid[s]).mean(axis=0)
            v_invalid = np.stack(step_invalid[s]).mean(axis=0)
            sv = v_valid - v_invalid
            vectors[s] = sv
            print(f"  {s:4d}  {n_valid:7d}  {n_invalid:5d}  {np.linalg.norm(sv):8.4f}  OK")
        else:
            print(f"  {s:4d}  {n_valid:7d}  {n_invalid:5d}  {'---':>8s}  skipped (insufficient samples)")

    print(f"  Computed steering vectors for {len(vectors)}/{len(all_steps)} steps")
    return vectors


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
    """Create a hook that steers only the last prompt token on the first forward pass.

    During model.generate() with KV caching:
      - First forward pass processes the full prompt: activation shape [1, prompt_len, d_model]
      - Subsequent passes process one new token each: activation shape [1, 1, d_model]

    We add the steering vector only to the last prompt token (position -1 on
    the first forward pass), which is the "decision point" right before
    generation begins.  Subsequent forward passes are left untouched.
    """
    fired = False

    def hook(activation, hook):  # noqa: ARG001
        nonlocal fired
        if not fired:
            # First forward pass: steer only the last prompt token.
            activation[:, -1, :] += alpha * sv
            fired = True
        return activation

    return hook


async def _run_steered_episodes_for_device(
    device: str,
    work_queue: "queue.Queue[tuple[str, float | None, int]]",
    middle_layer: int,
    per_step_vectors: dict[int, np.ndarray],
    max_steps_per_episode: int,
    secret_words: dict[int, str],
) -> list[dict]:
    """Load model on *device*, run steered episodes from *work_queue*.

    Each work item is (pooling, alpha, episode_idx). alpha=None means baseline.
    Steers at every step that has a vector in *per_step_vectors*.
    *secret_words* maps ep_idx -> word, ensuring the same word across conditions.
    Returns list of per-episode result dicts with full step data.
    """
    print(f"[{device}] Loading {MODEL_NAME}...")
    torch.cuda.set_device(device)

    with _MODEL_LOAD_LOCK:
        model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.bfloat16)
    model = model.to(device)

    client = mech_interp.HookedTransformerLLMClient(
        model=model,
        max_new_tokens=MAX_NEW_TOKENS,
        generation_kwargs={"verbose": False},
    )

    hook_name = f"blocks.{middle_layer}.hook_resid_post"
    # TODO: Cleanup hook_point usage, and add good devX interface for adding hooks in client
    hook_point = model.hook_dict[hook_name]

    # Pre-convert per-step steering vectors to device tensors.
    sv_tensors: dict[int, torch.Tensor] = {
        step: torch.tensor(sv, dtype=torch.bfloat16, device=device) for step, sv in per_step_vectors.items()
    }

    results: list[dict] = []

    try:
        while True:
            try:
                pooling, alpha, ep_idx = work_queue.get_nowait()
            except queue.Empty:
                break

            condition = f"{pooling}/alpha={alpha}" if alpha is not None else f"{pooling}/baseline"

            # Deterministic torch seed per (episode, device) — each device has
            # its own generator so no cross-thread interference.
            episode_seed = SEED + ep_idx
            torch.cuda.manual_seed(episode_seed)

            step_log: list[tuple[int, str, str, bool, bool]] = []

            async with ares.make(ENV_NAME) as env:
                ts = await env.reset()
                # Override the secret word with our pre-computed one so that
                # baseline and steered runs for the same ep_idx use the same word.
                # TODO: Clean up seeding logic in 20Q env for reproducibility
                env._hidden_object = secret_words[ep_idx]
                step_idx = 0

                while (not ts.last()) and (step_idx < max_steps_per_episode):
                    # Install steering hook if we have a vector for this step.
                    steer_active = False
                    if alpha is not None and step_idx in sv_tensors:
                        hook_fn = make_steer_hook(sv_tensors[step_idx], alpha)
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

                    # Extract model's generated question.
                    model_question = action.data[0].content.strip() if action.data else ""

                    is_invalid = _is_invalid_answer(oracle_answer)
                    step_log.append((step_idx, model_question, oracle_answer, steer_active, is_invalid))

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
                "n_steps": step_idx,
                "steps": [
                    {
                        "step_idx": s_idx,
                        "model_question": question,
                        "oracle_answer": oracle,
                        "steered": steered,
                        "is_invalid": is_invalid,
                    }
                    for s_idx, question, oracle, steered, is_invalid in step_log
                ],
            }
            results.append(result)

            n_invalid = sum(1 for _, _, _, _, inv in step_log if inv)
            tqdm.write(f"  [{device}] {condition:22s} ep={ep_idx:03d} DONE  invalid={n_invalid}/{len(step_log)} steps")

    finally:
        hook_point.remove_hooks("fwd")

    del model, client
    _free_gpu_memory()

    return results


def _run_device_worker(
    device: str,
    work_queue: "queue.Queue[tuple[str, float | None, int]]",
    middle_layer: int,
    per_step_vectors: dict[int, np.ndarray],
    max_steps_per_episode: int,
    secret_words: dict[int, str],
) -> list[dict]:
    """Run steered episodes in a dedicated event loop (for thread-based parallelism)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            _run_steered_episodes_for_device(
                device, work_queue, middle_layer, per_step_vectors, max_steps_per_episode, secret_words
            )
        )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Inspect mode: sequential, single-GPU, readable output
# ---------------------------------------------------------------------------


async def _run_inspect_mode(
    middle_layer: int,
    per_step_vectors: dict[int, np.ndarray],
    max_steps_per_episode: int,
    n_episodes: int,
    alphas: list[float],
    secret_words: dict[int, str],
) -> list[dict]:
    """Run episodes sequentially on a single GPU with grouped, readable output.

    Steers at *every* step that has a steering vector (not just one target step).
    For each episode, runs a baseline (no steering) and then one run per alpha,
    printing model completions and oracle responses at every step side-by-side.
    Output is printed to stdout and saved to a log file.
    """
    device = _get_devices()[0]

    inspect_dir = OUTPUT_DIR / f"inspect_{n_episodes}ep"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    log_path = inspect_dir / "inspect_log.txt"

    def _log(msg: str = "") -> None:
        print(msg)
        log_lines.append(msg)

    log_lines: list[str] = []

    _log(f"[inspect] Loading {MODEL_NAME} on {device}...")

    model = transformer_lens.HookedTransformer.from_pretrained(MODEL_NAME, device="cpu", dtype=torch.bfloat16)
    model = model.to(device)
    client = mech_interp.HookedTransformerLLMClient(
        model=model,
        max_new_tokens=MAX_NEW_TOKENS,
        generation_kwargs={"verbose": False},
    )

    hook_name = f"blocks.{middle_layer}.hook_resid_post"
    hook_point = model.hook_dict[hook_name]

    pooling = POOLING_STRATEGIES[0]

    # Pre-convert per-step steering vectors to device tensors.
    sv_tensors: dict[int, torch.Tensor] = {
        step: torch.tensor(sv, dtype=torch.bfloat16, device=device) for step, sv in per_step_vectors.items()
    }
    steered_steps = sorted(sv_tensors.keys())
    _log(f"[inspect] Steering at {len(steered_steps)} steps: {steered_steps}")

    all_results: list[dict] = []

    try:
        for ep_idx in range(n_episodes):
            _log(f"\n{'=' * 70}")
            _log(f"  Episode {ep_idx}")
            _log(f"{'=' * 70}")

            # Conditions to run: baseline (alpha=None) + each alpha.
            conditions: list[tuple[str, float | None]] = [("baseline", None)]
            for a in alphas:
                conditions.append((f"alpha={a}", a))

            # Store per-condition step logs for grouped printing.
            condition_logs: list[tuple[str, list[tuple[int, str, str, bool]]]] = []

            for label, alpha in conditions:
                step_log: list[tuple[int, str, str, bool]] = []  # (step_idx, question, oracle, steered)

                # Deterministic torch seed per episode.
                episode_seed = SEED + ep_idx
                torch.cuda.manual_seed(episode_seed)

                async with ares.make(ENV_NAME) as env:
                    ts = await env.reset()
                    # Override the secret word so all conditions for the same
                    # ep_idx use the same word.
                    env._hidden_object = secret_words[ep_idx]
                    step_idx = 0

                    while (not ts.last()) and (step_idx < max_steps_per_episode):
                        # Install steering hook if we have a vector for this step.
                        steer_active = False
                        if alpha is not None and step_idx in sv_tensors:
                            hook_fn = make_steer_hook(sv_tensors[step_idx], alpha)
                            hook_point.add_hook(hook_fn)
                            steer_active = True

                        assert ts.observation is not None
                        action = await client(ts.observation)

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

                        model_question = action.data[0].content.strip() if action.data else ""

                        step_log.append((step_idx, model_question, oracle_answer, steer_active))
                        step_idx += 1

                condition_logs.append((label, step_log))

                result = {
                    "pooling": pooling,
                    "condition": f"{pooling}/{label}",
                    "alpha": alpha,
                    "episode_idx": ep_idx,
                    "n_steps": step_idx,
                    "steps": [
                        {
                            "step_idx": s_idx,
                            "model_question": question,
                            "oracle_answer": oracle,
                            "steered": steered,
                            "is_invalid": _is_invalid_answer(oracle),
                        }
                        for s_idx, question, oracle, steered in step_log
                    ],
                }
                all_results.append(result)

            # Print grouped summary for this episode.
            for label, step_log in condition_logs:
                steered_at = " (steered at all steps)" if label != "baseline" else " (no steering)"
                _log(f"\n  --- {label}{steered_at} ---")
                for s_idx, question, oracle, steered in step_log:
                    marker = " <<< STEERED" if steered else ""
                    _log(f"    step {s_idx:2d}: MODEL: {question}")
                    _log(f"              ORACLE: {oracle}{marker}")

    finally:
        hook_point.remove_hooks("fwd")

    del model, client
    _free_gpu_memory()

    # Write log file.
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nInspect log saved to {log_path}")

    return all_results


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
    target_step = _select_target_step(probe_results, episodes, train_eps)
    step_acc = probe_results["step_accuracies"][str(target_step)]
    print(f"Auto-selected target step t* = {target_step} (probe accuracy: {step_acc:.3f})")

    # Compute steering vectors.
    pooling = POOLING_STRATEGIES[0]

    # Compute a separate steering vector for every step with enough samples.
    per_step_vectors = compute_per_step_steering_vectors(episodes, train_eps, pooling, MIN_CLASS_SAMPLES)

    # --- Online stage ---
    alphas = [1.0, 2.0, 4.0] if INSPECT_MODE else ALPHAS
    n_episodes = 20 if INSPECT_MODE else N_EPISODES

    # Pre-compute secret words so every condition for the same ep_idx
    # gets the same word (thread-safe, no global random state).
    secret_words = _precompute_secret_words(n_episodes)
    print(f"\nSecret words: { {i: secret_words[i] for i in range(min(5, n_episodes))} } ...")

    if INSPECT_MODE:
        # Sequential, single-GPU path with per-step steering.
        print("\nINSPECT MODE: per-step steering, sequential on single GPU")
        print(f"POOLING: {pooling}")
        print(f"ALPHAS: {alphas}")
        print(f"EPISODES: {n_episodes}")
        print(f"STEERED STEPS: {sorted(per_step_vectors.keys())}")
        print(f"{'=' * 70}\n")

        all_results = await _run_inspect_mode(
            middle_layer=middle_layer,
            per_step_vectors=per_step_vectors,
            max_steps_per_episode=MAX_STEPS_PER_EPISODE,
            n_episodes=n_episodes,
            alphas=alphas,
            secret_words=secret_words,
        )
        failed_devices: list[str] = []
    else:
        # Multi-GPU, queue/thread pool path with per-step steering.
        devices = _get_devices()

        # Build work queue: (pooling, alpha, episode_idx) for all combinations.
        work_queue: queue.Queue[tuple[str, float | None, int]] = queue.Queue()
        for ep_idx in range(n_episodes):
            work_queue.put((pooling, None, ep_idx))
        for alpha in alphas:
            for ep_idx in range(n_episodes):
                work_queue.put((pooling, alpha, ep_idx))

        total_episodes = work_queue.qsize()
        n_devices = min(len(devices), total_episodes)
        devices = devices[:n_devices]

        n_steered = len(alphas) * n_episodes
        print(f"\nDEVICES: {devices}")
        print(f"POOLING: {pooling}")
        print(f"ALPHAS: {alphas}")
        print(f"EPISODES PER CONDITION: {n_episodes}")
        print(f"TOTAL EPISODES: {total_episodes} ({n_episodes} baseline + {n_steered} steered)")
        print(f"STEERED STEPS: {sorted(per_step_vectors.keys())}")
        for s, sv in sorted(per_step_vectors.items()):
            print(f"  step {s:2d} steering vector norm: {np.linalg.norm(sv):.4f}")
        print(f"{'=' * 70}\n")

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            futures = [
                loop.run_in_executor(
                    executor,
                    _run_device_worker,
                    device,
                    work_queue,
                    middle_layer,
                    per_step_vectors,
                    MAX_STEPS_PER_EPISODE,
                    secret_words,
                )
                for device in devices
            ]
            worker_results = await asyncio.gather(*futures, return_exceptions=True)

        # Collect results.
        all_results: list[dict] = []
        failed_devices = []
        for device, result in zip(devices, worker_results, strict=True):
            if isinstance(result, BaseException):
                print(f"\n[{device}] FAILED: {result}")
                failed_devices.append(device)
                continue
            all_results.extend(result)

    # Aggregate invalid rates across all steps per condition.
    def _condition_stats(results: list[dict]) -> dict:
        all_steps = [s for r in results for s in r.get("steps", [])]
        n_total = len(all_steps)
        n_invalid = sum(1 for s in all_steps if s["is_invalid"])
        return {
            "n_episodes": len(results),
            "n_steps": n_total,
            "n_invalid": n_invalid,
            "invalid_rate": n_invalid / n_total if n_total > 0 else 0.0,
        }

    baseline_eps = [r for r in all_results if r["alpha"] is None]
    baseline_stats = _condition_stats(baseline_eps)

    all_condition_results: dict[str, dict[str, dict]] = {}
    for p in POOLING_STRATEGIES:
        cond_results: dict[str, dict] = {"baseline": baseline_stats}
        for alpha in alphas:
            label = f"alpha={alpha}"
            ep_results = [r for r in all_results if r["pooling"] == p and r["alpha"] == alpha]
            cond_results[label] = _condition_stats(ep_results)
        all_condition_results[p] = cond_results

    # Print summary.
    print(f"\n{'=' * 70}")
    print("STEERING RESULTS (invalid rate across all steps)")
    print(f"{'=' * 70}")
    print(
        f"  {'baseline':22s}:  invalid={baseline_stats['n_invalid']:3d}/{baseline_stats['n_steps']:3d}  "
        f"rate={baseline_stats['invalid_rate']:.1%}"
    )
    for p in POOLING_STRATEGIES:
        for alpha in alphas:
            label = f"alpha={alpha}"
            stats = all_condition_results[p][label]
            print(
                f"  {p + '/' + label:22s}:  invalid={stats['n_invalid']:3d}/{stats['n_steps']:3d}  "
                f"rate={stats['invalid_rate']:.1%}"
            )

    # Save results.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_data = {
        "model_name": model_name,
        "middle_layer": middle_layer,
        "target_step": target_step,
        "pooling_strategies": POOLING_STRATEGIES,
        "per_step_vector_norms": {s: float(np.linalg.norm(sv)) for s, sv in per_step_vectors.items()},
        "alphas": alphas,
        "n_episodes_per_condition": n_episodes,
        "conditions": all_condition_results,
        "secret_words": secret_words,
        "episodes": all_results,
        "failed_devices": failed_devices,
    }
    subdir = f"inspect_{n_episodes}ep" if INSPECT_MODE else f"results_{n_episodes}ep"
    results_path = OUTPUT_DIR / subdir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2))
    print(f"\nResults saved to {results_path}")

    # Plot.
    plot_output = results_path.parent / "steering_effectiveness.png"
    plot_steering_effectiveness(
        all_condition_results,
        target_step=target_step,
        middle_layer=middle_layer,
        model_name=model_name,
        output_path=plot_output,
    )

    if failed_devices:
        print(f"\nWARNING: {len(failed_devices)} device(s) failed: {failed_devices}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
