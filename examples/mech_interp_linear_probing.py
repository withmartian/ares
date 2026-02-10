#!/usr/bin/env python
# coding: utf-8

# # Residual Stream Linear Probe (Middle Layer)
#
# This script collects trajectories in ARES using a TransformerLens
# `HookedTransformer`, caches `resid_post` activations across all layers,
# and trains a linear probe to predict whether the episode will end in success
# (`final reward == 1`).
#
# ## Original notebook setup commands
# uv pip install -U numpy scikit-learn transformer-lens
# uv pip install -U transformer-lens
# uv pip install "transformers<4.41"
# uv pip install -U datasets pyarrow
# uv pip install --upgrade --force-reinstall "huggingface-hub>=0.34,<1.0" "transformers>=4.57.3"
# cd ../ && uv pip install -e .
# uv python install 3.12
# uv python pin 3.12

import asyncio
import shutil
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

import ares
from ares.contrib.mech_interp import ActivationCapture
from ares.contrib.mech_interp.hooked_transformer_client import (
    create_hooked_transformer_client_with_chat_template,
)


SEED = 42
np.random.seed(SEED)

# Model + client configuration
# Open-source coding-capable model (TransformerLens-compatible).
# Picked to be *small enough to run locally* while still being much more capable than GPT-2.
#
# Other good TL-supported options (tradeoffs: bigger = better but slower / more memory):
# - "Qwen/Qwen2.5-1.5B-Instruct"
# - "Qwen/Qwen2.5-7B-Instruct" (may be too big depending on your RAM)
# - "EleutherAI/pythia-1.4b-deduped"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "mps"  # Apple Silicon GPU. Use "cpu" if you hit MPS issues.
MAX_NEW_TOKENS = 128

# Data collection configuration
ENV_NAME = "sbv-mswea:0"
N_EPISODES = 30
MAX_STEPS_PER_EPISODE = 20

print(f"Loading {MODEL_NAME} on {DEVICE}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
client = create_hooked_transformer_client_with_chat_template(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    verbose=False,  # disable per-token tqdm spam
)

n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
middle_layer = n_layers // 2
print(f"n_layers={n_layers}, d_model={d_model}, middle_layer={middle_layer}")


@dataclass
class EpisodeRecord:
    features_middle: np.ndarray  # shape [d_model]
    label_success: int
    episode_idx: int
    n_steps: int


def _scalar_reward(x) -> float:
    """Convert reward payloads to a Python float in a robust way."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])


def _to_numpy_mean_over_tokens(tensor_like) -> np.ndarray:
    """Expected shape is [batch, pos, d_model] for resid_post; returns [d_model]."""
    arr = tensor_like.detach().cpu().numpy()
    if arr.ndim != 3:
        raise ValueError(f"Expected rank-3 resid_post, got shape={arr.shape}")
    return arr[0].mean(axis=0)


async def collect_probe_dataset(
    model,
    client,
    env_name: str,
    n_episodes: int,
    max_steps_per_episode: int,
    middle_layer: int,
):
    records = []

    for ep in range(n_episodes):
        async with ares.make(env_name) as env:
            ts = await env.reset()

            # Keep all-layer resid_post means for each step:
            # per step -> [n_layers, d_model]
            per_step_all_layers = []
            n_steps = 0

            with ActivationCapture(model) as capture:
                while (not ts.last()) and (n_steps < max_steps_per_episode):
                    capture.start_step()

                    assert ts.observation is not None
                    action = await client(ts.observation)

                    capture.end_step()

                    # Gather resid_post from all layers for this step.
                    step_layer_vecs = []
                    for layer in range(model.cfg.n_layers):
                        hook_name = f"blocks.{layer}.hook_resid_post"
                        resid_post = capture.get_trajectory().get_activation(n_steps, hook_name)
                        step_layer_vecs.append(_to_numpy_mean_over_tokens(resid_post))

                    per_step_all_layers.append(np.stack(step_layer_vecs, axis=0))

                    ts = await env.step(action)
                    n_steps += 1

            final_reward = _scalar_reward(ts.reward)
            label = int(final_reward == 1.0)

            if not per_step_all_layers:
                # If an episode terminates instantly, skip it.
                continue

            # [steps, n_layers, d_model]
            episode_tensor = np.stack(per_step_all_layers, axis=0)

            # Feature used for probing: middle layer, averaged over steps.
            middle_layer_episode_feature = episode_tensor[:, middle_layer, :].mean(axis=0)

            records.append(
                EpisodeRecord(
                    features_middle=middle_layer_episode_feature,
                    label_success=label,
                    episode_idx=ep,
                    n_steps=n_steps,
                )
            )

            print(
                f"episode={ep:03d} steps={n_steps} final_reward={final_reward:.3f} success={label}"
            )

    return records


records = asyncio.run(
    collect_probe_dataset(
        model=model,
        client=client,
        env_name=ENV_NAME,
        n_episodes=N_EPISODES,
        max_steps_per_episode=MAX_STEPS_PER_EPISODE,
        middle_layer=middle_layer,
    )
)

print(f"Collected {len(records)} usable episodes")
print(f"Successes: {sum(r.label_success for r in records)} / {len(records)}")

X = np.stack([r.features_middle for r in records], axis=0)
y = np.array([r.label_success for r in records], dtype=np.int64)

print("X shape:", X.shape)
print("y mean (success rate):", y.mean() if len(y) else "n/a")

if len(np.unique(y)) < 2:
    raise RuntimeError(
        "Need both positive and negative examples to train a classifier. "
        "Increase N_EPISODES or task diversity."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=SEED,
    stratify=y,
)

probe = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=2000,
    solver="liblinear",
    random_state=SEED,
)
probe.fit(X_train, y_train)

probs = probe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(np.int64)

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print(f"Test accuracy: {acc:.4f}")
print(f"Test ROC-AUC:  {auc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, preds, digits=4))

# Inspect the probe weights: largest-magnitude dimensions in middle-layer resid_post
coef = probe.coef_[0]
abs_order = np.argsort(np.abs(coef))[::-1]

TOP_K = 20
print(f"Top {TOP_K} most influential hidden dimensions:")
for rank, idx in enumerate(abs_order[:TOP_K], start=1):
    print(f"{rank:02d}. dim={idx:4d} weight={coef[idx]: .6f}")

# ## TODOs
#
# - If classes are highly imbalanced, increase `N_EPISODES` and consider
#   `class_weight="balanced"` in logistic regression.
# - You can swap the feature aggregation strategy from mean-over-steps to:
#   - last-step middle layer only
#   - max pool over steps
#   - concatenation of selected step summaries
# - To compare layers directly, build `X` from each layer in turn and train one
#   probe per layer.