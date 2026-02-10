# Mechanistic Interpretability for ARES

This module provides deep integration between ARES and [TransformerLens](https://github.com/neelnanda-io/TransformerLens), enabling mechanistic interpretability research on code agents across long-horizon tasks.

## Why ARES for Mechanistic Interpretability?

Traditional mechanistic interpretability focuses on static, single-step analysis. But modern AI agents:
- Make decisions across many steps (50-100+ steps per episode)
- Maintain internal state that evolves over time
- Exhibit temporal dependencies and long-horizon planning

ARES enables **trajectory-level mechanistic interpretability** by:
1. Capturing activations across entire agent episodes
2. Studying how internal representations evolve during multi-step reasoning
3. [COMING SOON] Identifying critical moments where interventions significantly alter episode-level outcomes
4. Seeing how activations differ across different agent frameworks for the same task
5. You tell us!

## Quick Start

### Installation

```bash
# Install ARES with mech_interp group (includes TransformerLens)
uv add ares[mech-interp]
# or with pip
pip install ares[mech-interp]
```

### Basic Example

```python
import asyncio
from transformer_lens import HookedTransformer
from ares.contrib.mech_interp import HookedTransformerLLMClient, ActivationCapture
from ares.environments import swebench_env

async def main():
    # Load model
    model = HookedTransformer.from_pretrained("gpt2-small")
    client = HookedTransformerLLMClient(model=model)

    # Run agent and capture activations
    tasks = swebench_env.swebench_verified_tasks()[:1]

    async with swebench_env.SweBenchEnv(tasks=tasks) as env:
        with ActivationCapture(model) as capture:
            ts = await env.reset()
            while not ts.last():
                capture.start_step()
                action = await client(ts.observation)
                capture.end_step()
                ts = await env.step(action)

        # Analyze trajectory
        trajectory = capture.get_trajectory()
        print(f"Captured {len(trajectory)} steps")
        trajectory.save("./activations/episode_001")

asyncio.run(main())
```

## Core Components

### 1. HookedTransformerLLMClient

An ARES-compatible LLM client that uses TransformerLens's `HookedTransformer` for inference.

```python
from transformer_lens import HookedTransformer
from ares.contrib.mech_interp import HookedTransformerLLMClient

model = HookedTransformer.from_pretrained("gpt2-medium")
client = HookedTransformerLLMClient(
    model=model,
    max_new_tokens=1024,
    generation_kwargs={"temperature": 0.7}
)
```

**With Chat Templates:**

```python
from transformers import AutoTokenizer
from ares.contrib.mech_interp import create_hooked_transformer_client_with_chat_template

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

client = create_hooked_transformer_client_with_chat_template(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2048,
)
```

### 2. ActivationCapture

Captures activations across an agent trajectory for later analysis.

```python
from ares.contrib.mech_interp import ActivationCapture

with ActivationCapture(model) as capture:
    # Filter which hooks to capture (optional)
    # capture = ActivationCapture(model, hook_filter=lambda name: "attn" in name)

    async with env:
        ts = await env.reset()
        while not ts.last():
            capture.start_step()
            action = await client(ts.observation)
            capture.end_step()

            # Optionally record metadata
            capture.record_step_metadata({"step_reward": ts.reward})

            ts = await env.step(action)

# Get trajectory
trajectory = capture.get_trajectory()

# Access specific activations
attn_pattern_step_5 = trajectory.get_activation(5, "blocks.0.attn.hook_pattern")

# Get activations across trajectory
all_layer0_resid = trajectory.get_activation_across_trajectory("blocks.0.hook_resid_post")

# Save for later
trajectory.save("./activations/episode_001")

# Load later
loaded = TrajectoryActivations.load("./activations/episode_001")
```

**Automatic Capture:**

```python
from ares.contrib.mech_interp import automatic_activation_capture

with automatic_activation_capture(model) as capture:
    # Activations are captured automatically during each client call
    async with env:
        ts = await env.reset()
        while not ts.last():
            response = await client(ts.observation)
            ts = await env.step(response)

    trajectory = capture.get_trajectory()
```

### 3. InterventionManager

Coming Soon!

## Research Use Cases

### 1. Attention Head Analysis Across Trajectories

Study how attention patterns evolve as agents work through tasks:

```python
with ActivationCapture(model) as capture:
    # Run agent episode
    ...

trajectory = capture.get_trajectory()

# Analyze attention patterns across all steps
for step in range(len(trajectory)):
    attn = trajectory.get_activation(step, "blocks.5.attn.hook_pattern")
    # Analyze attention to specific tokens, copy behavior, etc.
```

### 2. Identifying Critical Decision Points

NOTE: This example relies on a `InterventionManager` component that we are still finalizing the interface for. Nice support for intervention via hooks is coming soon!

Find steps where small perturbations significantly alter outcomes:

```python
baseline_trajectory = run_episode(env, client)

# Test interventions at each step
critical_steps = []
for step in range(len(baseline_trajectory)):
    manager = InterventionManager(model)
    manager.add_intervention(
        hook_name="blocks.3.hook_resid_post",
        hook_fn=create_zero_ablation_hook(positions=[10, 11]),
        apply_at_steps=[step]
    )

    perturbed_reward = run_episode_with_intervention(env, client, manager)

    if abs(perturbed_reward - baseline_trajectory.reward) > threshold:
        critical_steps.append(step)
```

### 3. Circuit Discovery in Multi-Step Reasoning

Use path patching to identify circuits responsible for specific capabilities:

```python
# Compare successful vs failed trajectories
success_cache = run_and_cache_episode(env, client, success_task)
failure_cache = run_and_cache_episode(env, client, failure_task)

# Systematically patch components from success to failure
for layer in range(model.cfg.n_layers):
    for step in range(len(failure_cache)):
        hook = create_path_patching_hook(
            clean_activation=success_cache.get_activation(step, f"blocks.{layer}.hook_resid_post"),
            positions=None
        )

        # Test if this component recovers successful behavior
        ...
```

### 4. Temporal Information Flow Analysis

Track how information propagates through the model across steps:

```python
# Capture activations with detailed metadata
with ActivationCapture(model) as capture:
    async with env:
        ts = await env.reset()
        while not ts.last():
            capture.start_step()
            action = await client(ts.observation)
            capture.end_step()

            # Record what the agent was doing
            capture.record_step_metadata({
                "action_type": classify_action(action),
                "error_present": check_for_errors(ts.observation),
                "file_context": extract_file_context(ts.observation),
            })

            ts = await env.step(action)

trajectory = capture.get_trajectory()

# Analyze when specific features activate relative to task events
# E.g., does the model activate "file-reading neurons" before bash commands?
```

### 5. Comparative Analysis: Different Models

Compare how different models solve the same task:

```python
models = [
    HookedTransformer.from_pretrained("gpt2-small"),
    HookedTransformer.from_pretrained("gpt2-medium"),
    HookedTransformer.from_pretrained("pythia-1.4b"),
]

trajectories = []
for model in models:
    client = HookedTransformerLLMClient(model=model)

    with ActivationCapture(model) as capture:
        # Run same task
        trajectory = run_episode(env, client, capture)
        trajectories.append(trajectory)

# Compare attention patterns, activation magnitudes, etc.
compare_trajectories(trajectories)
```

## Performance Tips

**Memory Optimization:**
```python
# Only capture specific activations
ActivationCapture(
    model,
    hook_filter=lambda name: "attn.hook_pattern" in name or "hook_resid" in name
)

# Clear trajectory periodically
if len(capture.step_activations) > 100:
    trajectory = capture.get_trajectory()
    trajectory.save(f"./checkpoints/step_{step}")
    capture.clear()
```

**Speed Optimization:**
```python
# Use smaller models for initial exploration
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# Reduce max_new_tokens during ablation studies
client = HookedTransformerLLMClient(model=model, max_new_tokens=256)
```

## Resources

- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- Example Notebook: Trajectory-Level Analysis *(coming soon)*
- [Blog Post: Beyond Static Mechanistic Interpretability](https://withmartian.com/post/beyond-static-mechanistic-interpretability-agentic-long-horizon-tasks-as-the-next-frontier)

## Citation

If you use this module in your research, please cite:

```bibtex
@software{ares_mech_interp_2025,
  title = {ARES Mechanistic Interpretability Module},
  author = {Martian},
  year = {2025},
  url = {https://github.com/withmartian/ares}
}
```
