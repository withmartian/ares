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
3. Identifying critical moments where interventions significantly alter outcomes
4. Analyzing information flow across dozens of inference steps

## Quick Start

### Installation

```bash
# Install ARES with mech_interp group (includes TransformerLens)
uv sync --group mech_interp
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
    client = HookedTransformerLLMClient(model=model, model_name="gpt2-small")

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
    model_name="gpt2-medium",
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
    model_name="Qwen/Qwen2.5-3B-Instruct",
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

Apply causal interventions during agent execution to study model behavior.

```python
from ares.contrib.mech_interp import (
    InterventionManager,
    create_zero_ablation_hook,
    create_path_patching_hook,
)

manager = InterventionManager(model)

# Ablate attention heads
manager.add_intervention(
    hook_name="blocks.0.attn.hook_result",
    hook_fn=create_zero_ablation_hook(heads=[0, 1, 2]),
    description="Ablate heads 0-2 in layer 0",
    apply_at_steps=[5, 6, 7]  # Optional: only at specific steps
)

# Run with interventions
with manager:
    async with env:
        ts = await env.reset()
        step_count = 0
        while not ts.last():
            action = await client(ts.observation)
            ts = await env.step(action)

            step_count += 1
            manager.increment_step()  # Track which step we're on
```

### 4. Hook Utilities

Pre-built hooks for common interventions:

**Zero Ablation:**
```python
from ares.contrib.mech_interp import create_zero_ablation_hook

# Ablate specific positions
hook = create_zero_ablation_hook(positions=[10, 11, 12])

# Ablate specific attention heads
hook = create_zero_ablation_hook(heads=[0, 1])
```

**Path Patching:**
```python
from ares.contrib.mech_interp import create_path_patching_hook

# Run clean and corrupted inputs
clean_cache, _ = model.run_with_cache(clean_tokens)
corrupted_cache, _ = model.run_with_cache(corrupted_tokens)

# Patch clean activations into corrupted run
hook = create_path_patching_hook(
    clean_activation=clean_cache["blocks.0.hook_resid_post"],
    positions=[5, 6, 7]
)
```

**Mean Ablation:**
```python
from ares.contrib.mech_interp import create_mean_ablation_hook

# Compute mean activation over dataset
mean_cache = compute_mean_activations(model, dataset)

hook = create_mean_ablation_hook(
    mean_activation=mean_cache["blocks.0.hook_resid_post"],
    positions=[10, 11, 12]
)
```

**Attention Knockout:**
```python
from ares.contrib.mech_interp import create_attention_knockout_hook

# Prevent position 20 from attending to positions 0-10
hook = create_attention_knockout_hook(
    source_positions=[20],
    target_positions=list(range(11))
)
```

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
    client = HookedTransformerLLMClient(model=model, model_name=model.cfg.model_name)

    with ActivationCapture(model) as capture:
        # Run same task
        trajectory = run_episode(env, client, capture)
        trajectories.append(trajectory)

# Compare attention patterns, activation magnitudes, etc.
compare_trajectories(trajectories)
```

## Advanced Examples

### Example: Finding "Code Understanding" Circuits

```python
import torch
from transformer_lens import HookedTransformer
from ares.contrib.mech_interp import *
from ares.environments import swebench_env

async def find_code_understanding_circuits():
    model = HookedTransformer.from_pretrained("gpt2-medium")
    client = HookedTransformerLLMClient(model=model, model_name="gpt2-medium")

    # 1. Collect baseline trajectory on a task
    tasks = swebench_env.swebench_verified_tasks()
    success_task = [t for t in tasks if is_success(t)][0]

    with ActivationCapture(model) as capture:
        baseline_reward = await run_episode(swebench_env.SweBenchEnv([success_task]), client, capture)

    baseline_trajectory = capture.get_trajectory()

    # 2. For each layer and step, ablate and measure impact
    impact_matrix = torch.zeros(model.cfg.n_layers, len(baseline_trajectory))

    for layer in range(model.cfg.n_layers):
        for step in range(len(baseline_trajectory)):
            manager = InterventionManager(model)
            manager.add_intervention(
                hook_name=f"blocks.{layer}.hook_resid_post",
                hook_fn=create_mean_ablation_hook(),
                apply_at_steps=[step]
            )

            with manager:
                perturbed_reward = await run_episode(swebench_env.SweBenchEnv([success_task]), client)

            impact_matrix[layer, step] = abs(baseline_reward - perturbed_reward)

    # 3. Identify critical (layer, step) pairs
    critical_components = (impact_matrix > threshold).nonzero()

    print(f"Found {len(critical_components)} critical components")

    # 4. Drill down to attention heads in critical layers
    for layer, step in critical_components[:10]:  # Top 10
        for head in range(model.cfg.n_heads):
            manager = InterventionManager(model)
            manager.add_intervention(
                hook_name=f"blocks.{layer}.attn.hook_result",
                hook_fn=create_zero_ablation_hook(heads=[head]),
                apply_at_steps=[step.item()]
            )

            with manager:
                head_reward = await run_episode(swebench_env.SweBenchEnv([success_task]), client)

            if abs(baseline_reward - head_reward) > head_threshold:
                print(f"Critical: Layer {layer}, Step {step}, Head {head}")

                # Visualize attention pattern
                attn_pattern = baseline_trajectory.get_activation(step, f"blocks.{layer}.attn.hook_pattern")
                visualize_attention(attn_pattern[:, head, :, :])

# Run analysis
asyncio.run(find_code_understanding_circuits())
```

## Best Practices

1. **Start Small**: Use small models (gpt2-small, gpt2-medium) for initial exploration
2. **Limit Steps**: Use `max_steps` during development to avoid long-running experiments
3. **Save Often**: Save trajectories immediately after capture for later analysis
4. **Filter Hooks**: Use `hook_filter` in ActivationCapture to reduce memory usage
5. **GPU Management**: Move activations to CPU after capture: `.detach().cpu()`
6. **Batch Analysis**: Process multiple trajectories in batch when possible

## Performance Tips

**Memory Optimization:**
```python
# Only capture specific hooks
capture = ActivationCapture(
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

# Use torch.no_grad() (already done in client, but good to remember)
with torch.no_grad():
    ...
```

## Comparison to Traditional MI

| Traditional MI | ARES + Mech Interp |
|----------------|---------------------|
| Single prompts | Multi-step trajectories (50-100+ steps) |
| Static analysis | Dynamic, evolving state |
| Local causality | Long-horizon dependencies |
| Fixed context | Context grows with episode |
| Immediate outcomes | Delayed rewards |

## Resources

- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)
- [Anthropic's Circuits Thread](https://transformer-circuits.pub/)
- [Example Notebook: Trajectory-Level Analysis](./notebooks/trajectory_analysis.ipynb) *(coming soon)*
- [Blog Post: Beyond Static MI](https://withmartian.com/post/beyond-static-mechanistic-interpretability-agentic-long-horizon-tasks-as-the-next-frontier)

## Citation

If you use this module in your research, please cite:

```bibtex
@software{ares_mech_interp_2025,
  title = {ARES Mechanistic Interpretability Module},
  author = {Martian},
  year = {2025},
  url = {https://github.com/anthropics/ares}
}
```
