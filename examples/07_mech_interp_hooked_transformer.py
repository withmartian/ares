"""Example of using ARES with TransformerLens for mechanistic interpretability.

This example demonstrates how to:
1. Use HookedTransformer with ARES environments
2. Capture activations across an agent trajectory
3. Apply interventions to study model behavior

Example usage:

    1. Make sure you have mech_interp dependencies installed
       `uv sync --group mech_interp`
    2. Run the example
       `uv run -m examples.03_mech_interp_hooked_transformer`
"""

import asyncio

from transformer_lens import HookedTransformer

import ares
from ares.contrib.mech_interp import ActivationCapture
from ares.contrib.mech_interp import HookedTransformerLLMClient
from ares.contrib.mech_interp.hook_utils import InterventionManager, create_zero_ablation_hook

from . import utils


async def main():
    print("=" * 80)
    print("ARES + TransformerLens Mechanistic Interpretability Example")
    print("=" * 80)

    # Load a small model for demonstration
    # For real work, you'd use a larger model like gpt2-medium or pythia-1.4b
    print("\nLoading HookedTransformer model...")
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        device="cpu",  # Change to "cuda" if you have a GPU
    )

    # Create the LLM client with reduced token limit for gpt2-small's context window
    # gpt2-small has max context of 1024 tokens, so we need to be conservative
    client = HookedTransformerLLMClient(
        model=model,
        max_new_tokens=128,  # Keep this small to avoid context overflow
    )

    # Example 1: Basic execution with activation capture
    print("\n[Example 1] Running agent with activation capture...")
    print("-" * 80)

    async with ares.make("sbv-mswea:0") as env:
        # Set up activation capture
        with ActivationCapture(model) as capture:
            ts = await env.reset()
            step_count = 0
            max_steps = 3  # Limit steps for demo

            while not ts.last() and step_count < max_steps:
                # Capture activations for this step
                capture.start_step()

                # Generate response
                assert ts.observation is not None
                action = await client(ts.observation)

                # End capture for this step
                capture.end_step()
                capture.record_step_metadata(
                    {
                        "step": step_count,
                        "action_preview": str(action.data[0].content)[:50],
                    }
                )

                utils.print_step(step_count, ts.observation, action)

                # Step environment
                ts = await env.step(action)
                step_count += 1

            # Analyze captured activations
            trajectory = capture.get_trajectory()
            print(f"\nCaptured activations for {len(trajectory)} steps")

            # Example: Look at attention patterns in layer 0
            if len(trajectory) > 0:
                attn_pattern = trajectory.get_activation(0, "blocks.0.attn.hook_pattern")
                print(f"Layer 0 attention pattern shape: {attn_pattern.shape}")
                print("  [batch, n_heads, query_pos, key_pos]")

            # Save trajectory for later analysis
            print("\nSaving trajectory activations to ./mech_interp_demo/trajectory_001/")
            trajectory.save("./mech_interp_demo/trajectory_001")

    # Example 2: Running with interventions
    print("\n[Example 2] Running agent with attention head ablation...")
    print("-" * 80)

    def create_zero_ablation_hook_with_log(*args, **kwargs):
        hook_fn = create_zero_ablation_hook(*args, **kwargs)
        def wrapped_hook_fn(*args, **kwargs):
            print(f"Running zero ablation hook")
            return hook_fn(*args, **kwargs)
        return wrapped_hook_fn

    async with ares.make("sbv-mswea:0") as env:
        # Set up intervention: ablate heads 0-2 in layer 0
        manager = InterventionManager(model)
        manager.add_intervention(
            hook_name="blocks.0.attn.hook_result",
            hook_fn=create_zero_ablation_hook_with_log(heads=[0, 1, 2]),
            description="Ablate attention heads 0-2 in layer 0",
        )

        print(manager.get_intervention_summary())

        with manager:
            ts = await env.reset()
            step_count = 0
            max_steps = 2  # Limit steps for demo

            while not ts.last() and step_count < max_steps:
                assert ts.observation is not None
                action = await client(ts.observation)

                utils.print_step(step_count, ts.observation, action)

                ts = await env.step(action)
                step_count += 1
                manager.increment_step()

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nNext steps for mechanistic interpretability research:")
    print("1. Load saved activations: TrajectoryActivations.load('./mech_interp_demo/trajectory_001')")
    print("2. Analyze attention patterns across the trajectory")
    print("3. Use interventions to study causal effects")
    print("4. Compare 'clean' vs 'corrupted' trajectories with path patching")
    print("\nSee src/ares/contrib/mech_interp/README.md for more examples!")


if __name__ == "__main__":
    asyncio.run(main())
