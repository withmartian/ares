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

from . import utils


async def main():
    print("=" * 80)
    print("ARES + TransformerLens Mechanistic Interpretability Example")
    print("=" * 80)

    # Load a small model for demonstration
    # For real work, you'd use a larger model like gpt2-medium or pythia-1.4b
    print("\nLoading HookedTransformer model...")
    model = HookedTransformer.from_pretrained(
        "qwen2.5-1.5b-instruct",
        device="cpu",  # Change to "cuda" if you have a GPU
        n_ctx=4096,  # Setting increased context window here since coding takes a lot of tokens
    )

    # Create the LLM client using the HookedTransformer model
    client = HookedTransformerLLMClient(model=model)

    # Example 1: Basic execution with activation capture
    print("\nRunning agent with activation capture...")
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
                # Limit the number of tokens generated to just 256 for this example
                action = await client(ts.observation, max_output_tokens=256)

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

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)
    print("\nSee src/ares/contrib/mech_interp/README.md for more examples!")


if __name__ == "__main__":
    asyncio.run(main())
