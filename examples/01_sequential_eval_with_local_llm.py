"""Minimal example of using ARES with SWE-bench environment.

No API keys required - just local Docker containers and a local LLM.
It runs a few steps of the first task in SWEBench Verified.

This example shows the basic way of interacting with an ARES environment.

This example uses Qwen2-0.5B-Instruct with a Llama CPP-backed LLM client.
Unfortunately this model is too weak to solve the task we've set it,
so we only run it for 5 steps.
In example 02_sequential_eval_with_api.py we'll use a more powerful LLM.

Prerequisites:

    - Install docker & make sure the daemon is running.
    - Install dependencies: `uv sync --group examples`
    - If you see Docker authentication errors (e.g., "email must be verified"):
        * RECOMMENDED: Set DOCKER_SKIP_AUTH=true to use anonymous pulls (no account needed)
        * Or run `docker logout` to clear stored credentials
        * Or verify your email at https://hub.docker.com/settings/general

Example usage:

    uv run -m examples.01_sequential_eval_with_local_llm
"""

import asyncio

import ares
from ares.contrib import llama_cpp

from . import utils


async def main():
    # Load Qwen2-0.5B-Instruct using a Llama CPP-backed LLM client.
    agent = llama_cpp.create_qwen2_0_5b_instruct_llama_cpp_client(n_ctx=32_768)

    # `sbv-mswea` is SWEBench Verified with mini-swe-agent.
    # `:0` means load only the first task.
    # By default, ares.make will use local Docker containers.
    async with ares.make("sbv-mswea:0") as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0
        total_reward = 0.0

        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)

            # Print the observation and action.
            utils.print_step(step_count, ts.observation, action)

            # Step the environment with the action
            ts = await env.step(action)

            assert ts.reward is not None
            total_reward += ts.reward
            step_count += 1

            # We only run for 5 steps; this model isn't strong enough to solve the task.
            if step_count >= 5:
                break

        # Episode complete!
        print()
        print("=" * 80)
        print(f"Episode truncated after {step_count} steps")
        print(f"Total reward: {total_reward}")
        print()
        print("🎉 You've seen ARES in action!")
        print()
        print("Next steps:")
        print("  - Try example 02_sequential_eval_with_api.py for a more powerful LLM")
        print("  - Try example 03_parallel_eval_with_api.py to evaluate an entire suite of tasks")
        print("  - Read the docs to learn more about ARES")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
