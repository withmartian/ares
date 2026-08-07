"""Evaluate a single task in SWE-bench Verified with a frontier LLM.

This example shows how to swap out a local LLM for an LLM being run via API.

Prerequisites:

    - `CHAT_COMPLETION_API_KEY` set in `.env` with a Martian API key.
       To get a Martian API key:
       1) sign up at https://app.withmartian.com.
       2) on the `Billing` tab, add a payment method + top up some credits.
       3) on the `API Keys` tab create an API key.
    - Install dependencies: `uv sync --group examples`

Example usage:

    uv run -m examples.02_sequential_eval_with_api

    uv run -m examples.02_sequential_eval_with_api \
        --model openai/gpt-5-mini \
        --preset-name tbench-opencode
"""

import argparse
import asyncio

import ares
from ares import config
from ares.llms import chat_completions_compatible

from . import utils


async def main(*, model: str, preset_name: str):
    # Fail fast if env vars aren't set.
    if not config.CONFIG.chat_completion_api_key:
        raise ValueError("CHAT_COMPLETION_API_KEY is not set")

    # Create an LLM client using the ChatCompletionCompatibleLLMClient
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model=model)

    # `:0` means load only the first task.
    # By default, ares.make will use local Docker containers.
    async with ares.make(f"{preset_name}:0") as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0
        total_reward = 0.0

        # Continue until the episode is done
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

        # Display final results
        print(f"\n{'=' * 80}")
        print(f"Episode completed after {step_count} steps")
        print(f"Total reward: {total_reward}")
        artifact_dir = getattr(env, "artifact_dir", None)
        if artifact_dir is not None:
            print(f"Artifacts: {artifact_dir}")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--preset-name", default="sbv-mswea")
    args = parser.parse_args()
    asyncio.run(main(model=args.model, preset_name=args.preset_name))
