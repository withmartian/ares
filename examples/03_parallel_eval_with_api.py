"""Evaluate all tasks in SWEBench Verified with a frontier LLM.

This example shows how you can use ARES' async-first design to
evaluate SWEBench Verified tasks in parallel.
It also shows how to swap out the container factory from a local
docker factory to a Daytona factory, so you can get higher
throughput without needing a beefy local machine.

Prerequisites:

    - `CHAT_COMPLETION_API_KEY` set in `.env` with a Martian API key.
       To get a Martian API key:
       1) sign up at https://app.withmartian.com.
       2) on the `Billing` tab, add a payment method + top up some credits.
       3) on the `API Keys` tab create an API key.
    - `DAYTONA_API_KEY` and `DAYTONA_API_URL` set in `.env` with a Daytona API key and URL.
       1) Sign up at https://www.daytona.io.
       2) Go to API Keys and create an API key.
    - Install dependencies: `uv sync --group examples`

Example usage:

    uv run -m examples.03_parallel_eval_with_api
"""

import asyncio
from collections.abc import Awaitable
from typing import Any

import ares
from ares.containers import containers
from ares.containers import daytona
from ares.environments import base
from ares.llms import chat_completions_compatible

from . import utils


async def evaluate_task(
    preset_name: str,
    task_idx: int,
    agent: chat_completions_compatible.ChatCompletionCompatibleLLMClient,
    container_factory: containers.ContainerFactory,
) -> base.TimeStep[Any, float, float]:
    async with ares.make(f"{preset_name}:{task_idx}", container_factory=container_factory) as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()

        # Continue until the episode is done
        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)
            # Step the environment with the action
            ts = await env.step(action)

        return ts  # Return the final timestep.


async def main():
    # Create an LLM client using the ChatCompletionCompatibleLLMClient
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="openai/gpt-5-mini")

    # We want to run our tasks on daytona because we can get much better throughput.
    container_factory = daytona.DaytonaContainer

    # We will be running MiniSweAgent on SWEBench Verified.
    # We can find out how many tasks are available by looking at the preset info.
    # We will make a distinct environment with a single task; this ensures that when we reset
    # the environment we get one specific task.
    preset_name = "sbv-mswea"
    num_tasks = ares.info(preset_name).num_tasks

    num_parallel_workers = 50
    sem = asyncio.Semaphore(num_parallel_workers)

    async def _await_with_semaphore(
        task: Awaitable[base.TimeStep[Any, float, float]],
    ) -> base.TimeStep[Any, float, float]:
        async with sem:
            return await task

    tasks = [
        _await_with_semaphore(evaluate_task(preset_name, task_idx, agent, container_factory))
        for task_idx in range(num_tasks)
    ]

    results = await utils.gather_with_scores(*tasks)

    total_successes = sum([r.reward for r in results if not isinstance(r, Exception) and r.reward is not None])

    print("All tasks completed!")
    print(f"Success rate: {total_successes / num_tasks}")


if __name__ == "__main__":
    asyncio.run(main())
