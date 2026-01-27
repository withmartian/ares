"""Evaluate all tasks in SWEBench Verified with a frontier LLM.

This example shows how you can use ARES' async-first design to
evaluate SWEBench Verified tasks in parallel with a rich TUI dashboard.

The dashboard displays:
- Real-time summary statistics (running, completed, errors, success rate, avg return, total cost)
- Task status table with current step, reward, cost, and duration
- Histogram of agent step distribution
- Recent activity logs

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

from . import viz


async def evaluate_task(
    preset_name: str,
    task_idx: int,
    agent: chat_completions_compatible.ChatCompletionCompatibleLLMClient,
    container_factory: containers.ContainerFactory,
    dashboard: viz.EvaluationDashboard,
) -> base.TimeStep[Any, float, float]:
    """Evaluate a single task and report progress to the dashboard.

    Args:
        preset_name: The preset name to use.
        task_idx: The task index within the preset.
        agent: The LLM client to use.
        container_factory: Factory for creating containers.
        dashboard: Dashboard to update with progress.

    Returns:
        The final timestep of the episode.
    """
    async with dashboard.wrap(
        task_idx, ares.make(f"{preset_name}:{task_idx}", container_factory=container_factory)
    ) as env:
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
    # model = "moonshotai/kimi-k2-thinking"
    model = "openai/gpt-5-mini"
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model=model)

    # We want to run our tasks on daytona because we can get much better throughput.
    container_factory = daytona.DaytonaContainer

    # We will be running MiniSweAgent on SWEBench Verified.
    # We can find out how many tasks are available by looking at the preset info.
    # We will make a distinct environment with a single task; this ensures that when we reset
    # the environment we get one specific task.
    preset_name = "sbv-mswea"
    num_tasks = ares.info(preset_name).num_tasks

    # For testing, you can limit the number of tasks
    # num_tasks = min(num_tasks, 5)  # Uncomment to test with just 5 tasks

    num_parallel_workers = 200
    sem = asyncio.Semaphore(num_parallel_workers)

    # Create the dashboard
    dashboard = viz.EvaluationDashboard(
        total_tasks=num_tasks,
        preset_name=preset_name,
        max_parallel=num_parallel_workers,
    )

    async def _await_with_semaphore(
        task: Awaitable[base.TimeStep[Any, float, float]],
    ) -> base.TimeStep[Any, float, float]:
        async with sem:
            return await task

    tasks = [
        _await_with_semaphore(evaluate_task(preset_name, task_idx, agent, container_factory, dashboard))
        for task_idx in range(num_tasks)
    ]

    # Run the evaluation with the dashboard
    async with dashboard:
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Compute final statistics
    total_successes = sum([r.reward for r in results if isinstance(r, base.TimeStep) and r.reward is not None])

    print("\nAll tasks completed!")
    print(f"Success rate: {total_successes / num_tasks:.2%}")
    print(f"Total tasks: {num_tasks}")
    print(f"Successful: {int(total_successes)}")
    print(f"Failed: {num_tasks - int(total_successes)}")


if __name__ == "__main__":
    asyncio.run(main())
