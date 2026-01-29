"""Evaluate all tasks in SWE-bench Verified with a frontier LLM.

This example shows how you can use ARES' async-first design to
evaluate SWE-bench Verified tasks in parallel with a rich TUI dashboard.

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

    Or with command line arguments:

    uv run -m examples.03_parallel_eval_with_api \
        --model openai/gpt-5-mini \
        --preset-name sbv-mswea \
        --num-parallel-workers 20 \
        --num-tasks 10
"""

import asyncio
from collections.abc import Awaitable
import dataclasses
import os
from typing import Any

import ares
from ares import config
from ares import containers
from ares import llms
from ares.contrib import eval_visualizer
import simple_parsing


@dataclasses.dataclass(frozen=True)
class Args:
    # gpt-5-mini is a good and relatively cheap model for this example.
    model: str = "openai/gpt-5-mini"
    # We will be running mini-swe-agent on SWE-bench Verified.
    preset_name: str = "sbv-mswea"
    # The higher the parallelism, the quicker the evaluation will be.
    # However, you may need higher quota in Daytona.
    num_parallel_workers: int = 20
    # If None, run on all tasks. Otherwise, limit to `num_tasks` tasks.
    num_tasks: int | None = None


async def evaluate_task(
    preset_name: str,
    task_idx: int,
    agent: llms.ChatCompletionCompatibleLLMClient,
    container_factory: containers.ContainerFactory,
    dashboard: eval_visualizer.EvaluationDashboard,
) -> ares.TimeStep[Any, float, float]:
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


async def main(args: Args):
    # Fail fast if env vars aren't set.
    if not config.CONFIG.chat_completion_api_key:
        raise ValueError("CHAT_COMPLETION_API_KEY is not set")
    if "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY is not set")

    # Create an LLM client using the ChatCompletionCompatibleLLMClient
    agent = llms.ChatCompletionCompatibleLLMClient(model=args.model)

    # We want to run our tasks on daytona because we can get much better throughput.
    container_factory = containers.DaytonaContainer

    # We can find out how many tasks are available by looking at the preset info.
    # We will make a distinct environment with a single task; this ensures that when we reset
    # the environment we get one specific task.
    num_tasks = ares.info(args.preset_name).num_tasks
    if args.num_tasks is not None:
        num_tasks = min(num_tasks, args.num_tasks)  # Potentially limit to a subset of tasks.

    # For testing, you can limit the number of tasks
    # num_tasks = min(num_tasks, 5)  # Uncomment to test with just 5 tasks
    sem = asyncio.Semaphore(args.num_parallel_workers)

    # Create the dashboard
    dashboard = eval_visualizer.EvaluationDashboard(
        total_tasks=num_tasks,
        preset_name=args.preset_name,
        max_parallel=args.num_parallel_workers,
    )

    async def _await_with_semaphore(
        task: Awaitable[ares.TimeStep[Any, float, float]],
    ) -> ares.TimeStep[Any, float, float]:
        async with sem:
            return await task

    tasks = [
        _await_with_semaphore(evaluate_task(args.preset_name, task_idx, agent, container_factory, dashboard))
        for task_idx in range(num_tasks)
    ]

    # Run the evaluation with the dashboard
    async with dashboard:
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Compute final statistics
    total_successes = sum([r.reward for r in results if isinstance(r, ares.TimeStep) and r.reward is not None])

    print("\nAll tasks completed!")
    print(f"Success rate: {total_successes / num_tasks:.2%}")
    print(f"Total tasks: {num_tasks}")
    print(f"Successful: {int(total_successes)}")
    print(f"Failed: {num_tasks - int(total_successes)}")
    print(f"Errors: {len([r for r in results if isinstance(r, Exception)])}")


if __name__ == "__main__":
    asyncio.run(main(simple_parsing.parse(Args, add_option_string_dash_variants=simple_parsing.DashVariant.DASH)))
