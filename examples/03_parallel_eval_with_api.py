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
        --model openai/gpt-5.5 \
        --preset-name sbv-mswea \
        --num-parallel-workers 20 \
        --num-tasks 10
"""

import asyncio
from collections.abc import Awaitable
import dataclasses
import datetime
import json
import os
import pathlib
import time
import traceback
from typing import Any

import ares
from ares import config
from ares import containers
from ares import llms
from ares.contrib import eval_visualizer
import simple_parsing


def _base_preset_name(preset_name: str) -> str:
    return preset_name.split("@", 1)[0].split(":", 1)[0]


def _selected_task_count(preset_name: str, base_num_tasks: int) -> int:
    if "@" in preset_name:
        shard_spec = preset_name.split("@", 1)[1]
        shard_index_str, total_shards_str = shard_spec.split("/", 1)
        shard_index = int(shard_index_str)
        total_shards = int(total_shards_str)
        start = round(shard_index * base_num_tasks / total_shards)
        end = round((shard_index + 1) * base_num_tasks / total_shards)
        return end - start

    parts = preset_name.split(":")
    if len(parts) == 2:
        return 1
    if len(parts) == 3:
        start = int(parts[1]) if parts[1] else 0
        end = int(parts[2]) if parts[2] else base_num_tasks
        return end - start
    return base_num_tasks


def _selected_task_selectors(preset_name: str, base_num_tasks: int) -> list[str]:
    base_preset = _base_preset_name(preset_name)
    if "@" in preset_name:
        shard_spec = preset_name.split("@", 1)[1]
        shard_index_str, total_shards_str = shard_spec.split("/", 1)
        shard_index = int(shard_index_str)
        total_shards = int(total_shards_str)
        start = round(shard_index * base_num_tasks / total_shards)
        end = round((shard_index + 1) * base_num_tasks / total_shards)
        return [f"{base_preset}:{task_idx}" for task_idx in range(start, end)]

    parts = preset_name.split(":")
    if len(parts) == 2:
        return [preset_name]
    if len(parts) == 3:
        start = int(parts[1]) if parts[1] else 0
        end = int(parts[2]) if parts[2] else base_num_tasks
        return [f"{base_preset}:{task_idx}" for task_idx in range(start, end)]
    return [f"{base_preset}:{task_idx}" for task_idx in range(base_num_tasks)]


@dataclasses.dataclass(frozen=True)
class Args:
    # Uses Responses API with reasoning effort fixed to xhigh.
    model: str = "openai/gpt-5.5"
    # We will be running mini-swe-agent on SWE-bench Verified.
    preset_name: str = "sbv-mswea"
    # The higher the parallelism, the quicker the evaluation will be.
    # However, you may need higher quota in Daytona.
    num_parallel_workers: int = 20
    # If None, run on all tasks. Otherwise, limit to `num_tasks` tasks.
    num_tasks: int | None = None
    # Disable the Textual dashboard for headless/error-debug runs.
    nogui: bool = False
    # Optional path to write structured JSON results.
    output_json: pathlib.Path | None = None


@dataclasses.dataclass(frozen=True)
class TaskResult:
    task_idx: int
    task_name: str | None
    reward: float | None
    cost: float
    duration_s: float
    steps: int
    num_turns: int
    prompt_tokens: int
    cached_prompt_tokens: int
    uncached_prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    status: str
    error: str | None = None
    traceback: str | None = None


async def evaluate_task(
    preset_name: str,
    dashboard_task_idx: int,
    selector: str,
    agent: llms.LLMClient,
    container_factory: containers.ContainerFactory,
    dashboard: eval_visualizer.EvaluationDashboard,
) -> TaskResult:
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
    del preset_name
    start_time = time.time()
    async with dashboard.wrap(dashboard_task_idx, ares.make(selector, container_factory=container_factory)) as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        wrapped_env = getattr(env, "_env", env)
        task = getattr(wrapped_env, "_current_task", None)
        task_name = getattr(task, "name", None)

        # Continue until the episode is done
        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)
            # Step the environment with the action
            ts = await env.step(action)

        task_info = dashboard.tasks[dashboard_task_idx]
        return TaskResult(
            task_idx=dashboard_task_idx,
            task_name=task_name,
            reward=float(ts.reward) if ts.reward is not None else None,
            cost=task_info.cost,
            duration_s=time.time() - start_time,
            steps=task_info.current_step,
            num_turns=task_info.current_step,
            prompt_tokens=task_info.prompt_tokens,
            cached_prompt_tokens=task_info.cached_prompt_tokens,
            uncached_prompt_tokens=task_info.prompt_tokens - task_info.cached_prompt_tokens,
            generated_tokens=task_info.generated_tokens,
            total_tokens=task_info.prompt_tokens + task_info.generated_tokens,
            status="completed",
        )


def _serialize_result(result: TaskResult | BaseException, task_idx: int, dashboard: eval_visualizer.EvaluationDashboard):
    task_info = dashboard.tasks[task_idx]
    if isinstance(result, TaskResult):
        return dataclasses.asdict(result)

    return {
        "task_idx": task_idx,
        "task_name": None,
        "reward": None,
        "cost": task_info.cost,
        "duration_s": task_info.duration,
        "steps": task_info.current_step,
        "num_turns": task_info.current_step,
        "prompt_tokens": task_info.prompt_tokens,
        "cached_prompt_tokens": task_info.cached_prompt_tokens,
        "uncached_prompt_tokens": task_info.prompt_tokens - task_info.cached_prompt_tokens,
        "generated_tokens": task_info.generated_tokens,
        "total_tokens": task_info.prompt_tokens + task_info.generated_tokens,
        "status": "error",
        "error": f"{type(result).__name__}: {result}",
        "traceback": "".join(traceback.format_exception(result)),
    }


def _to_datapoint(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": result["task_name"] or result["task_idx"],
        "score": result["reward"],
        "cost": result["cost"],
        "run_time_s": result["duration_s"],
        "num_turns": result["num_turns"],
        "prompt_tokens": result["prompt_tokens"],
        "cached_prompt_tokens": result["cached_prompt_tokens"],
        "uncached_prompt_tokens": result["uncached_prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
        "total_tokens": result["total_tokens"],
    }


async def main(args: Args):
    # Fail fast if env vars aren't set.
    if not config.CONFIG.chat_completion_api_key:
        raise ValueError("CHAT_COMPLETION_API_KEY is not set")
    if "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY is not set")

    agent = llms.OpenAIResponsesCompatibleLLMClient(model=args.model)

    # We want to run our tasks on daytona because we can get much better throughput.
    container_factory = containers.DaytonaContainer

    # We can find out how many tasks are available by looking at the preset info.
    # We will make a distinct environment with a single task; this ensures that when we reset
    # the environment we get one specific task.
    base_num_tasks = ares.info(_base_preset_name(args.preset_name)).num_tasks
    selectors = _selected_task_selectors(args.preset_name, base_num_tasks)
    num_tasks = len(selectors)
    if args.num_tasks is not None:
        num_tasks = min(num_tasks, args.num_tasks)  # Potentially limit to a subset of tasks.
        selectors = selectors[:num_tasks]

    # For testing, you can limit the number of tasks
    # num_tasks = min(num_tasks, 5)  # Uncomment to test with just 5 tasks
    sem = asyncio.Semaphore(args.num_parallel_workers)

    # Create the dashboard
    dashboard = eval_visualizer.EvaluationDashboard(
        total_tasks=num_tasks,
        preset_name=args.preset_name,
        max_parallel=args.num_parallel_workers,
        nogui=args.nogui,
    )

    async def _await_with_semaphore(
        task: Awaitable[TaskResult],
    ) -> TaskResult:
        async with sem:
            return await task

    tasks = [
        _await_with_semaphore(evaluate_task(args.preset_name, task_idx, selector, agent, container_factory, dashboard))
        for task_idx, selector in enumerate(selectors)
    ]

    # Run the evaluation with the dashboard
    async with dashboard:
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Compute final statistics
    total_successes = sum([r.reward for r in results if isinstance(r, TaskResult) and r.reward is not None])
    serialized_results = [_serialize_result(result, task_idx, dashboard) for task_idx, result in enumerate(results)]

    print("\nAll tasks completed!")
    print(f"Success rate: {total_successes / num_tasks:.2%}")
    print(f"Total tasks: {num_tasks}")
    print(f"Successful: {int(total_successes)}")
    print(f"Failed: {num_tasks - int(total_successes)}")
    print(f"Errors: {len([r for r in results if isinstance(r, Exception)])}")
    for task_idx, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\nTask {task_idx} error:")
            traceback.print_exception(result)

    if args.output_json is not None:
        output = {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "model": args.model,
            "preset_name": args.preset_name,
            "num_parallel_workers": args.num_parallel_workers,
            "num_tasks": num_tasks,
            "success_rate": total_successes / num_tasks,
            "total_successes": total_successes,
            "total_cost": sum(float(result["cost"] or 0.0) for result in serialized_results),
            "data_points": [_to_datapoint(result) for result in serialized_results],
            "results": serialized_results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"Wrote results to {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main(simple_parsing.parse(Args, add_option_string_dash_variants=simple_parsing.DashVariant.DASH)))
