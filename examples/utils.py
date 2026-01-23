"""Helpful utility functions for examples."""

import asyncio
from collections.abc import Awaitable
from typing import Any

import tqdm

from ares.environments import base
from ares.llms import llm_clients


def print_step(
    step_count: int,
    observation: llm_clients.LLMRequest | None,
    action: llm_clients.LLMResponse,
) -> None:
    """Print a step in the RL loop."""
    print(f"\n[Step {step_count}]")
    print("-" * 80)

    # Print the observation (what the environment sends to the LLM)
    if observation is not None:
        observation_content = list(observation.messages)[-1].get("content", "")
        observation_preview = str(observation_content)[:200]
        if len(str(observation_content)) > 200:
            observation_preview += "..."
        print(f"Observation (from environment): {observation_preview}")

    # Print the action (what the LLM responds with)
    action_content = action.chat_completion_response.choices[0].message.content or ""
    action_preview = str(action_content)[:200]
    if len(action_content) > 200:
        action_preview += "..."
    print(f"Action (from LLM): {action_preview}")


async def gather_with_scores(
    *futs: Awaitable[base.TimeStep[Any, float, float] | Exception],
) -> list[base.TimeStep[Any, float, float] | Exception]:
    results: list[base.TimeStep[Any, float, float] | Exception] = []
    reward_sum = 0.0
    completed = 0
    failed = 0

    # Create a tqdm bar manually
    with tqdm.tqdm(total=len(futs), desc="Processing items", unit="item") as pbar:
        for future in asyncio.as_completed(futs):
            result = await future

            if isinstance(result, Exception):
                failed += 1
            else:
                # Otherwise, it's completed and a timestep.
                assert isinstance(result, base.TimeStep)
                assert result.reward is not None
                reward_sum += result.reward
                completed += 1

            results.append(result)

            # Update the postfix with dynamic information
            average_score = reward_sum / completed if completed > 0 else 0.0
            pbar.set_postfix_str(f"{average_score=:.4f}, {completed=}, {failed=}")
            pbar.update(1)  # Increment the progress bar

    return results
