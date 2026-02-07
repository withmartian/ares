"""Helpful utility functions for examples."""

import asyncio
from collections.abc import Awaitable
import logging
from typing import Any

import ares
from ares import llms
import tqdm

_LOGGER = logging.getLogger(__name__)


def print_step(
    step_count: int,
    observation: llms.LLMRequest | None,
    action: llms.LLMResponse,
) -> None:
    """Print a step in the RL loop.

    Args:
        step_count: The current step count.
        observation: The observation from the environment.
        action: The action from the LLM.
    """
    print(f"\n[Step {step_count}]")
    print("-" * 80)

    if observation is not None:
        messages = list(observation.messages)
        observation_content = messages[-1].get("content", "") if messages else "(no messages)"
        observation_preview = str(observation_content)[:200]
        if len(str(observation_content)) > 200:
            observation_preview += "..."
        print(f"Observation (from environment): {observation_preview}")

    action_content = action.data[0].content
    action_preview = str(action_content)[:200]
    if len(action_content) > 200:
        action_preview += "..."
    print(f"Action (from LLM): {action_preview}")


async def gather_with_scores(
    *futs: Awaitable[ares.TimeStep[Any, float, float] | Exception],
) -> list[ares.TimeStep[Any, float, float] | Exception]:
    """Gather the results of a list of futures while reporting scores.

    Creates a tqdm bar which reports:
    - average score
    - number of finished tasks
    - number of errors

    Args:
        *futs: The futures to gather.

    Returns:
        A list of results.
    """
    results: list[ares.TimeStep[Any, float, float] | Exception] = []
    reward_sum = 0.0
    finished = 0
    errors = 0

    # Create a tqdm bar manually so we can add dynamic information to the postfix.
    with tqdm.tqdm(total=len(futs), desc="Processing items", unit="item") as pbar:
        for future in asyncio.as_completed(futs):
            try:
                result = await future
            except Exception as e:
                _LOGGER.error("Error encountered while processing future: %s", e)
                result = e

            results.append(result)

            if isinstance(result, ares.TimeStep):
                assert result.reward is not None
                reward_sum += result.reward
                finished += 1
            elif isinstance(result, Exception):
                errors += 1

            average_score = reward_sum / finished if finished > 0 else 0.0
            pbar.set_postfix_str(f"{average_score=:.4f}, {finished=}, {errors=}")
            pbar.update(1)

    return results
