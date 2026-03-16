"""Lightweight smoke example for the canonical Open Responses request path.

This example requires no Docker, no local model weights, and no API keys.
It uses the built-in Twenty Questions environment with a single fixed object so we
can exercise the full observation -> agent -> action -> environment loop locally.

Example usage:

    uv run -m examples.00_twenty_questions_smoke
"""

import asyncio

from ares.environments import twenty_questions
from ares.testing import mock_llm

from . import utils


async def main() -> None:
    agent = mock_llm.MockLLMClient(responses=["Is it Basketball?"])

    async with twenty_questions.TwentyQuestionsEnvironment(
        objects=("Basketball",), oracle_model="openai/gpt-4o-mini"
    ) as env:
        ts = await env.reset()
        assert ts.observation is not None

        action = await agent(ts.observation)
        utils.print_step(0, ts.observation, action)

        ts = await env.step(action)

        print()
        print("=" * 80)
        print(f"Episode finished: {ts.last()}")
        print(f"Reward: {ts.reward}")
        print(f"Agent calls: {agent.call_count}")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
