# ARES

ARES (Agentic Research and Evaluation Suite) is an RL-first framework for training and evaluating agents.

## Installation

For now, we recommend running ARES locally from this directory:

`uv sync --all-groups`

and you're ready to get started.

Alternatively, include it as a dependency in your own project's pyproject.toml using a relative path.
PyPI installation will be coming soon.

## Getting Started

ARES environments use an async version of the [dm_env](https://github.com/google-deepmind/dm_env) spec.
Below is an example snippet of what this might look like in your code.

By default, containers are run in Daytona, so you will need to:

1. Create a daytona account at [https://www.daytona.io](https://www.daytona.io/)
1. Create a `.env` with `DAYTONA_API_KEY=...` and `DAYTONA_API_URL=...` set with an API key generated from your account.

This example also makes use of Martian for API inference. Similarly, you will need to

1. Create an account at [https://app.withmartian.com](https://app.withmartian.com)
1. Add `CHAT_COMPLETION_API_KEY=...` to your `.env` with a Martian API key.

Then, you can run the following example:

```
import asyncio

from ares.code_agents import llms
from ares.environments import swebench_env


async def main():
    agent = llms.ChatCompletionCompatibleLLMClient(model="openai/gpt-4.1-mini")
    all_tasks = swebench_env.swebench_verified_tasks()
    tasks = [all_tasks[0]]  # Run on only one task for now.

    async with swebench_env.SweBenchEnv(tasks=tasks) as env:
        ts = await env.reset()
        while not ts.last():
            # The agent takes the observation (LLM Request)
            # and returns an action (LLM Response).
            print(f"Observation: {ts.observation}")
            action = await agent(ts.observation)

            # The environment takes the action (LLM Response)
            # and returns the next LLM request, reward, and discount.
            print(f"Action: {action}")
            ts = await env.step(action)


if __name__ == "__main__":
    asyncio.run(main())
```