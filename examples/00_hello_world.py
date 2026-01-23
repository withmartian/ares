"""Hello World example for ARES - no API keys required!

This example demonstrates the ARES RL loop using:
- Local Docker containers (no Daytona account needed)
- A mock LLM (no API keys needed)

This is the fastest way to see ARES in action.

Example usage:

    1. Make sure you have Docker installed and running
    2. Install dependencies: `uv sync --group examples`
    3. Run: `uv run -m examples.00_hello_world`
"""

import asyncio
import time
import uuid

import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage

from ares.code_agents import mini_swe_agent
from ares.containers import docker
from ares.environments import swebench_env
from ares.llms import llm_clients


class MockLLM:
    """A simple mock LLM that demonstrates the RL loop without requiring API keys.

    This mock LLM will:
    1. First, explore the repository structure
    2. Then, read a file
    3. Finally, submit (ending the episode)

    This demonstrates how ARES intercepts LLM requests and responses in the RL loop.
    """

    def __init__(self):
        self.step_count = 0
        # Pre-defined responses to demonstrate the RL loop
        self.responses = [
            "Let me start by exploring the repository structure.\n\n```bash\nls -la\n```",
            "Now let me check the README.\n\n```bash\ncat README.md | head -20\n```",
            "I've explored the repository. Let me submit.\n\n```bash\necho 'MINI_SWE_AGENT_FINAL_OUTPUT'\n```",
        ]

    async def __call__(self, _request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
        """Mock LLM call that returns pre-defined responses."""
        # Get the response for this step (cycle through if we run out)
        response_text = self.responses[min(self.step_count, len(self.responses) - 1)]
        self.step_count += 1

        # Build a proper ChatCompletion response (required by ARES)
        return llm_clients.LLMResponse(
            chat_completion_response=openai.types.chat.chat_completion.ChatCompletion(
                id=str(uuid.uuid4()),
                choices=[
                    openai.types.chat.chat_completion.Choice(
                        message=openai.types.chat.chat_completion_message.ChatCompletionMessage(
                            content=response_text,
                            role="assistant",
                        ),
                        finish_reason="stop",
                        index=0,
                    )
                ],
                created=int(time.time()),
                model="mock-llm",
                object="chat.completion",
                usage=openai.types.completion_usage.CompletionUsage(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                ),
            ),
            cost=0.0,
        )


async def main():
    print("=" * 80)
    print("ARES Hello World - No API Keys Required!")
    print("=" * 80)
    print()
    print("This example demonstrates the ARES RL loop using:")
    print("  ✓ Local Docker containers (no cloud account needed)")
    print("  ✓ Mock LLM (no API keys needed)")
    print()
    print("The mock LLM will execute a few simple bash commands to show")
    print("how ARES intercepts LLM requests and responses.")
    print("=" * 80)
    print()

    # Create a mock LLM (no API keys required!)
    mock_llm = MockLLM()

    # Load SWE-bench tasks (we'll just use one for this demo)
    all_tasks = swebench_env.swebench_verified_tasks()
    tasks = [all_tasks[204]]  # Just one task

    print(f"Running on task: {tasks[0].instance_id}")
    print(f"Repository: {tasks[0].repo}")
    print("-" * 80)
    print()

    # Create the environment with:
    # - Local Docker containers (no Daytona)
    # - MiniSWE code agent
    code_agent_factory = mini_swe_agent.MiniSWECodeAgent
    container_factory = docker.DockerContainer

    async with swebench_env.SweBenchEnv(
        tasks=tasks,
        code_agent_factory=code_agent_factory,
        container_factory=container_factory,
    ) as env:
        # Reset to get initial observation
        ts = await env.reset()
        step_count = 0

        # The RL loop: observation -> action -> next observation
        while not ts.last():
            # The mock LLM processes the observation and returns an action
            action = await mock_llm(ts.observation)

            # Print what's happening
            _print_step(step_count, ts.observation, action)

            # Step the environment with the action
            ts = await env.step(action)
            step_count += 1

        # Episode complete!
        print()
        print("=" * 80)
        print(f"Episode completed after {step_count} steps")
        print(f"Final reward: {ts.reward}")
        print()
        print("🎉 You've seen ARES in action!")
        print()
        print("Next steps:")
        print("  - Try example 01_minimal_loop.py with a real LLM")
        print("  - Try example 02_local_llm.py with a local model")
        print("  - Read the docs to learn more about ARES")
        print("=" * 80)


def _print_step(
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


if __name__ == "__main__":
    asyncio.run(main())
