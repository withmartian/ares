"""Minimal example of using ARES with SWE-bench environment.

Example usage:

    1. Make sure you have examples dependencies installed
       `uv sync --group examples`
    2. Run the example
       `uv run -m examples.01_minimal_loop`
"""

import asyncio

from ares.code_agents import mini_swe_agent
from ares.containers import docker
from ares.environments import swebench_env
from ares.llms import chat_completions_compatible
from ares.llms import llm_clients


async def main():
    # Create an LLM client using the ChatCompletionCompatibleLLMClient
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="openai/gpt-5-mini")

    # Load all SWE-bench verified tasks
    all_tasks = swebench_env.swebench_verified_tasks()

    # Select just one task for this minimal example.
    tasks = [all_tasks[0]]

    print(f"Running on task: {tasks[0].instance_id}")
    print(f"Repository: {tasks[0].repo}")
    print("-" * 80)

    # Create the SWE-bench environment.
    # The environment will run with a specific code agent, defaulting to the MiniSWECodeAgent.
    # It will also use a specific container factory, which defaults to Daytona.
    # For this example, we are using local Docker containers.
    code_agent_factory = mini_swe_agent.MiniSWECodeAgent
    container_factory = docker.DockerContainer
    async with swebench_env.SweBenchEnv(
        tasks=tasks,
        code_agent_factory=code_agent_factory,
        container_factory=container_factory,
    ) as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0

        # Continue until the episode is done
        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)

            # Print the observation and action.
            _print_observation_and_action(step_count, ts.observation, action)

            # Step the environment with the action
            ts = await env.step(action)

            step_count += 1

        # Display final results
        print(f"\n{'=' * 80}")
        print(f"Episode completed after {step_count} steps")
        print(f"Final reward: {ts.reward}")
        print(f"{'=' * 80}")


def _print_observation_and_action(
    step_count: int, observation: llm_clients.LLMRequest | None, action: llm_clients.LLMResponse
) -> None:
    """A helper function to print the action and observation."""
    action_str = str(action.chat_completion_response.choices[0].message.content)[:100]

    observation_str = "No observation"
    if observation is not None:
        observation_content = list(observation.messages)[-1].get("content", "")
        observation_str = str(observation_content)[:100]
    print(f"\n[Step {step_count}]\nObservation: {observation_str}\nAction: {action_str}")


if __name__ == "__main__":
    asyncio.run(main())
