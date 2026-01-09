"""Minimal example of using ARES with SWE-bench environment and a local LLM.

Example usage:

    1. Make sure you have examples dependencies installed
       `uv sync --group examples`
    2. Run the example
       `uv run -m examples.02_local_llm`
"""

import asyncio
import time
import uuid

import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage
import transformers

from ares.environments import swebench_env
from ares.llms import llm_clients


async def main():
    # To run the local LLM, we need the tokenizer and model.
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    model = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    # Load all SWE-bench verified tasks
    all_tasks = swebench_env.swebench_verified_tasks()

    # Select just one task for this minimal example.
    tasks = [all_tasks[0]]

    print(f"Running on task: {tasks[0].instance_id}")
    print(f"Repository: {tasks[0].repo}")
    print("-" * 80)

    # Create the SWE-bench environment
    async with swebench_env.SweBenchEnv(tasks=tasks) as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0

        # Continue until the episode is done
        while not ts.last():
            # First, tokenize the chat messages so far.
            assert ts.observation is not None
            inputs = tokenizer.apply_chat_template(
                ts.observation.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # Then, generate the completion with the model.
            outputs = model.generate(**inputs, max_new_tokens=2_048)

            # We need token counts for later.
            num_input_tokens = inputs["input_ids"].shape[-1]
            num_output_tokens = outputs[0].shape[-1] - num_input_tokens

            # The output text includes the input too, so we strip it.
            # We remove special tokens, such as the end of turn token.
            output_text = tokenizer.decode(outputs[0][num_input_tokens:], skip_special_tokens=True)

            # Since an action is a chat completion response, we have to build one here.
            action = llm_clients.LLMResponse(
                chat_completion_response=openai.types.chat.chat_completion.ChatCompletion(
                    id=str(uuid.uuid4()),
                    choices=[
                        openai.types.chat.chat_completion.Choice(
                            message=openai.types.chat.chat_completion_message.ChatCompletionMessage(
                                content=output_text,
                                role="assistant",
                            ),
                            finish_reason="stop",
                            index=0,
                        )
                    ],
                    created=int(time.time()),
                    model="Qwen/Qwen2.5-3B-Instruct",
                    object="chat.completion",
                    usage=openai.types.completion_usage.CompletionUsage(
                        prompt_tokens=num_input_tokens,
                        completion_tokens=num_output_tokens,
                        total_tokens=num_input_tokens + num_output_tokens,
                    ),
                ),
                cost=0.0,
            )

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
