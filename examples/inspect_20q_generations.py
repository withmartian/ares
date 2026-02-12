#!/usr/bin/env python

# # Inspect Twenty Questions Model Generations
#
# Runs a single model on the Twenty Questions environment and prints
# detailed per-step information so you can inspect exactly what the
# model sees and generates at every turn.
#
# ## Run
#   uv run -m examples.inspect_20q_generations

import asyncio

import ares
from ares.contrib.mech_interp.hooked_transformer_client import HookedTransformerLLMClient, create_hooked_transformer_client_with_chat_template
from ares.environments import twenty_questions
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
#MODEL_LIST = ["gpt2", "Qwen/Qwen2.5-3B-Instruct"]
DEVICE = "mps"
ORACLE_MODEL = "openai/gpt-4o"
N_EPISODES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_reward(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    import numpy as np

    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[-1])


def _print_separator(char: str = "=", width: int = 80) -> None:
    print(char * width)


def _print_step(step_num: int, input_text: str, output_text: str, oracle_answer: str | None, reward: float) -> None:
    """Print detailed info for a single step."""
    print(f"\n  --- Step {step_num} ---")
    print(f"  INPUT (what the model sees):")
    # Indent each line of the input for readability
    for line in input_text.strip().splitlines():
        print(f"    | {line}")
    print()
    print(f"  OUTPUT (what the model generated):")
    for line in output_text.strip().splitlines():
        print(f"    > {line}")
    print()
    if oracle_answer is not None:
        print(f"  ORACLE ANSWER: {oracle_answer}")
    print(f"  STEP REWARD: {reward}")


# ---------------------------------------------------------------------------
# Patched client that captures formatted input
# ---------------------------------------------------------------------------


class InspectableClient:
    """Wraps HookedTransformerLLMClient to capture the formatted input text."""

    def __init__(self, client: HookedTransformerLLMClient) -> None:
        self._client = client
        self.last_input_text: str = ""
        self.last_output_text: str = ""

    async def __call__(self, request, **kwargs):
        # Capture what the model actually sees (the formatted prompt)
        messages_list = list(request.messages)
        self.last_input_text = self._client.format_messages_fn(messages_list)

        # Run the actual client
        response = await self._client(request, **kwargs)

        # Capture what the model generated
        self.last_output_text = response.data[0].content if response.data else ""

        return response


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_episodes() -> None:
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inner_client = create_hooked_transformer_client_with_chat_template(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        verbose=False,
    )
    client = InspectableClient(inner_client)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Model loaded: n_layers={n_layers}, d_model={d_model}, n_ctx={model.cfg.n_ctx}")

    for ep in range(N_EPISODES):
        _print_separator("=")
        print(f"EPISODE {ep}")
        _print_separator("=")

        async with twenty_questions.TwentyQuestionsEnvironment(oracle_model=ORACLE_MODEL) as env:
            ts = await env.reset()

            # Print the hidden object (we can get it from the env internals)
            hidden = env._hidden_object
            print(f"Hidden object: {hidden}")

            step_num = 0
            while not ts.last():
                # Get the agent's response
                assert ts.observation is not None
                action = await client(ts.observation)

                # The env will process this action and call the oracle
                prev_history_len = len(env._conversation_history)
                ts = await env.step(action)
                step_num += 1

                # Extract the oracle answer from the conversation history
                oracle_answer = None
                if len(env._conversation_history) > prev_history_len:
                    last_entry = env._conversation_history[-1]
                    if last_entry.startswith("A:"):
                        oracle_answer = last_entry

                reward = _scalar_reward(ts.reward)
                _print_step(
                    step_num=step_num,
                    input_text=client.last_input_text,
                    output_text=client.last_output_text,
                    oracle_answer=oracle_answer,
                    reward=reward,
                )

            # Episode summary
            final_reward = _scalar_reward(ts.reward)
            success = int(final_reward == 0.0 and ts.last())
            print()
            _print_separator("-")
            print(f"EPISODE {ep} SUMMARY: steps={step_num}, final_reward={final_reward:+.1f}, success={success}")
            _print_separator("-")

    del model, tokenizer, client


def main() -> None:
    asyncio.run(run_episodes())


if __name__ == "__main__":
    main()
