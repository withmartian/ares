"""Minimal SeedRL-style architecture for SWE-bench using asyncio.

This example demonstrates a distributed RL architecture with:
- Multiple async actors running SweBenchEnv episodes in parallel
- A centralized Learner that handles model inference
- Queue-based communication between actors and learner
- A minimal C51 distributional value head (structure only, not trained)

This is a demo of the SeedRL architecture pattern, not a full trainer.

Example usage:

    1. Make sure you have examples dependencies installed
       `uv sync --group examples`
    2. Run the example
       `uv run -m examples.03_seedrl_swebench`
"""

import asyncio
import dataclasses
import time
import uuid
from typing import Any

import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage
import torch
import torch.nn as nn
import transformers

from ares.environments import swebench_env
from ares.llms import llm_clients


# ============================================================================
# Value Head (C51 Distributional Critic)
# ============================================================================


class C51ValueHead(nn.Module):
    """Minimal C51-style distributional value head.

    This demonstrates where a value head would attach to model hidden states.
    In a full implementation, this would be trained alongside the policy to
    predict value distributions for RL algorithms like IMPALA or R2D2.

    Args:
        hidden_size: Size of the input hidden states from the transformer
        num_atoms: Number of atoms in the categorical distribution (C51 default: 51)
        v_min: Minimum value of the support
        v_max: Maximum value of the support
    """

    def __init__(
        self,
        hidden_size: int = 1536,  # Qwen2.5-1.5B hidden size
        num_atoms: int = 51,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        # Project hidden states to atom logits
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_atoms),
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass of the value head.

        Args:
            hidden_states: Transformer hidden states, shape [batch, seq_len, hidden_size]

        Returns:
            Dictionary with:
                - logits: Raw logits over atoms, shape [batch, seq_len, num_atoms]
                - probs: Softmax probabilities, shape [batch, seq_len, num_atoms]
                - value: Expected value (dot product of probs and support)
        """
        # Take the last token's hidden state for value prediction
        last_hidden = hidden_states[:, -1, :]  # [batch, hidden_size]

        logits = self.value_head(last_hidden)  # [batch, num_atoms]
        probs = torch.softmax(logits, dim=-1)

        # Expected value: E[Z] = sum(p_i * z_i)
        value = (probs * self.support).sum(dim=-1, keepdim=True)  # [batch, 1]

        return {
            "logits": logits,
            "probs": probs,
            "value": value,
        }


# ============================================================================
# Inference Request & Response
# ============================================================================


@dataclasses.dataclass
class InferenceRequest:
    """Request from an actor to the learner for inference."""
    actor_id: int
    request: llm_clients.LLMRequest
    response_future: asyncio.Future[llm_clients.LLMResponse]


# ============================================================================
# Learner (Centralized Inference)
# ============================================================================


class Learner:
    """Centralized learner that handles model inference for all actors.

    In a full SeedRL implementation, the learner would also:
    - Receive trajectories from actors
    - Update model parameters via gradient descent
    - Periodically sync updated weights back to actors

    For this demo, we focus on the inference path.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str | None = None,
        max_batch_size: int = 8,
        batch_timeout: float = 0.5,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout

        print(f"[Learner] Loading model {model_name} on {self.device}...")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Inference mode

        # Attach the C51 value head (not trained, just for demonstration)
        hidden_size = self.model.config.hidden_size
        self.value_head = C51ValueHead(hidden_size=hidden_size).to(self.device)
        print(f"[Learner] Attached C51 value head with {hidden_size}-dim inputs")
        print(f"[Learner] Batching enabled: max_batch_size={max_batch_size}, "
              f"batch_timeout={batch_timeout}s")

        self.request_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue()
        self.inference_task: asyncio.Task[None] | None = None

        # Stats
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_batches = 0
        self.max_batch_size_seen = 0

    async def start(self) -> None:
        """Start the inference loop."""
        print("[Learner] Starting inference loop...")
        self.inference_task = asyncio.create_task(self._inference_loop())

    async def stop(self) -> None:
        """Stop the inference loop."""
        if self.inference_task:
            self.inference_task.cancel()
            try:
                await self.inference_task
            except asyncio.CancelledError:
                pass
        avg_batch_size = self.total_requests / self.total_batches if self.total_batches > 0 else 0
        print(f"[Learner] Stopped. Processed {self.total_requests} requests "
              f"in {self.total_batches} batches, "
              f"generated {self.total_tokens_generated} tokens.")
        print(f"[Learner] Batch stats: avg_size={avg_batch_size:.2f}, "
              f"max_size={self.max_batch_size_seen}")

    async def request_inference(self, actor_id: int, request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
        """Submit an inference request and await the response.

        Args:
            actor_id: ID of the requesting actor
            request: LLMRequest containing messages

        Returns:
            LLMResponse with the model's generated text
        """
        response_future: asyncio.Future[llm_clients.LLMResponse] = asyncio.Future()
        inference_request = InferenceRequest(
            actor_id=actor_id,
            request=request,
            response_future=response_future,
        )
        await self.request_queue.put(inference_request)
        return await response_future

    async def _inference_loop(self) -> None:
        """Main inference loop that processes requests from the queue.

        This implements continuous batching:
        - Collects requests up to max_batch_size OR waits up to batch_timeout
        - Batches multiple requests together for efficiency
        - Uses dynamic batching with padding for variable-length sequences
        - Distributes responses back to the correct futures
        """
        print("[Learner] Inference loop running...")

        while True:
            try:
                batch: list[InferenceRequest] = []

                # Get first request (blocks until available)
                first_req = await self.request_queue.get()
                batch.append(first_req)

                # Collect additional requests up to max_batch_size or batch_timeout
                batch_start_time = time.time()
                while len(batch) < self.max_batch_size:
                    time_remaining = self.batch_timeout - (time.time() - batch_start_time)
                    if time_remaining <= 0:
                        break

                    try:
                        # Try to get more requests with timeout
                        req = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=time_remaining
                        )
                        batch.append(req)
                    except asyncio.TimeoutError:
                        # Timeout reached, process current batch
                        break

                # Update stats
                batch_size = len(batch)
                self.total_batches += 1
                self.max_batch_size_seen = max(self.max_batch_size_seen, batch_size)

                # Run batched inference in a thread to avoid blocking the event loop
                responses = await asyncio.to_thread(self._run_batched_inference, batch)

                # Distribute responses back to the correct futures
                for req, response in zip(batch, responses):
                    req.response_future.set_result(response)

                self.total_requests += batch_size

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Learner] Error in inference loop: {e}")
                # Set exception on all requests in the batch that haven't been completed
                for req in batch:
                    if not req.response_future.done():
                        req.response_future.set_exception(e)

    def _run_inference(self, req: InferenceRequest) -> llm_clients.LLMResponse:
        """Run inference on a single request (blocking call, runs in thread pool).

        Args:
            req: InferenceRequest from an actor

        Returns:
            LLMResponse with generated text and value head output
        """
        # Tokenize the chat messages
        inputs = self.tokenizer.apply_chat_template(
            req.request.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=req.request.temperature or 1.0,
                do_sample=True if req.request.temperature else False,
            )

            # Demonstrate value head usage:
            # Get hidden states for the last input token
            # (In a real trainer, you'd do this on full trajectories)
            model_outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden_states = model_outputs.hidden_states[-1]  # Last layer
            value_output = self.value_head(last_hidden_states)

            # Print value head output shape once for demonstration
            if self.total_requests == 0:
                print(f"[Learner] Value head output shapes:")
                print(f"  - logits: {value_output['logits'].shape}")
                print(f"  - probs: {value_output['probs'].shape}")
                print(f"  - value: {value_output['value'].shape}")
                print(f"  - estimated value: {value_output['value'].item():.4f}")

        # Decode output
        num_input_tokens = inputs["input_ids"].shape[-1]
        num_output_tokens = outputs[0].shape[-1] - num_input_tokens
        output_text = self.tokenizer.decode(
            outputs[0][num_input_tokens:],
            skip_special_tokens=True,
        )

        self.total_tokens_generated += num_output_tokens

        # Wrap in OpenAI-compatible response
        response = llm_clients.LLMResponse(
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
                model=self.model_name,
                object="chat.completion",
                usage=openai.types.completion_usage.CompletionUsage(
                    prompt_tokens=num_input_tokens,
                    completion_tokens=num_output_tokens,
                    total_tokens=num_input_tokens + num_output_tokens,
                ),
            ),
            cost=0.0,
        )

        return response

    def _run_batched_inference(
        self, batch: list[InferenceRequest]
    ) -> list[llm_clients.LLMResponse]:
        """Run inference on a batch of requests with proper padding (blocking call, runs in thread pool).

        Args:
            batch: List of InferenceRequests to process together

        Returns:
            List of LLMResponses, one per request in the same order
        """
        batch_size = len(batch)

        if batch_size == 1:
            # Fall back to single inference for batch size 1
            return [self._run_inference(batch[0])]

        # Tokenize all requests separately first
        tokenized_inputs = []
        for req in batch:
            inputs = self.tokenizer.apply_chat_template(
                req.request.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            tokenized_inputs.append(inputs)

        # Find the maximum sequence length in the batch
        max_length = max(inputs["input_ids"].shape[1] for inputs in tokenized_inputs)

        # Pad all inputs to the same length (left padding for decoder-only models)
        padded_input_ids = []
        padded_attention_masks = []

        for inputs in tokenized_inputs:
            input_ids = inputs["input_ids"][0]  # Remove batch dimension
            attention_mask = inputs["attention_mask"][0]

            # Calculate padding needed
            padding_length = max_length - len(input_ids)

            if padding_length > 0:
                # Left padding for decoder-only models
                pad_token_id = self.tokenizer.pad_token_id
                input_ids = torch.cat([
                    torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype),
                    input_ids
                ])
                attention_mask = torch.cat([
                    torch.zeros(padding_length, dtype=attention_mask.dtype),
                    attention_mask
                ])

            padded_input_ids.append(input_ids)
            padded_attention_masks.append(attention_mask)

        # Stack into batched tensors
        batched_input_ids = torch.stack(padded_input_ids).to(self.device)
        batched_attention_mask = torch.stack(padded_attention_masks).to(self.device)

        # Get generation parameters from requests (use first request's params)
        temperature = batch[0].request.temperature or 1.0
        do_sample = True if batch[0].request.temperature else False

        # Generate completions for the entire batch
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                max_new_tokens=2048,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # Get value head outputs for the batch
            model_outputs = self.model(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                output_hidden_states=True,
            )
            last_hidden_states = model_outputs.hidden_states[-1]  # Last layer
            value_output = self.value_head(last_hidden_states)

            # Print value head output shape once for demonstration
            if self.total_requests == 0:
                print(f"[Learner] Value head output shapes (batch_size={batch_size}):")
                print(f"  - logits: {value_output['logits'].shape}")
                print(f"  - probs: {value_output['probs'].shape}")
                print(f"  - value: {value_output['value'].shape}")
                print(f"  - estimated values: {value_output['value'].squeeze().tolist()}")

        # Process outputs for each request in the batch
        responses = []
        for i, (req, original_inputs) in enumerate(zip(batch, tokenized_inputs)):
            # Calculate number of input tokens for this specific request
            num_input_tokens = original_inputs["input_ids"].shape[-1]

            # Decode output for this batch element
            output_tokens = outputs[i]
            num_output_tokens = len(output_tokens) - max_length

            # Extract only the generated tokens (after the padded input)
            generated_tokens = output_tokens[max_length:]
            output_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            self.total_tokens_generated += num_output_tokens

            # Wrap in OpenAI-compatible response
            response = llm_clients.LLMResponse(
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
                    model=self.model_name,
                    object="chat.completion",
                    usage=openai.types.completion_usage.CompletionUsage(
                        prompt_tokens=num_input_tokens,
                        completion_tokens=num_output_tokens,
                        total_tokens=num_input_tokens + num_output_tokens,
                    ),
                ),
                cost=0.0,
            )
            responses.append(response)

        return responses


# ============================================================================
# Actor (Environment Runner)
# ============================================================================


class Actor:
    """Actor that runs SweBenchEnv episodes and requests inference from the learner.

    In a full SeedRL implementation, actors would:
    - Collect full trajectories (obs, action, reward, done)
    - Send trajectories to the learner for training
    - Periodically fetch updated model weights from the learner

    For this demo, we just run the environment loop.
    """

    def __init__(
        self,
        actor_id: int,
        learner: Learner,
        task: swebench_env.SwebenchTask,
        max_steps: int = 5,
    ):
        self.actor_id = actor_id
        self.learner = learner
        self.task = task
        self.max_steps = max_steps

    async def run_episode(self) -> dict[str, Any]:
        """Run a single episode of the environment.

        Returns:
            Episode statistics (steps, reward, etc.)
        """
        print(f"[Actor {self.actor_id}] Starting episode on task {self.task.instance_id}")

        # Create environment for this task
        async with swebench_env.SweBenchEnv(tasks=[self.task]) as env:
            ts = await env.reset()
            step_count = 0

            while not ts.last() and step_count < self.max_steps:
                # Request inference from the centralized learner
                action = await self.learner.request_inference(self.actor_id, ts.observation)

                # Print step info
                _print_step(self.actor_id, step_count, ts.observation, action)

                # Step the environment
                ts = await env.step(action)
                step_count += 1

            print(f"[Actor {self.actor_id}] Episode finished: {step_count} steps, "
                  f"reward={ts.reward:.2f}")

            return {
                "actor_id": self.actor_id,
                "steps": step_count,
                "reward": ts.reward,
                "task_id": self.task.instance_id,
            }


# ============================================================================
# Main Orchestration
# ============================================================================


async def main():
    """Main entry point for the SeedRL demo.

    This demonstrates the architecture:
    1. Start a centralized Learner with a shared model
    2. Create N actors that run environments in parallel
    3. Actors request inference from the Learner via a queue
    4. Learner processes requests and returns actions
    """
    # Configuration
    n_actors = 1  # Start with 1, easily parameterize to scale up
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print("=" * 80)
    print("SeedRL-style SWE-bench Demo")
    print("=" * 80)
    print(f"Actors: {n_actors}")
    print(f"Model: {model_name}")
    print("=" * 80)

    # Load tasks
    all_tasks = swebench_env.swebench_verified_tasks()
    selected_tasks = [all_tasks[0]]  # Use first task for demo

    print(f"\nTask: {selected_tasks[0].instance_id}")
    print(f"Repo: {selected_tasks[0].repo}")
    print("-" * 80)

    # Initialize centralized learner
    learner = Learner(model_name=model_name)
    await learner.start()

    try:
        # Create actors
        actors = [
            Actor(actor_id=i, learner=learner, task=selected_tasks[0], max_steps=5)
            for i in range(n_actors)
        ]

        # Run actors in parallel (for this demo with 1 actor, it's just one task)
        # In a full system with n_actors > 1, this would run multiple episodes concurrently
        results = await asyncio.gather(*[actor.run_episode() for actor in actors])

        # Print summary
        print("\n" + "=" * 80)
        print("Episode Results")
        print("=" * 80)
        for result in results:
            print(f"Actor {result['actor_id']}: {result['steps']} steps, "
                  f"reward={result['reward']:.2f}")
        print("=" * 80)

    finally:
        # Clean up
        await learner.stop()


def _print_step(
    actor_id: int,
    step: int,
    observation: llm_clients.LLMRequest | None,
    action: llm_clients.LLMResponse,
) -> None:
    """Helper to print step information."""
    action_str = str(action.chat_completion_response.choices[0].message.content)[:100]
    observation_str = "No observation"
    if observation is not None:
        observation_content = list(observation.messages)[-1].get("content", "")
        observation_str = str(observation_content)[:100]

    print(f"\n[Actor {actor_id} | Step {step}]")
    print(f"Observation: {observation_str}")
    print(f"Action: {action_str}")


if __name__ == "__main__":
    asyncio.run(main())
