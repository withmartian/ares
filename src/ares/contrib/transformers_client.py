"""LLM client using HuggingFace transformers with batching and activation hooks.

This client supports automatic request batching and activation capture for mechanistic
interpretability research (e.g., linear probes on hidden states).

WARNING: This is not a core ARES component. The interface may change without notice.
For production use, consider implementing your own LLM client following the LLMClient protocol.

Required dependency group:
    uv add withmartian-ares[transformers_client]
    OR
    uv sync --group transformers_client

Example usage:
    from ares.contrib import transformers_client
    from ares.llms import request

    # Initialize with a HuggingFace model
    client = transformers_client.TransformersLLMClient(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        batch_wait_ms=50,
        max_batch_size=8,
    )

    # Use like any other LLM client (no context manager needed)
    req = request.LLMRequest(messages=[{"role": "user", "content": "Hello!"}])
    response = await client(req)

    # With activation hooks for mech interp
    activations = []
    def capture_activations(acts):
        activations.append(acts)

    client = transformers_client.TransformersLLMClient(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        activation_callback=capture_activations,
        output_hidden_states=True,
    )

Note: The background batching task runs until the client is garbage collected.
For explicit cleanup, call client.close() or use try/finally.
"""

import asyncio
from collections.abc import Callable
import dataclasses
import functools
import logging
from typing import Any, Literal, cast
import weakref

import torch
import transformers

from ares.async_utils import ValueAndFuture
from ares.llms import llm_clients
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


# This is defined in transformers, but not exposed.
# We re-create it here to enable better type hints.
class _BaseModelWithGenerate(transformers.PreTrainedModel, transformers.GenerationMixin):
    pass


def _detect_device() -> str:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_torch_dtype(dtype_str: str, device: str) -> torch.dtype:
    """Get torch dtype, with device compatibility checks.

    Args:
        dtype_str: One of "float32", "float16", "bfloat16", "auto"
        device: Target device ("cuda", "mps", "cpu")

    Returns:
        Appropriate torch dtype for the device
    """
    if dtype_str == "auto":
        # Use bfloat16 on CUDA if available, float16 on MPS, float32 on CPU
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif device == "mps":
            return torch.float16  # MPS has better float16 support
        return torch.float32

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    selected_dtype = dtype_map[dtype_str]

    # Warn about potential compatibility issues
    if device == "mps" and selected_dtype == torch.bfloat16:
        _LOGGER.warning("bfloat16 may not be well-supported on MPS, consider float16 instead")
    elif device == "cpu" and selected_dtype in (torch.float16, torch.bfloat16):
        _LOGGER.warning("float16/bfloat16 may be slow on CPU, consider float32 instead")

    return selected_dtype


@dataclasses.dataclass(kw_only=True)
class TransformersLLMClient(llm_clients.LLMClient):
    """LLM client with automatic batching and activation hooks for mech interp.

    Supports automatic request batching by collecting requests over a time window,
    then running batch inference. Also supports capturing activations (hidden states,
    attention weights) for mechanistic interpretability research.

    Attributes:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        batch_wait_ms: Max milliseconds to wait collecting requests for a batch (default: 50)
        max_batch_size: Maximum number of requests per batch (default: 8)
        device: Device to use - "auto", "cuda", "mps", or "cpu" (default: "auto")
        torch_dtype: Precision - "auto", "float32", "float16", "bfloat16" (default: "auto")
        max_new_tokens: Maximum tokens to generate per request (default: 512)
        temperature: Default temperature for generation (default: 1.0)
        output_hidden_states: Capture hidden states from all layers (default: False)
        output_attentions: Capture attention weights from all layers (default: False)
        activation_callback: Optional callback receiving activations dict (default: None)
            Called with dict containing:
            - "hidden_states": tuple of tensors (layer_count, batch_size, seq_len, hidden_dim)
            - "attentions": tuple of tensors (layer_count, batch_size, num_heads, seq_len, seq_len)
            - "input_ids": tensor (batch_size, seq_len)
            - "generated_ids": tensor (batch_size, generated_len)
    """

    model_name: str
    batch_wait_ms: int = 50
    max_batch_size: int = 8
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_new_tokens: int = 512
    temperature: float = 1.0
    output_hidden_states: bool = False
    output_attentions: bool = False
    activation_callback: Callable[[dict[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable state for async batching.

        Note: The background inference task is started lazily on first request
        and runs until the client object is garbage collected.
        """
        self._request_queue: asyncio.Queue[ValueAndFuture[request.LLMRequest, response.LLMResponse]] = asyncio.Queue()
        self._inference_task: asyncio.Task[None] | None = None

    @functools.cached_property
    def _device(self) -> str:
        """Resolved device."""
        if self.device == "auto":
            return _detect_device()
        return self.device

    @functools.cached_property
    def _dtype(self) -> torch.dtype:
        """Resolved torch dtype."""
        return _get_torch_dtype(self.torch_dtype, self._device)

    @functools.cached_property
    def _model(self) -> _BaseModelWithGenerate:
        """Load model with specified precision and device."""
        _LOGGER.info(
            "Loading model %s on device=%s with dtype=%s",
            self.model_name,
            self._device,
            self._dtype,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self._dtype,
        )
        model = model.to(self._device)  # type: ignore[arg-type]
        model.eval()  # Set to eval mode for inference
        assert hasattr(model, "generate")
        assert isinstance(model.generate, Callable)
        return cast(_BaseModelWithGenerate, model)

    @functools.cached_property
    def _tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Load tokenizer."""
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        # Set padding side to left for batch generation
        tokenizer.padding_side = "left"
        # Use eos_token as pad_token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _ensure_inference_task_started(self) -> None:
        """Lazy-start the background inference task if not already running."""
        if self._inference_task is None or self._inference_task.done():
            # Create weakref to self so task can detect when client is GC'd
            weak_self = weakref.ref(self)
            self._inference_task = asyncio.create_task(self._inference_loop(weak_self))
            _LOGGER.info("TransformersLLMClient started background inference task")

    async def __call__(self, req: request.LLMRequest) -> response.LLMResponse:
        """Queue request and wait for batched inference.

        The background inference task is started automatically on first call.

        Args:
            req: The LLM request containing messages and optional temperature

        Returns:
            LLMResponse with the generated completion
        """
        # Lazy-start the background task if needed
        self._ensure_inference_task_started()

        # Create future for this request
        future: asyncio.Future[response.LLMResponse] = asyncio.Future()

        # Queue request with its future
        await self._request_queue.put(ValueAndFuture(value=req, future=future))

        # Wait for inference result
        return await future

    async def _inference_loop(self, weak_self: weakref.ReferenceType) -> None:
        """Background task that batches and processes requests.

        Args:
            weak_self: Weakref to the client - task exits when client is GC'd
        """
        while weak_self() is not None:
            try:
                # Wait for first request (with timeout to check _running periodically)
                try:
                    first_item = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=60.0,  # Check _running every 60s
                    )
                except TimeoutError:
                    continue

                # Collect batch of requests
                batch: list[ValueAndFuture[request.LLMRequest, response.LLMResponse]] = [first_item]

                # Wait for more requests up to batch_wait_ms
                deadline = asyncio.get_event_loop().time() + (self.batch_wait_ms / 1000.0)

                while len(batch) < self.max_batch_size:
                    timeout = max(0, deadline - asyncio.get_event_loop().time())
                    if timeout <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
                        batch.append(item)
                    except TimeoutError:
                        break

                # Process batch
                _LOGGER.debug("Processing batch of %d request(s)", len(batch))
                await self._process_batch(batch)

            except Exception as e:
                _LOGGER.exception("Error in inference loop: %s", e)

    async def _process_batch(
        self,
        batch: list[ValueAndFuture[request.LLMRequest, response.LLMResponse]],
    ) -> None:
        """Process a batch of requests using batch inference.

        Args:
            batch: List of request-future pairs
        """
        try:
            # Extract requests
            requests = [item.value for item in batch]

            # Convert requests to chat format and prepare inputs
            chat_conversations = []
            temperatures = []
            max_tokens_list = []

            for req in requests:
                # Convert to chat completion format
                kwargs = req.to_chat_completion_kwargs()
                chat_conversations.append(kwargs["messages"])
                temperatures.append(kwargs.get("temperature", self.temperature))
                max_tokens_list.append(kwargs.get("max_completion_tokens", self.max_new_tokens))

            # Use the max of all max_tokens requests for the batch
            max_new_tokens = max(max_tokens_list)
            # Use the mean temperature (or could use per-request temperatures with generate's generation_config)
            temperature = sum(temperatures) / len(temperatures)

            # Run inference in thread pool (transformers is not async)
            responses = await asyncio.to_thread(
                self._generate_batch,
                chat_conversations,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

            # Set futures with results
            for item, resp in zip(batch, responses, strict=True):
                item.future.set_result(resp)

        except Exception as e:
            _LOGGER.exception("Error processing batch: %s", e)
            # Set exception on all futures
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def _generate_batch(
        self,
        chat_conversations: list[list[dict[str, str]]],
        temperature: float,
        max_new_tokens: int,
    ) -> list[response.LLMResponse]:
        """Generate responses for a batch of chat conversations.

        Args:
            chat_conversations: List of chat message lists
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of LLMResponses
        """
        # Apply chat template to all conversations
        input_texts: list[str] = []
        for conv in chat_conversations:
            text = self._tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            # apply_chat_template with tokenize=False returns str
            assert isinstance(text, str)
            input_texts.append(text)

        # Tokenize batch
        inputs: transformers.BatchEncoding = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        input_lengths = inputs["input_ids"].shape[1]  # type: ignore[union-attr]

        # Generate with optional activation outputs
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,  # type: ignore[arg-type]
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                output_hidden_states=self.output_hidden_states,
                output_attentions=self.output_attentions,
                return_dict_in_generate=True,
            )

        # Extract generated tokens (remove input prefix)
        generated_ids = outputs.sequences[:, input_lengths:]  # type: ignore[union-attr]

        # Decode responses
        decoded_responses = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Call activation callback if provided
        if self.activation_callback and (self.output_hidden_states or self.output_attentions):
            activation_dict: dict[str, Any] = {
                "input_ids": inputs["input_ids"].cpu(),  # type: ignore[union-attr]
                "generated_ids": generated_ids.cpu(),
            }

            if self.output_hidden_states:
                # hidden_states: tuple of tuples - (generation_step, (layer, batch, seq, hidden))
                # For mech interp, we typically want the final hidden states per layer
                activation_dict["hidden_states"] = outputs.hidden_states  # type: ignore[union-attr]

            if self.output_attentions:
                activation_dict["attentions"] = outputs.attentions  # type: ignore[union-attr]

            self.activation_callback(activation_dict)

        # Create LLMResponse objects
        responses = []
        for i, text in enumerate(decoded_responses):
            prompt_tokens = inputs["input_ids"][i].ne(self._tokenizer.pad_token_id).sum().item()  # type: ignore[union-attr]
            generated_tokens = len(generated_ids[i])

            responses.append(
                response.LLMResponse(
                    data=[response.TextData(content=text)],
                    cost=0.0,  # Local inference, no cost
                    usage=response.Usage(
                        prompt_tokens=prompt_tokens,
                        generated_tokens=generated_tokens,
                    ),
                )
            )

        return responses


# Convenience factory for popular small models
create_qwen2_0_5b_instruct_client = functools.partial(
    TransformersLLMClient,
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
)
