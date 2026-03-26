"""LLM client using HuggingFace transformers with automatic request batching.

WARNING: This is not a core ARES component. The interface may change without notice.
For production use, consider implementing your own LLM client following the LLMClient protocol.

Required dependency group:
    uv add withmartian-ares[transformers_client]
    OR
    uv sync --group transformers_client

Example usage:
    from ares.contrib import transformers_client
    from ares.llms import open_responses

    client = transformers_client.TransformersLLMClient(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        batch_wait_ms=50,
        max_batch_size=8,
    )

    req = open_responses.make_request([open_responses.user_message("Hello!")])
    response = await client(req)
"""

import asyncio
import collections
import contextlib
import dataclasses
import functools
import json
import logging
from typing import Literal, cast

from linguafranca import types as lft
import torch
import transformers

from ares.async_utils import ValueAndFuture
from ares.llms import llm_clients
from ares.llms import open_responses
from ares.llms import response

_LOGGER = logging.getLogger(__name__)
_SUPPORTED_RENDER_FIELDS = frozenset({"input", "instructions", "max_output_tokens", "model", "temperature", "top_p"})


# This is defined in transformers, but not exposed.
# We re-create it here to enable better type hints.
class _BaseModelWithGenerate(transformers.PreTrainedModel, transformers.GenerationMixin):
    pass


def _render_content_to_text(content: object, *, context: str) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    text_parts: list[str] = []
    dropped_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in {"input_text", "text"}:
            text_parts.append(str(part.get("text", "")))
            continue
        dropped_parts.append(
            str(part.get("type", type(part).__name__)) if isinstance(part, dict) else type(part).__name__
        )

    if dropped_parts:
        _LOGGER.warning(
            "TransformersLLMClient dropped unsupported %s parts: %s",
            context,
            ", ".join(dropped_parts),
        )

    return "".join(text_parts)


def _render_value_to_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _render_request_to_chat_messages(request: lft.OpenResponsesRequest) -> list[dict[str, str]]:
    """Convert an Open Responses request to simple ``{role, content}`` chat dicts.

    We use a custom renderer instead of ``open_responses.to_chat_messages()`` because
    local model tokenizers (via ``apply_chat_template``) generally don't handle OpenAI-
    format ``tool_calls`` arrays or ``role="tool"`` messages.  This function flattens
    tool interactions into plain user/assistant text that any chat template can process.
    """
    payload = open_responses.request_to_jsonable(request)

    dropped_fields = sorted(
        field
        for field, value in payload.items()
        if field not in _SUPPORTED_RENDER_FIELDS and value not in (None, False, [], {})
    )
    if dropped_fields:
        _LOGGER.warning(
            "TransformersLLMClient dropped unsupported Open Responses fields: %s",
            ", ".join(dropped_fields),
        )

    system_parts: list[str] = []
    instructions = payload.get("instructions")
    if instructions:
        system_parts.append(str(instructions))

    rendered_messages: list[dict[str, str]] = []
    input_value = payload["input"]
    if isinstance(input_value, str):
        rendered_messages.append({"role": "user", "content": input_value})
    else:
        for item in input_value:
            if not isinstance(item, dict):
                _LOGGER.warning(
                    "TransformersLLMClient dropped unsupported input item type: %s",
                    type(item).__name__,
                )
                continue

            item_type = item.get("type")
            if item_type == "message":
                role = str(item.get("role", "user"))
                content = _render_content_to_text(item.get("content"), context=f"{role} message")
                if role in {"developer", "system"}:
                    system_parts.append(content)
                elif role in {"assistant", "user"}:
                    rendered_messages.append({"role": role, "content": content})
                else:
                    _LOGGER.warning("TransformersLLMClient dropped unsupported message role: %s", role)
                continue

            if item_type == "function_call":
                rendered_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Called tool {item.get('name', '')} with call_id {item.get('call_id', '')}:\n"
                            f"{item.get('arguments', '')}"
                        ),
                    }
                )
                continue

            if item_type == "function_call_output":
                rendered_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool result for call_id {item.get('call_id', '')}:\n"
                            f"{_render_value_to_text(item.get('output', ''))}"
                        ),
                    }
                )
                continue

            _LOGGER.warning("TransformersLLMClient dropped unsupported input item type: %s", item_type)

    if system_parts:
        rendered_messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    return rendered_messages


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


@dataclasses.dataclass(frozen=True, kw_only=True)
class TransformersLLMClient(llm_clients.LLMClient):
    """LLM client with automatic request batching.

    Collects requests over a time window and processes them in batches for efficiency.

    Attributes:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        batch_wait_ms: Max milliseconds to wait collecting requests for a batch (default: 50)
        max_batch_size: Maximum number of requests per batch (default: 8)
        device: Device to use - "auto", "cuda", "mps", or "cpu" (default: "auto")
        torch_dtype: Precision - "auto", "float32", "float16", "bfloat16" (default: "auto")
        max_new_tokens: Maximum tokens to generate per request (default: 512)
        temperature: Default temperature for generation (default: 1.0)
    """

    model_name: str
    batch_wait_ms: int = 50
    max_batch_size: int = 8
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    max_new_tokens: int = 512
    temperature: float = 1.0

    @functools.cached_property
    def _request_queue(self) -> asyncio.Queue[ValueAndFuture[lft.OpenResponsesRequest, response.LLMResponse]]:
        """Lazy-initialized queue for batching requests."""
        return asyncio.Queue()

    @functools.cached_property
    def _inference_task(self) -> asyncio.Task[None]:
        """Lazy-initialized background inference task."""
        task = asyncio.create_task(self._inference_loop())
        _LOGGER.info("TransformersLLMClient started background inference task")
        return task

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

    async def __call__(self, req: lft.OpenResponsesRequest) -> response.LLMResponse:
        """Queue request and wait for batched inference.

        The background inference task is started automatically on first call.

        Args:
            req: The Open Responses request to run.

        Returns:
            LLMResponse with the generated completion
        """
        _ = self._inference_task  # Trigger lazy initialization
        future: asyncio.Future[response.LLMResponse] = asyncio.Future()
        await self._request_queue.put(ValueAndFuture(value=req, future=future))
        return await future

    async def close(self) -> None:
        """Stop the background inference task."""
        self._inference_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._inference_task

    async def __aenter__(self) -> "TransformersLLMClient":
        if self._inference_task.done():
            raise RuntimeError("TransformersLLMClient has already been closed and cannot be reused")
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    async def _inference_loop(self) -> None:
        """Background task that batches and processes requests.

        Groups requests by (temperature, max_tokens, top_p) during collection to preserve per-request
        semantics. Collects until batch_wait_ms elapses OR any group reaches max_batch_size.

        Tradeoff: Grouping at collection time is more efficient than collecting mixed batches
        and splitting later, but less efficient than a full meta-queue with per-parameter timers.
        Future: Could extend to a queue-of-queues where each parameter combo gets its own queue
        and timer, allowing truly independent batching per parameter set.
        """
        while True:
            collected_items: list[ValueAndFuture[lft.OpenResponsesRequest, response.LLMResponse]] = []
            try:
                first_item = await self._request_queue.get()
                collected_items.append(first_item)
                temp = first_item.value.temperature
                max_tokens = first_item.value.max_output_tokens
                top_p = first_item.value.top_p
                first_params = (
                    self.temperature if temp is None else temp,
                    self.max_new_tokens if max_tokens is None else max_tokens,
                    top_p,
                )

                groups: dict[
                    tuple[float, int, float | None],
                    list[ValueAndFuture[lft.OpenResponsesRequest, response.LLMResponse]],
                ] = collections.defaultdict(list)
                groups[first_params].append(first_item)

                deadline = asyncio.get_event_loop().time() + (self.batch_wait_ms / 1000.0)

                while (
                    asyncio.get_event_loop().time() < deadline
                    and max((len(group) for group in groups.values()), default=0) < self.max_batch_size
                ):
                    timeout = max(0, deadline - asyncio.get_event_loop().time())

                    try:
                        item = await asyncio.wait_for(self._request_queue.get(), timeout=timeout)
                        collected_items.append(item)
                        temp = item.value.temperature
                        max_tokens = item.value.max_output_tokens
                        top_p = item.value.top_p
                        params = (
                            self.temperature if temp is None else temp,
                            self.max_new_tokens if max_tokens is None else max_tokens,
                            top_p,
                        )
                        groups[params].append(item)
                    except TimeoutError:
                        break

                for params, group in groups.items():
                    if group:
                        _LOGGER.debug("Processing batch of %d request(s) for params %s", len(group), params)
                        await self._process_batch(group, params)

            except Exception as e:
                _LOGGER.exception("Error in inference loop: %s", e)
                for item in collected_items:
                    if not item.future.done():
                        item.future.set_exception(e)

    async def _process_batch(
        self,
        batch: list[ValueAndFuture[lft.OpenResponsesRequest, response.LLMResponse]],
        params: tuple[float, int, float | None],
    ) -> None:
        """Process a batch of requests with homogeneous parameters.

        Args:
            batch: List of request-future pairs (all with same temperature/max_tokens)
            params: (temperature, max_new_tokens, top_p) for this batch
        """
        try:
            temperature, max_new_tokens, top_p = params

            chat_conversations = []
            for item in batch:
                chat_conversations.append(_render_request_to_chat_messages(item.value))

            responses = await asyncio.to_thread(  # transformers is not async
                self._generate_batch,
                chat_conversations,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )

            for item, resp in zip(batch, responses, strict=True):
                item.future.set_result(resp)

        except Exception as e:
            _LOGGER.exception("Error processing batch: %s", e)
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    def _generate_batch(
        self,
        chat_conversations: list[list[dict[str, str]]],
        temperature: float,
        max_new_tokens: int,
        top_p: float | None,
    ) -> list[response.LLMResponse]:
        """Generate responses for a batch of chat conversations.

        Args:
            chat_conversations: List of chat message lists
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            top_p: Optional top-p sampling parameter

        Returns:
            List of LLMResponses
        """
        input_texts: list[str] = []
        for conv in chat_conversations:
            text = self._tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
            assert isinstance(text, str)  # apply_chat_template with tokenize=False returns str
            input_texts.append(text)

        # TODO: Add truncation strategies.
        # see https://github.com/withmartian/ares/pull/89#discussion_r2805921801
        inputs: transformers.BatchEncoding = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        input_lengths = inputs["input_ids"].shape[1]  # type: ignore[union-attr]

        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if top_p is not None:
                generation_kwargs["top_p"] = top_p

            outputs = self._model.generate(
                **inputs,  # type: ignore[arg-type]
                **generation_kwargs,  # type: ignore[arg-type]
            )

        generated_ids = outputs[:, input_lengths:]
        decoded_responses = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        responses = []
        for i, text in enumerate(decoded_responses):
            prompt_tokens = inputs["input_ids"][i].ne(self._tokenizer.pad_token_id).sum().item()  # type: ignore[union-attr]
            generated_tokens = generated_ids[i].ne(self._tokenizer.pad_token_id).sum().item()  # type: ignore[arg-type]

            responses.append(
                response.LLMResponse(
                    data=[response.TextData(content=text)],
                    cost=0.0,  # Local inference has no API cost
                    usage=response.Usage(
                        prompt_tokens=prompt_tokens,
                        generated_tokens=generated_tokens,
                    ),
                )
            )

        return responses
