"""LLM client protocols and response types."""

import dataclasses
import time
from typing import Protocol
import uuid

import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage

from ares.llms import request


class OutputLengthExceededError(Exception):
    """Raised when LLM output exceeds the maximum allowed length.

    This matches the terminal-bench reference implementation's error handling.
    """

    def __init__(self, message: str, truncated_response: str | None = None):
        super().__init__(message)
        self.truncated_response = truncated_response


# TODO: Move to its own file.
# TODO: Improve the data model so it's not chat completions specific.
@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: openai.types.chat.chat_completion.ChatCompletion
    cost: float


class LLMClient(Protocol):
    async def __call__(self, request: request.LLMRequest) -> LLMResponse: ...


def build_openai_compatible_llm_response(
    output_text: str,
    num_input_tokens: int,
    num_output_tokens: int,
    model: str,
    cost: float = 0.0,
) -> LLMResponse:
    """Build an LLMResponse from raw generation outputs in OpenAI-compatible format.

    This helper constructs a complete OpenAI ChatCompletion-compatible LLMResponse
    from basic generation outputs. It's useful when working with local models or
    non-OpenAI endpoints that need to conform to the ARES LLMResponse interface.

    Args:
        output_text: The generated text content from the model.
        num_input_tokens: Number of tokens in the input/prompt.
        num_output_tokens: Number of tokens in the generated output.
        model: Model identifier string (e.g., "Qwen/Qwen2.5-3B-Instruct").
        cost: Cost of the API call in USD. Defaults to 0.0 for local models.

    Returns:
        LLMResponse: A complete response object with OpenAI-compatible structure.

    Example:
        >>> response = build_openai_compatible_llm_response(
        ...     output_text="Hello, world!",
        ...     num_input_tokens=10,
        ...     num_output_tokens=3,
        ...     model="Qwen/Qwen2.5-3B-Instruct",
        ... )
    """
    return LLMResponse(
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
            model=model,
            object="chat.completion",
            usage=openai.types.completion_usage.CompletionUsage(
                prompt_tokens=num_input_tokens,
                completion_tokens=num_output_tokens,
                total_tokens=num_input_tokens + num_output_tokens,
            ),
        ),
        cost=cost,
    )
