"""LLM response model."""

import dataclasses
from typing import Any, Literal
import uuid

import anthropic.types
import openai.types.responses


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token usage information for an LLM call."""

    prompt_tokens: int
    generated_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + generated)."""
        return self.prompt_tokens + self.generated_tokens


@dataclasses.dataclass(frozen=True)
class TextData:
    """Text content from an LLM response."""

    content: str


@dataclasses.dataclass(frozen=True)
class ToolUseData:
    """Tool use (function call) from an LLM response."""

    id: str
    name: str
    input: dict[str, Any]


@dataclasses.dataclass(frozen=True, kw_only=True)
class LLMResponse:
    """Response from an LLM call.

    Common params:
        data: List of content blocks (TextData for text, ToolUseData for tool calls)
        cost: Cost of the LLM call in USD.
        usage: Token usage information.
        id: Response ID from the API.
        model: Model name that generated this response.

    Anthropic-only params:
        stop_reason: Reason the response stopped (e.g., "end_turn", "max_tokens").
        stop_sequence: Stop sequence that triggered completion.

    Responses-only params:
        created_at: Unix timestamp when response was created.
        status: Response status (e.g., "completed", "failed").
        parallel_tool_calls: Whether parallel tool calls are enabled.
        response_tool_choice: Tool choice setting used for this response.
        response_tools: List of tools that were available for this response.
    """

    data: list[TextData | ToolUseData]
    cost: float
    usage: Usage
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""

    # Anthropic-only properties
    stop_reason: anthropic.types.StopReason | None = None
    stop_sequence: str | None = None

    # Chat Completions-only properties
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"] | None = None

    # Responses-only properties
    created_at: float | None = None
    status: openai.types.responses.ResponseStatus | None = None
    parallel_tool_calls: bool | None = None
    response_tool_choice: (
        Literal["none", "auto", "required"]
        | openai.types.responses.ToolChoiceAllowed
        | openai.types.responses.ToolChoiceTypes
        | openai.types.responses.ToolChoiceFunction
        | openai.types.responses.ToolChoiceMcp
        | openai.types.responses.ToolChoiceCustom
        | None
    ) = None
    response_tools: list[openai.types.responses.Tool] | None = None


def extract_text_content(response: LLMResponse) -> str:
    """Extract text content from the first block of an LLM response.

    Args:
        response: LLM response to extract text from

    Returns:
        Text content from the first TextData block

    Raises:
        ValueError: If the first block is not TextData or content is empty
    """
    if not response.data:
        raise ValueError("Response has no data blocks")

    first_block = response.data[0]
    if not isinstance(first_block, TextData):
        raise ValueError(f"Expected TextData as first block, got {type(first_block).__name__}")

    if not first_block.content:
        raise ValueError("TextData block has empty content")

    return first_block.content
