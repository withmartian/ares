"""LLM response model."""

import dataclasses
from typing import Any


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


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call.

    Attributes:
        data: List of content blocks (TextData for text, ToolUseData for tool calls)
        cost: Cost of the LLM call in USD.
        usage: Token usage information.
    """

    data: list[TextData | ToolUseData]
    cost: float
    usage: Usage


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
