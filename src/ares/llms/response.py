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
