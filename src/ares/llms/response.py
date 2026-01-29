"""LLM response model."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token usage information for an LLM call."""

    prompt_tokens: int
    generation_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + generation)."""
        return self.prompt_tokens + self.generation_tokens


@dataclasses.dataclass(frozen=True)
class TextData:
    """Text content from an LLM response."""

    content: str


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    """Response from an LLM call.

    Attributes:
        data: List of content blocks (currently only TextData, but extensible to ImageData, etc.)
        cost: Cost of the LLM call in USD.
        usage: Token usage information.
    """

    data: list[TextData]
    cost: float
    usage: Usage
