"""LLM response model wrapping linguafranca types."""

import dataclasses
import time
import uuid

from linguafranca import types as lft


@dataclasses.dataclass(frozen=True)
class InferenceResult:
    """Result from an LLM inference call.

    Attributes:
        response: The linguafranca Open Responses response.
        cost: Cost of the inference call in USD.
    """

    response: lft.OpenResponsesResponse
    cost: float

    @property
    def usage(self) -> lft.Usage | None:
        """Token usage information."""
        return self.response.usage


_TEXT_CONTENT_TYPES = (
    lft.ContentPartInputText,
    lft.ContentPartOutputText,
    lft.ContentPartText,
    lft.ContentPartSummaryText,
    lft.ContentPartReasoningText,
)


def extract_text_content(response: lft.OpenResponsesResponse) -> str:
    """Extract text content from an Open Responses response.

    Returns the concatenated text from all output message content parts.
    Returns empty string if no text content is found.
    """
    text_parts: list[str] = []
    for item in response.output:
        if not isinstance(item, lft.OutputItemMessage):
            continue
        for part in item.content:
            if isinstance(part, _TEXT_CONTENT_TYPES):
                text_parts.append(part.text)
    return "".join(text_parts)


def make_response(
    content: str,
    *,
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    response_id: str | None = None,
) -> lft.OpenResponsesResponse:
    """Create an Open Responses response with the given text content.

    This is a convenience function for constructing lft.OpenResponsesResponse
    objects from simple text completions.

    Args:
        content: The text content of the response.
        model: The model that generated the response.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        response_id: Optional response ID. Generates a UUID if not provided.

    Returns:
        A fully-formed lft.OpenResponsesResponse.
    """
    output: list[lft.OutputItem] = [
        lft.OutputItemMessage(
            content=[lft.ContentPartOutputText(text=content, type="output_text", annotations=None, logprobs=None)],
            id=f"msg_{uuid.uuid4().hex}",
            role=lft.MessageRole.assistant,
            status=lft.ItemStatus.completed,
            type="message",
        )
    ]
    return lft.OpenResponsesResponse(
        created_at=int(time.time()),
        id=response_id or f"resp_{uuid.uuid4().hex}",
        model=model,
        object="response",
        output=output,
        status=lft.ResponseStatus.completed,
        usage=lft.Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=None,
            output_tokens_details=None,
        ),
    )
