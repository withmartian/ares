"""Unified LLM request abstraction supporting multiple API formats."""

import dataclasses
import logging
from typing import Any, Literal, NotRequired, Protocol, Required, TypedDict

_LOGGER = logging.getLogger(__name__)


def _extract_string_content(content: Any, *, strict: bool = True, context: str = "content") -> str:
    """Extract string from content, raising error for unsupported block formats.

    Args:
        content: Content value - should be a string
        strict: If True, raise ValueError for non-string content. If False, log warning and return empty string.
        context: Description of where this content came from (for error messages)

    Returns:
        The content string

    Raises:
        ValueError: If strict=True and content is not a plain string

    Note:
        This function currently does NOT support content blocks (lists of text/image/tool blocks).
        If you encounter a ValueError about list content, this means the API returned structured
        content that needs proper handling - see the issue about extracting text from blocks.
    """
    if isinstance(content, str):
        return content

    if not content:
        return ""

    # Non-string content - this is where we lose information!
    content_type = type(content).__name__
    preview = str(content)[:100] if content else ""

    if isinstance(content, list):
        msg = (
            f"{context} is a list of blocks (structured content), but we only support plain strings. "
            f"This will lose information. Preview: {preview}..."
        )
    else:
        msg = f"{context} has unsupported type {content_type}, expected str. Preview: {preview}..."

    if strict:
        raise ValueError(msg)

    _LOGGER.warning(msg)
    return ""


class UserMessage(TypedDict):
    """User message in a conversation."""

    role: Literal["user"]
    content: str
    name: NotRequired[str]  # Optional - for identifying the user


class AssistantMessage(TypedDict, total=False):
    """Assistant message in a conversation."""

    role: Required[Literal["assistant"]]
    content: str  # Optional - might not have content if tool_calls present
    name: str  # Optional
    tool_calls: list[dict[str, Any]]  # Optional - for tool usage


class ToolCallMessage(TypedDict):
    """Tool call message (tool invocation by assistant).

    Note: In Chat Completions, these are embedded in AssistantMessage.tool_calls.
    In Responses/Messages APIs, these are separate items.
    """

    call_id: str
    name: str
    arguments: str


class ToolCallResponseMessage(TypedDict):
    """Tool call response message (tool result)."""

    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: NotRequired[str]  # Optional


# Union type for all supported message types
Message = UserMessage | AssistantMessage | ToolCallMessage | ToolCallResponseMessage

# Valid roles (excludes system/developer which go in system_prompt)
_VALID_ROLES = frozenset(["user", "assistant", "tool"])


class JSONSchema(TypedDict):
    """JSON Schema definition for tool parameters/input."""

    type: Literal["object"]
    properties: dict[str, Any]
    required: NotRequired[list[str]]  # Optional - list of required property names
    # Additional schema fields (additionalProperties, etc.) can be passed via properties


class Tool(TypedDict):
    """Unified tool definition.

    Uses Claude's simpler format internally (flat structure with input_schema).
    Converts to/from OpenAI's nested format (type: "function" with function.parameters).
    """

    name: str
    description: str
    input_schema: JSONSchema


class ToolChoiceTool(TypedDict):
    """Tool choice: model must use a specific named tool."""

    type: Literal["tool"]
    name: str


# Internal tool choice format
# - "auto": Model decides whether to use tools
# - "any": Model must use at least one tool (maps to OpenAI "required")
# - "none": Model must not use any tools
# - ToolChoiceTool: Model must use a specific named tool
ToolChoice = Literal["auto", "any", "none"] | ToolChoiceTool


@dataclasses.dataclass(frozen=True, kw_only=True)
class LLMRequest:
    """Unified request format for OpenAI Chat Completions, OpenAI Responses, and Claude Messages APIs.

    This class provides a common abstraction over three major LLM API formats, making it easy
    to convert between them. It includes the 9 core parameters that exist across all APIs.

    Core Parameters (all APIs):
        messages: List of user/assistant/tool messages (system messages go in system_prompt)
        max_output_tokens: Maximum tokens to generate (field name varies by API)
        temperature: Sampling temperature (range 0-2 for OpenAI, auto-converted to 0-1 for Claude)
        top_p: Nucleus sampling parameter
        stream: Enable streaming responses
        tools: Tool definitions (schema varies by API, will be converted)
        tool_choice: Tool selection strategy (options vary by API, will be converted)
        metadata: Custom key-value pairs (location varies by API)

    Extended Parameters (partial support):
        service_tier: Processing tier (options differ by API)
        stop_sequences: Stop sequences (not supported in OpenAI Responses)
        system_prompt: System instructions (single source of truth, not in messages list)
        top_k: Top-K sampling (Claude only)

    Note:
        - Model is NOT stored here - it should be managed by the LLMClient
        - Messages only include user/assistant/tool roles (system/developer go in system_prompt)
        - Temperature is stored in OpenAI range (0-2), converted to Claude range (0-1) on export
        - Tool schemas are converted as needed for each API
        - Some parameters may be lost or unsupported when converting between APIs
    """

    messages: list[Message]
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[Tool] | None = None
    tool_choice: ToolChoice | None = None
    metadata: dict[str, Any] | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority", "standard_only"] | None = None
    stop_sequences: list[str] | None = None
    system_prompt: str | None = None
    top_k: int | None = None


class RequestConverter[RequestType](Protocol):
    """Converts between ARES LLMRequest and external API formats.

    This protocol defines the interface for bidirectional conversion between ARES's internal
    LLMRequest format and external API request formats (OpenAI Chat Completions, OpenAI Responses,
    Anthropic Messages, etc.).

    Type Parameters:
        RequestType: The external API's request parameters type (e.g., dict[str, Any] for kwargs)

    Note:
        Implementations are provided as modules with module-level to_external() and from_external()
        functions. The module itself conforms to this Protocol through structural subtyping.
        The model parameter is NOT included in conversions - it should be managed by the LLMClient.

        Available converters:
        - openai_chat_converter: OpenAI Chat Completions API
        - openai_responses_converter: OpenAI Responses API
        - anthropic_converter: Anthropic Messages API
    """

    def to_external(self, request: LLMRequest, *, strict: bool = True) -> RequestType:
        """Convert ARES LLMRequest to external API format.

        Args:
            request: ARES internal request format
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Request parameters in external API format (without model parameter)

        Raises:
            ValueError: If strict=True and information would be lost in conversion
        """
        ...

    def from_external(self, request: RequestType, *, strict: bool = True) -> LLMRequest:
        """Convert external API format to ARES LLMRequest.

        Args:
            request: External API request parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters
        """
        ...
