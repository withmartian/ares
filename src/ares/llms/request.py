"""Unified LLM request abstraction supporting multiple API formats."""

import dataclasses
import logging
from typing import Any, Literal, NotRequired, Protocol, Required, TypedDict, cast

import anthropic.types
import openai.types.chat
import openai.types.chat.completion_create_params
import openai.types.responses
import openai.types.responses.response_create_params
import openai.types.shared_params

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


def _tool_to_chat_completions(tool: Tool) -> openai.types.chat.ChatCompletionToolParam:
    """Convert Tool from ARES internal format to OpenAI Chat Completions format.

    Args:
        tool: Tool in ARES internal format (flat with input_schema)

    Returns:
        Tool in OpenAI Chat Completions format (nested with type and function.parameters)
    """
    return openai.types.chat.ChatCompletionToolParam(
        type="function",
        function=openai.types.shared_params.FunctionDefinition(
            name=tool["name"],
            description=tool["description"],
            parameters=cast(dict[str, object], tool["input_schema"]),
        ),
    )


def _tool_from_chat_completions(chat_completions_tool: openai.types.chat.ChatCompletionToolParam) -> Tool:
    """Convert tool from OpenAI Chat Completions format to ARES internal format.

    Args:
        chat_completions_tool: Tool in OpenAI Chat Completions format (nested with type and function.parameters)

    Returns:
        Tool in ARES internal format (flat with input_schema)
    """
    function = chat_completions_tool["function"]
    parameters = function.get("parameters", {"type": "object", "properties": {}})

    # Validate that parameters is a valid JSONSchema
    if not isinstance(parameters, dict):
        raise ValueError(f"Tool parameters must be a dict, got {type(parameters)}")
    if "type" not in parameters:
        raise ValueError("Tool parameters must have a 'type' field")

    return Tool(
        name=function["name"],
        description=function.get("description", ""),
        input_schema=cast(JSONSchema, parameters),
    )


def _tool_to_responses(tool: Tool) -> openai.types.responses.FunctionToolParam:
    """Convert Tool from ARES internal format to OpenAI Responses format.

    Args:
        tool: Tool in ARES internal format (flat with input_schema)

    Returns:
        Tool in OpenAI Responses format (flat with type, name, description, parameters)
    """
    return openai.types.responses.FunctionToolParam(
        type="function",
        name=tool["name"],
        description=tool["description"],
        parameters=cast(dict[str, object], tool["input_schema"]),
        strict=True,
    )


def _tool_from_responses(responses_tool: openai.types.responses.ToolParam) -> Tool:
    """Convert tool from OpenAI Responses format to ARES internal format.

    Args:
        responses_tool: Tool in OpenAI Responses format (flat with type, name, parameters)

    Returns:
        Tool in ARES internal format (flat with input_schema)

    Note:
        Currently only supports FunctionToolParam. Other tool types are not converted.
    """
    # Only handle FunctionToolParam for now
    if responses_tool.get("type") == "function":
        # Type guard: if type is "function", this is FunctionToolParam
        func_tool = cast(openai.types.responses.FunctionToolParam, responses_tool)
        parameters = func_tool.get("parameters") or {"type": "object", "properties": {}}

        # Validate that parameters is a valid JSONSchema
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool parameters must be a dict, got {type(parameters)}")
        if "type" not in parameters:
            raise ValueError("Tool parameters must have a 'type' field")

        return Tool(
            name=func_tool["name"],
            description=func_tool.get("description") or "",
            input_schema=cast(JSONSchema, parameters),
        )
    # For other tool types, we can't convert them to Claude format
    raise ValueError(f"Unsupported tool type for conversion: {responses_tool.get('type')}")


def _tool_to_anthropic(tool: Tool) -> anthropic.types.ToolParam:
    """Convert Tool from ARES internal format to Anthropic Messages format.

    Args:
        tool: Tool in ARES internal format (flat with input_schema)

    Returns:
        Tool in Anthropic Messages format (custom tool with type, name, description, input_schema)
    """
    return anthropic.types.ToolParam(
        type="custom",
        name=tool["name"],
        description=tool["description"],
        input_schema=cast(dict[str, object], tool["input_schema"]),
    )


def _tool_from_anthropic(
    anthropic_tool: anthropic.types.ToolUnionParam,
) -> Tool:
    """Convert tool from Anthropic Messages format to ARES internal format.

    Args:
        anthropic_tool: Tool in Anthropic format (ToolParam with type='custom'/None, or built-in tool types)

    Returns:
        Tool in ARES internal format

    Raises:
        ValueError: If tool type is unsupported or required fields are missing

    Note:
        Only supports ToolParam with type='custom' or type=None. Built-in tool types
        (bash_20250124, text_editor_*, web_search_*) are not supported.
    """
    # Check tool type - we only accept "custom" (or None which defaults to custom)
    # Reject built-in tool types like bash_20250124, text_editor_*, web_search_*
    tool_type = anthropic_tool.get("type")
    if tool_type is not None and tool_type != "custom":
        raise ValueError(
            f"Unsupported tool type: {tool_type}. Only 'custom' tools are supported. "
            f"Built-in tools (bash, text_editor, web_search) are not supported."
        )

    # Validate required fields
    if "name" not in anthropic_tool:
        raise ValueError("Tool missing required 'name' field")

    if "input_schema" not in anthropic_tool:
        raise ValueError(f"Tool '{anthropic_tool.get('name')}' missing required 'input_schema' field")

    # Validate input_schema structure
    input_schema = anthropic_tool["input_schema"]
    if not isinstance(input_schema, dict):
        raise ValueError(f"Tool '{anthropic_tool['name']}' input_schema must be a dict, got {type(input_schema)}")

    if "type" not in input_schema:
        raise ValueError(f"Tool '{anthropic_tool['name']}' input_schema must have a 'type' field")

    return Tool(
        name=anthropic_tool["name"],
        description=anthropic_tool.get("description", ""),
        input_schema=cast(JSONSchema, input_schema),
    )


def _tool_choice_to_openai(tool_choice: ToolChoice | None) -> str | dict[str, Any] | None:
    """Convert ARES internal ToolChoice to OpenAI Chat Completions format.

    Args:
        tool_choice: ARES internal tool choice

    Returns:
        Tool choice in OpenAI format:
        - "auto": Model decides
        - "required": Must use at least one tool
        - "none": Must not use any tools
        - {"type": "function", "function": {"name": "..."}}: Specific function
    """
    if tool_choice is None:
        return None

    if tool_choice == "auto":
        return "auto"
    elif tool_choice == "any":
        return "required"  # Map "any" to OpenAI's "required"
    elif tool_choice == "none":
        return "none"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }

    return None


def _tool_choice_to_responses(tool_choice: ToolChoice | None) -> str | dict[str, Any] | None:
    """Convert ARES internal ToolChoice to OpenAI Responses format.

    Args:
        tool_choice: ARES internal tool choice

    Returns:
        Tool choice in OpenAI Responses format:
        - "auto": Model decides
        - "required": Must use at least one tool
        - "none": Must not use any tools
        - {"type": "function", "name": "..."}: Specific function (flat structure)
    """
    if tool_choice is None:
        return None

    if tool_choice == "auto":
        return "auto"
    elif tool_choice == "any":
        return "required"  # Map "any" to OpenAI's "required"
    elif tool_choice == "none":
        return "none"
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        # Responses API uses flat structure: {"type": "function", "name": "..."}
        return {
            "type": "function",
            "name": tool_choice["name"],
        }

    return None


def _tool_choice_from_openai(
    tool_choice: str | dict[str, Any] | None,
) -> ToolChoice | None:
    """Convert OpenAI Chat Completions tool_choice to internal format.

    Args:
        tool_choice: OpenAI tool choice parameter

    Returns:
        Internal ToolChoice format
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        result = {"auto": "auto", "required": "any", "none": "none"}.get(tool_choice)
        if not result:
            raise ValueError(f"Unsupported tool choice: {tool_choice}")
        return cast(Literal["auto", "any", "none"], result)

    elif isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            # {"type": "function", "function": {"name": "x"}} -> {"type": "tool", "name": "x"}
            function_data = tool_choice.get("function", {})
            if isinstance(function_data, dict) and "name" in function_data:
                return ToolChoiceTool(type="tool", name=function_data["name"])

    return None


def _tool_choice_to_anthropic(tool_choice: ToolChoice | None) -> dict[str, Any] | None:
    """Convert internal ToolChoice to Anthropic Messages format.

    Args:
        tool_choice: Internal tool choice

    Returns:
        Tool choice in Anthropic format:
        - {"type": "auto"}: Model decides
        - {"type": "any"}: Must use at least one tool
        - {"type": "none"}: Must not use any tools
        - {"type": "tool", "name": "..."}: Specific tool
    """
    if tool_choice is None:
        return None

    if tool_choice == "auto":
        return {"type": "auto"}
    elif tool_choice == "any":
        return {"type": "any"}
    elif tool_choice == "none":
        return {"type": "none"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return {"type": "tool", "name": tool_choice["name"]}

    return None


def _tool_choice_from_anthropic(
    tool_choice: dict[str, Any] | None,
) -> ToolChoice | None:
    """Convert Anthropic Messages tool_choice to internal format.

    Args:
        tool_choice: Anthropic tool choice parameter

    Returns:
        Internal ToolChoice format
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "any"
        elif choice_type == "none":
            return "none"
        elif choice_type == "tool" and "name" in tool_choice:
            return ToolChoiceTool(type="tool", name=tool_choice["name"])

    return None


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

    def to_external(self, request: "LLMRequest", *, strict: bool = True) -> RequestType:
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

    def from_external(self, kwargs: RequestType, *, strict: bool = True) -> "LLMRequest":
        """Convert external API format to ARES LLMRequest.

        Args:
            kwargs: External API request parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters
        """
        ...


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

    def to_chat_completion_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions API format.

        This is a convenience wrapper around openai_chat_converter module.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for openai.ChatCompletion.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - top_k is not supported (Claude-specific)
            - service_tier="standard_only" is not supported
            - stop_sequences truncated to 4 if more provided
        """
        from ares.llms import openai_chat_converter

        return openai_chat_converter.to_external(self, strict=strict)

    def to_responses_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to OpenAI Responses API format.

        This is a convenience wrapper around openai_responses_converter module.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for openai.Responses.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - messages are converted to input items
            - system_prompt is mapped to instructions parameter
            - stop_sequences are not supported in Responses API
            - top_k is not supported (Claude-specific)
            - service_tier="standard_only" is not supported
        """
        from ares.llms import openai_responses_converter

        return openai_responses_converter.to_external(self, strict=strict)

    def to_messages_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to Claude Messages API format.

        This is a convenience wrapper around anthropic_converter module.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for anthropic.messages.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - temperature is converted from OpenAI range (0-2) to Claude range (0-1)
            - messages must alternate user/assistant (enforced by Claude API)
            - system_prompt is mapped to system parameter
            - service_tier options are limited to "auto" and "standard_only"
            - tool schemas may need conversion (not implemented yet)
        """
        from ares.llms import anthropic_converter

        return anthropic_converter.to_external(self, strict=strict)

    @classmethod
    def from_chat_completion(
        cls,
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Chat Completions API kwargs.

        This is a convenience wrapper around openai_chat_converter module.

        Args:
            kwargs: OpenAI Chat Completions API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters

        Note:
            Model parameter is ignored - it should be managed by the LLMClient
        """
        from ares.llms import openai_chat_converter

        return openai_chat_converter.from_external(kwargs, strict=strict)

    @classmethod
    def from_responses(
        cls,
        kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Responses API kwargs.

        This is a convenience wrapper around openai_responses_converter module.

        Args:
            kwargs: OpenAI Responses API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters

        Note:
            Model parameter is ignored - it should be managed by the LLMClient
        """
        from ares.llms import openai_responses_converter

        return openai_responses_converter.from_external(kwargs, strict=strict)

    @classmethod
    def from_messages(
        cls,
        kwargs: anthropic.types.MessageCreateParams,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from Claude Messages API kwargs.

        This is a convenience wrapper around anthropic_converter module.

        Args:
            kwargs: Claude Messages API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters
        """
        from ares.llms import anthropic_converter

        return anthropic_converter.from_external(kwargs, strict=strict)
