"""Unified LLM request abstraction supporting multiple API formats."""

import dataclasses
import logging
from typing import Any, Literal, NotRequired, Required, TypedDict, cast

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
        # Check for information loss
        lost_info = []
        if self.top_k is not None:
            lost_info.append(f"top_k={self.top_k} (Claude-specific, not supported)")
        if self.service_tier == "standard_only":
            lost_info.append("service_tier='standard_only' (not supported by Chat API)")
        if self.stop_sequences and len(self.stop_sequences) > 4:
            lost_info.append(
                f"stop_sequences truncated from {len(self.stop_sequences)} to 4 "
                f"(Chat API limit: {self.stop_sequences[4:]} will be dropped)"
            )

        if lost_info:
            msg = f"Converting to Chat Completions will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Convert messages, flattening ToolCallMessage into AssistantMessage.tool_calls
        chat_messages: list[dict[str, Any]] = []
        pending_tool_calls: list[dict[str, Any]] = []

        for msg in self.messages:
            msg_dict = dict(msg)

            # ToolCallMessage → collect for previous assistant message
            if "call_id" in msg_dict and "name" in msg_dict and "arguments" in msg_dict:
                # This is a ToolCallMessage
                pending_tool_calls.append(
                    {
                        "id": msg_dict["call_id"],
                        "type": "function",
                        "function": {
                            "name": msg_dict["name"],
                            "arguments": msg_dict["arguments"],
                        },
                    }
                )
            else:
                # Flush any pending tool calls to the last assistant message
                if pending_tool_calls and chat_messages:
                    last_msg = chat_messages[-1]
                    if last_msg.get("role") == "assistant":
                        last_msg["tool_calls"] = pending_tool_calls
                        pending_tool_calls = []
                    else:
                        if strict:
                            role = last_msg.get("role")
                            raise ValueError(
                                f"ToolCallMessage found but previous message is not assistant (role={role})"
                            )
                        _LOGGER.warning(
                            "ToolCallMessage found but previous message is not assistant, discarding tool calls"
                        )
                        pending_tool_calls = []

                # Add the current message
                chat_messages.append(msg_dict)

        # Flush any remaining tool calls
        if pending_tool_calls and chat_messages:
            last_msg = chat_messages[-1]
            if last_msg.get("role") == "assistant":
                last_msg["tool_calls"] = pending_tool_calls
            elif strict:
                raise ValueError("ToolCallMessage at end but last message is not assistant")

        kwargs: dict[str, Any] = {
            "messages": chat_messages,
        }

        # Add system prompt as first message if present
        if self.system_prompt:
            kwargs["messages"] = [
                {"role": "system", "content": self.system_prompt},
                *kwargs["messages"],
            ]

        # Add optional parameters (filter None values)
        if self.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            kwargs["tools"] = [_tool_to_chat_completions(tool) for tool in self.tools]
        if self.tool_choice is not None:
            kwargs["tool_choice"] = _tool_choice_to_openai(self.tool_choice)
        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.service_tier and self.service_tier != "standard_only":
            kwargs["service_tier"] = self.service_tier
        if self.stop_sequences:
            # OpenAI Chat supports up to 4 stop sequences
            kwargs["stop"] = self.stop_sequences[:4]

        return kwargs

    def to_responses_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to OpenAI Responses API format.

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
        # Check for information loss
        lost_info = []
        if self.stop_sequences:
            lost_info.append(f"stop_sequences={self.stop_sequences} (not supported by Responses API)")
        if self.top_k is not None:
            lost_info.append(f"top_k={self.top_k} (Claude-specific, not supported)")
        if self.service_tier == "standard_only":
            lost_info.append("service_tier='standard_only' (not supported by Responses API)")

        if lost_info:
            msg = f"Converting to Responses will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        kwargs: dict[str, Any] = {
            "input": self._messages_to_responses_input(),
        }

        if self.system_prompt:
            kwargs["instructions"] = self.system_prompt

        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            kwargs["tools"] = [_tool_to_responses(tool) for tool in self.tools]
        if self.tool_choice is not None:
            kwargs["tool_choice"] = _tool_choice_to_responses(self.tool_choice)
        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.service_tier and self.service_tier != "standard_only":
            kwargs["service_tier"] = self.service_tier

        return kwargs

    def to_messages_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to Claude Messages API format.

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
        # Check for information loss
        lost_info = []
        if self.service_tier not in (None, "auto", "standard_only"):
            lost_info.append(f"service_tier='{self.service_tier}' (Claude only supports 'auto' and 'standard_only')")

        # Check for filtered messages
        filtered_messages = []
        for msg in self.messages:
            msg_dict = dict(msg)
            role = msg_dict["role"]
            if role in ("system", "developer"):
                content = str(msg_dict.get("content", ""))[:50]
                filtered_messages.append(f"{role} message: {content}...")

        if filtered_messages:
            lost_info.append(f"Messages filtered out (use system_prompt instead): {'; '.join(filtered_messages)}")

        if lost_info:
            msg = f"Converting to Claude Messages will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        kwargs: dict[str, Any] = {
            "messages": self._messages_to_claude_format(strict=strict),
            "max_tokens": self.max_output_tokens or 1024,  # max_tokens is required by Claude
        }

        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        if self.temperature is not None:
            # Convert from OpenAI range (0-2) to Claude range (0-1)
            kwargs["temperature"] = min(self.temperature / 2.0, 1.0)
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            # Convert tools to Anthropic format (adds explicit type: "custom")
            kwargs["tools"] = [_tool_to_anthropic(tool) for tool in self.tools]
        if self.tool_choice is not None:
            kwargs["tool_choice"] = _tool_choice_to_anthropic(self.tool_choice)
        if self.metadata:
            # Claude uses metadata.user_id specifically
            kwargs["metadata"] = self.metadata
        if self.service_tier in ("auto", "standard_only"):
            kwargs["service_tier"] = self.service_tier
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        return kwargs

    def _messages_to_responses_input(self) -> list[dict[str, Any]]:
        """Convert messages from internal format to Responses input items.

        Returns:
            List of input items for Responses API

        Note:
            - ToolCallMessage → function_call items
            - ToolCallResponseMessage → function_call_output items
            - Other messages → message items
        """
        input_items = []
        for msg in self.messages:
            msg_dict = dict(msg)  # Convert to regular dict for type safety

            # ToolCallMessage (tool invocation) → function_call
            if "call_id" in msg_dict and "name" in msg_dict and "arguments" in msg_dict:
                item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": msg_dict["call_id"],
                    "name": msg_dict["name"],
                    "arguments": msg_dict["arguments"],
                }
                input_items.append(item)

            # ToolCallResponseMessage (tool result) → function_call_output
            elif msg_dict.get("role") == "tool":
                item = {
                    "type": "function_call_output",
                    "output": msg_dict.get("content", ""),
                }
                # Include call_id if present (required for routing)
                if "tool_call_id" in msg_dict:
                    item["call_id"] = msg_dict["tool_call_id"]

                input_items.append(item)

            # Regular messages → message items
            else:
                role = msg_dict.get("role")
                item = {
                    "type": "message",
                    "role": role,
                    "content": msg_dict.get("content", ""),
                }

                # Include optional name field if present
                if "name" in msg_dict:
                    item["name"] = msg_dict["name"]

                input_items.append(item)

        return input_items

    def _messages_to_claude_format(self, *, strict: bool = True) -> list[dict[str, Any]]:
        """Convert messages from Chat format to Claude alternating format.

        Args:
            strict: If True, raise ValueError on non-alternating messages. If False, drop
                   consecutive messages with the same role (keeping first) with a warning.

        Returns:
            List of messages in Claude format (user/assistant alternating)

        Raises:
            ValueError: If strict=True and messages don't alternate roles

        Note:
            Claude requires strict alternation. This method filters out system/developer
            messages (should be in system_prompt) and ensures alternation.
        """
        claude_messages = []
        last_role = None
        dropped_count = 0

        for msg in self.messages:
            msg_dict = dict(msg)  # Convert to regular dict for type safety
            role = msg_dict["role"]
            # Skip system/developer messages (should be in system_prompt)
            if role in ("system", "developer"):
                continue
            # Map tool/function to user role (tool results)
            if role in ("tool", "function"):
                role = "user"

            # Check for alternation
            if last_role == role:
                content = str(msg_dict.get("content", ""))[:50]
                if strict:
                    raise ValueError(
                        f"Messages must alternate between user and assistant roles for Claude API. "
                        f"Found consecutive '{role}' messages. Message content: {content}..."
                    )
                else:
                    _LOGGER.warning(
                        "Dropping non-alternating message with role '%s' (content: %s...). "
                        "Claude requires strict alternation between user and assistant.",
                        role,
                        content,
                    )
                    dropped_count += 1
                    continue

            # Keep user/assistant as-is
            claude_messages.append(
                {
                    "role": role,
                    "content": msg_dict.get("content", ""),
                }
            )
            last_role = role

        if dropped_count > 0 and not strict:
            _LOGGER.warning("Dropped %d non-alternating message(s) when converting to Claude format", dropped_count)

        return claude_messages

    @classmethod
    def from_chat_completion(
        cls,
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Chat Completions API kwargs.

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
        # Define parameters we actually store/handle
        handled_params = {
            "model",  # Handled by being ignored (LLMClient manages this)
            "messages",
            "max_completion_tokens",
            "max_tokens",  # Fallback for max_output_tokens
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "stop",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Chat Completions parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Extract system prompt and filter messages
        system_prompt = None
        filtered_messages: list[Message] = []

        for msg in kwargs["messages"]:
            role = msg.get("role")

            # Extract system/developer messages as system_prompt (use first one)
            if role in ("system", "developer"):
                if system_prompt is None:
                    content = msg.get("content", "")
                    system_prompt = _extract_string_content(
                        content, strict=strict, context=f"System/developer message content (role={role})"
                    )
                continue

            # Validate role is supported
            if role not in _VALID_ROLES:
                if strict:
                    raise ValueError(f"Unsupported message role: {role}. Must be one of {_VALID_ROLES}")
                _LOGGER.warning("Skipping message with unsupported role: %s", role)
                continue

            # Extract tool_calls from assistant messages
            if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                # Add assistant message without tool_calls (or with content only)
                assistant_msg = dict(msg)
                # Remove tool_calls from the message we store
                tool_calls_list = assistant_msg.pop("tool_calls", [])

                # Validate content if present
                if "content" in assistant_msg:
                    assistant_msg["content"] = _extract_string_content(
                        assistant_msg["content"], strict=strict, context="Assistant message content"
                    )

                filtered_messages.append(cast(Message, assistant_msg))

                # Create separate ToolCallMessage for each tool call
                if isinstance(tool_calls_list, list):
                    for tool_call in tool_calls_list:
                        if tool_call.get("type") == "function":
                            function = tool_call.get("function", {})
                            filtered_messages.append(
                                cast(
                                    Message,
                                    {
                                        "call_id": tool_call.get("id", ""),
                                        "name": function.get("name", ""),
                                        "arguments": function.get("arguments", ""),
                                    },
                                )
                            )
            else:
                # Convert to our Message format, validating content
                message_dict = dict(msg)
                if "content" in message_dict:
                    message_dict["content"] = _extract_string_content(
                        message_dict["content"], strict=strict, context=f"Message content (role={role})"
                    )
                filtered_messages.append(cast(Message, message_dict))

        # Convert tools from OpenAI to Claude format
        tools_param = kwargs.get("tools")
        converted_tools: list[Tool] | None = None
        if tools_param:
            converted_tools = []
            for tool in tools_param:
                tool_type = tool.get("type")
                if tool_type != "function":
                    if strict:
                        raise ValueError(f"Unsupported tool type: {tool_type}. Only 'function' tools are supported.")
                    _LOGGER.warning("Skipping tool with unsupported type: %s", tool_type)
                    continue
                converted_tools.append(
                    _tool_from_chat_completions(cast(openai.types.chat.ChatCompletionToolParam, tool))
                )

        # Handle stop sequences - convert single string to list
        stop_param = kwargs.get("stop")
        stop_sequences: list[str] | None = None
        if isinstance(stop_param, list):
            stop_sequences = stop_param
        elif isinstance(stop_param, str):
            stop_sequences = [stop_param]

        # Handle system prompt - extract string from various formats
        final_system_prompt: str | None = None
        if system_prompt:
            final_system_prompt = _extract_string_content(system_prompt, strict=strict, context="System prompt")

        return cls(
            messages=filtered_messages,
            max_output_tokens=kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stream=bool(kwargs.get("stream", False)),
            tools=converted_tools,
            tool_choice=_tool_choice_from_openai(cast(str | dict[str, Any] | None, kwargs.get("tool_choice"))),
            metadata=cast(dict[str, Any] | None, kwargs.get("metadata")),
            service_tier=kwargs.get("service_tier"),
            stop_sequences=stop_sequences,
            system_prompt=final_system_prompt,
        )

    @classmethod
    def from_responses(
        cls,
        kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Responses API kwargs.

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
        # Define parameters we actually store/handle
        handled_params = {
            "model",  # Handled by being ignored (LLMClient manages this)
            "input",
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "instructions",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Responses parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Convert input items to messages
        input_param = kwargs.get("input", [])
        filtered_messages: list[Message] = []

        if isinstance(input_param, str):
            filtered_messages = [UserMessage(role="user", content=input_param)]
        elif isinstance(input_param, list):
            for item in input_param:
                item_type = item.get("type")

                # Handle function_call (tool invocations)
                if item_type == "function_call":
                    call_id = item.get("call_id")
                    name = item.get("name")
                    arguments = item.get("arguments", "")

                    if call_id is None or name is None:
                        if strict:
                            raise ValueError(
                                f"Tool call (function_call) missing required 'call_id' or 'name' field. Item: {item}"
                            )
                        _LOGGER.warning("Tool call (function_call) missing required fields, skipping. Item: %s", item)
                        continue

                    # Create ToolCallMessage
                    filtered_messages.append(
                        cast(
                            Message,
                            {
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments if isinstance(arguments, str) else str(arguments),
                            },
                        )
                    )

                # Handle function_call_output (tool results)
                elif item_type == "function_call_output":
                    call_id = item.get("call_id")
                    output = item.get("output", "")
                    output_str = output if isinstance(output, str) else str(output)

                    if call_id is None:
                        if strict:
                            raise ValueError(
                                "Tool result (function_call_output) missing required 'call_id' field for routing. "
                                f"Output: {output_str[:50]}..."
                            )
                        _LOGGER.warning(
                            "Tool result (function_call_output) missing 'call_id' field (output: %s...). "
                            "This may cause routing issues.",
                            output_str[:50],
                        )
                        # Create tool message without call_id
                        filtered_messages.append(cast(Message, {"role": "tool", "content": output_str}))
                    else:
                        # Create tool message with call_id
                        filtered_messages.append(
                            cast(Message, {"role": "tool", "content": output_str, "tool_call_id": call_id})
                        )

                # Handle regular messages
                elif item_type == "message":
                    role = item.get("role")

                    # Validate role is supported
                    if role not in _VALID_ROLES:
                        if strict:
                            raise ValueError(f"Unsupported message role: {role}. Must be one of {_VALID_ROLES}")
                        _LOGGER.warning("Skipping message with unsupported role: %s", role)
                        continue

                    # Extract content - use helper to detect unsupported block formats
                    content_param = item.get("content", "")
                    content_str = _extract_string_content(
                        content_param, strict=strict, context=f"Message content (role={role})"
                    )

                    # Build message dict with required fields
                    message_dict: dict[str, Any] = {"role": role, "content": content_str}

                    # Include optional name field if present
                    if "name" in item:
                        message_dict["name"] = item["name"]

                    # Cast to Message after validating role and building dict
                    filtered_messages.append(cast(Message, message_dict))

        # Convert tools from Responses format to Claude format
        tools_param = kwargs.get("tools")
        converted_tools: list[Tool] | None = None
        if tools_param:
            temp_tools: list[Tool] = []
            for tool in tools_param:
                try:
                    temp_tools.append(_tool_from_responses(tool))
                except ValueError as e:
                    if strict:
                        raise
                    _LOGGER.warning("Skipping tool that cannot be converted: %s", e)
            # Only set converted_tools if we successfully converted at least one tool
            if temp_tools:
                converted_tools = temp_tools

        return cls(
            messages=filtered_messages,
            max_output_tokens=kwargs.get("max_output_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stream=bool(kwargs.get("stream", False)),
            tools=converted_tools,
            tool_choice=_tool_choice_from_openai(cast(str | dict[str, Any] | None, kwargs.get("tool_choice"))),
            metadata=cast(dict[str, Any] | None, kwargs.get("metadata")),
            service_tier=kwargs.get("service_tier"),
            system_prompt=kwargs.get("instructions"),
        )

    @classmethod
    def from_messages(
        cls,
        kwargs: anthropic.types.MessageCreateParams,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from Claude Messages API kwargs.

        Args:
            kwargs: Claude Messages API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters
        """
        # Define parameters we handle (model is accepted but not stored)
        handled_params = {
            "model",  # Accepted but not stored - managed by LLMClient
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "stop_sequences",
            "system",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Claude Messages parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Convert temperature from Claude range (0-1) to OpenAI range (0-2)
        temperature = kwargs.get("temperature")
        if temperature is not None:
            temperature = temperature * 2.0

        # Extract system prompt (can be str or list of text blocks)
        system_param = kwargs.get("system")
        system_prompt = None
        if system_param:
            system_prompt = _extract_string_content(system_param, strict=strict, context="System prompt")

        # Filter and validate messages
        filtered_messages: list[Message] = []
        for msg in kwargs["messages"]:
            role = msg.get("role")

            # Validate role is supported
            if role not in _VALID_ROLES:
                if strict:
                    raise ValueError(f"Unsupported message role: {role}. Must be one of {_VALID_ROLES}")
                _LOGGER.warning("Skipping message with unsupported role: %s", role)
                continue

            # Convert to our Message format, validating content
            message_dict = dict(msg)
            if "content" in message_dict:
                message_dict["content"] = _extract_string_content(
                    message_dict["content"], strict=strict, context=f"Message content (role={role})"
                )
            filtered_messages.append(cast(Message, message_dict))

        # Convert tools from Anthropic format to internal format
        tools_param = kwargs.get("tools")
        converted_tools: list[Tool] | None = None
        if tools_param:
            converted_tools = []
            for tool in tools_param:
                try:
                    converted_tools.append(_tool_from_anthropic(tool))
                except ValueError as e:
                    if strict:
                        raise
                    _LOGGER.warning("Skipping invalid tool: %s", e)

        return cls(
            messages=filtered_messages,
            max_output_tokens=kwargs["max_tokens"],
            temperature=temperature,
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            stream=bool(kwargs.get("stream", False)),
            tools=converted_tools,
            tool_choice=_tool_choice_from_anthropic(cast(dict[str, Any] | None, kwargs.get("tool_choice"))),
            metadata=cast(dict[str, Any] | None, kwargs.get("metadata")),
            service_tier=kwargs.get("service_tier"),
            stop_sequences=cast(list[str] | None, kwargs.get("stop_sequences")),
            system_prompt=system_prompt,
        )
