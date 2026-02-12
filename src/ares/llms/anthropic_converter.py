"""Converter for Anthropic Messages API format.

This module provides bidirectional conversion between ARES's internal LLMRequest format
and the Anthropic Messages API format. The module itself conforms to the
RequestConverter Protocol through its to_external and from_external functions.

Conversion Notes:
    - temperature converted between OpenAI range (0-2) and Claude range (0-1)
    - messages must alternate user/assistant (enforced by Claude API)
    - system_prompt mapped to/from system parameter
    - service_tier options limited to "auto" and "standard_only"
    - top_k is Claude-specific (supported)
"""

import logging
from typing import Any, cast

import anthropic.types

from ares.llms import request as llm_request

_LOGGER = logging.getLogger(__name__)


def _tool_to_anthropic(tool: llm_request.Tool) -> anthropic.types.ToolParam:
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
) -> llm_request.Tool:
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

    return llm_request.Tool(
        name=anthropic_tool["name"],
        description=anthropic_tool.get("description", ""),
        input_schema=cast(llm_request.JSONSchema, input_schema),
    )


def _tool_choice_to_anthropic(tool_choice: llm_request.ToolChoice | None) -> dict[str, Any] | None:
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
) -> llm_request.ToolChoice | None:
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
            return llm_request.ToolChoiceTool(type="tool", name=tool_choice["name"])

    return None


def _messages_to_claude_format(messages: list[llm_request.Message], *, strict: bool = True) -> list[dict[str, Any]]:
    """Convert messages from Chat format to Claude alternating format.

    Args:
        messages: List of messages in internal format
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

    for msg in messages:
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

        # Keep only role and content - Claude API only accepts these fields
        claude_messages.append(
            {
                "role": role,
                "content": msg_dict.get("content", ""),
            }
        )
        last_role = role

    if dropped_count > 0 and not strict:
        _LOGGER.warning("Dropped %d non-alternating messages for Claude API compliance", dropped_count)

    return claude_messages


def to_external(request: llm_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert ARES LLMRequest to Claude Messages format.

    Args:
        request: ARES internal request format
        strict: If True, raise ValueError on information loss. If False, log warnings.

    Returns:
        Dictionary of kwargs for anthropic.messages.create() (without model)

    Raises:
        ValueError: If strict=True and information would be lost in conversion

    Note:
        Model parameter is NOT included - it should be added by the LLMClient
    """
    # Check for information loss
    lost_info = []
    if request.service_tier not in (None, "auto", "standard_only"):
        lost_info.append(f"service_tier='{request.service_tier}' (Claude only supports 'auto' and 'standard_only')")

    # Check for filtered messages
    filtered_messages = []
    for msg in request.messages:
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
        "messages": _messages_to_claude_format(request.messages, strict=strict),
        "max_tokens": request.max_output_tokens or 1024,  # max_tokens is required by Claude
    }

    if request.system_prompt:
        kwargs["system"] = request.system_prompt

    if request.temperature is not None:
        # Convert from OpenAI range (0-2) to Claude range (0-1)
        kwargs["temperature"] = min(request.temperature / 2.0, 1.0)
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k
    if request.stream:
        kwargs["stream"] = True
    if request.tools:
        # Convert tools to Anthropic format (adds explicit type: "custom")
        kwargs["tools"] = [_tool_to_anthropic(tool) for tool in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = _tool_choice_to_anthropic(request.tool_choice)
    if request.metadata:
        # Claude uses metadata.user_id specifically
        kwargs["metadata"] = request.metadata
    if request.service_tier in ("auto", "standard_only"):
        kwargs["service_tier"] = request.service_tier
    if request.stop_sequences:
        kwargs["stop_sequences"] = request.stop_sequences

    return kwargs


def from_external(
    kwargs: anthropic.types.MessageCreateParams,
    *,
    strict: bool = True,
) -> llm_request.LLMRequest:
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
        system_prompt = llm_request._extract_string_content(system_param, strict=strict, context="System prompt")

    # Filter and validate messages
    filtered_messages: list[llm_request.Message] = []
    for msg in kwargs["messages"]:
        role = msg.get("role")

        # Validate role is supported
        if role not in llm_request._VALID_ROLES:
            if strict:
                raise ValueError(f"Unsupported message role: {role}. Must be one of {llm_request._VALID_ROLES}")
            _LOGGER.warning("Skipping message with unsupported role: %s", role)
            continue

        # Convert to our Message format, validating content
        message_dict = dict(msg)
        if "content" in message_dict:
            message_dict["content"] = llm_request._extract_string_content(
                message_dict["content"], strict=strict, context=f"Message content (role={role})"
            )
        filtered_messages.append(cast(llm_request.Message, message_dict))

    # Convert tools from Anthropic format to internal format
    tools_param = kwargs.get("tools")
    converted_tools: list[llm_request.Tool] | None = None
    if tools_param:
        converted_tools = []
        for tool in tools_param:
            try:
                converted_tools.append(_tool_from_anthropic(tool))
            except ValueError as e:
                if strict:
                    raise
                _LOGGER.warning("Skipping invalid tool: %s", e)

    return llm_request.LLMRequest(
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
