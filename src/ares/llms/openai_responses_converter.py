"""Converter for OpenAI Responses API format.

This module provides bidirectional conversion between ARES's internal LLMRequest format
and the OpenAI Responses API format. The module itself conforms to the
RequestConverter Protocol through its to_external and from_external functions.

Conversion Notes:
    - stop_sequences are not supported in Responses API
    - top_k is not supported (Claude-specific)
    - service_tier="standard_only" is not supported
    - system_prompt mapped to/from instructions parameter
    - messages converted to/from input items
"""

import logging
from typing import Any, cast

import openai.types.responses
import openai.types.responses.response_create_params

from ares.llms import request as llm_request

_LOGGER = logging.getLogger(__name__)


def _tool_to_responses(tool: llm_request.Tool) -> openai.types.responses.FunctionToolParam:
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


def _tool_from_responses(responses_tool: openai.types.responses.ToolParam) -> llm_request.Tool:
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

        return llm_request.Tool(
            name=func_tool["name"],
            description=func_tool.get("description") or "",
            input_schema=cast(llm_request.JSONSchema, parameters),
        )
    # For other tool types, we can't convert them to Claude format
    raise ValueError(f"Unsupported tool type for conversion: {responses_tool.get('type')}")


def _tool_choice_to_responses(tool_choice: llm_request.ToolChoice | None) -> str | dict[str, Any] | None:
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
) -> llm_request.ToolChoice | None:
    """Convert OpenAI Chat Completions tool_choice to internal format.

    Args:
        tool_choice: OpenAI tool choice parameter

    Returns:
        Internal ToolChoice format
    """
    if tool_choice is None:
        return None

    if isinstance(tool_choice, str):
        from typing import Literal

        result = {"auto": "auto", "required": "any", "none": "none"}.get(tool_choice)
        if not result:
            raise ValueError(f"Unsupported tool choice: {tool_choice}")
        return cast(Literal["auto", "any", "none"], result)

    elif isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            # {"type": "function", "function": {"name": "x"}} -> {"type": "tool", "name": "x"}
            # Responses API uses flat format: {"type": "function", "name": "x"}
            if "name" in tool_choice:
                # Flat format (Responses API)
                return llm_request.ToolChoiceTool(type="tool", name=tool_choice["name"])
            else:
                # Nested format (Chat API)
                function_data = tool_choice.get("function", {})
                if isinstance(function_data, dict) and "name" in function_data:
                    return llm_request.ToolChoiceTool(type="tool", name=function_data["name"])

    return None


def _messages_to_responses_input(messages: list[llm_request.Message]) -> list[dict[str, Any]]:
    """Convert messages from internal format to Responses input items.

    Args:
        messages: List of messages in internal format

    Returns:
        List of input items for Responses API

    Note:
        - ToolCallMessage → function_call items
        - ToolCallResponseMessage → function_call_output items
        - Other messages → message items
    """
    input_items = []
    for msg in messages:
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


def to_external(request: llm_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert ARES LLMRequest to OpenAI Responses format.

    Args:
        request: ARES internal request format
        strict: If True, raise ValueError on information loss. If False, log warnings.

    Returns:
        Dictionary of kwargs for openai.Responses.create() (without model)

    Raises:
        ValueError: If strict=True and information would be lost in conversion

    Note:
        Model parameter is NOT included - it should be added by the LLMClient
    """
    # Check for information loss
    lost_info = []
    if request.stop_sequences:
        lost_info.append(f"stop_sequences={request.stop_sequences} (not supported by Responses API)")
    if request.top_k is not None:
        lost_info.append(f"top_k={request.top_k} (Claude-specific, not supported)")
    if request.service_tier == "standard_only":
        lost_info.append("service_tier='standard_only' (not supported by Responses API)")

    if lost_info:
        msg = f"Converting to Responses will lose information: {'; '.join(lost_info)}"
        if strict:
            raise ValueError(msg)
        _LOGGER.warning(msg)

    kwargs: dict[str, Any] = {
        "input": _messages_to_responses_input(request.messages),
    }

    if request.system_prompt:
        kwargs["instructions"] = request.system_prompt

    if request.max_output_tokens is not None:
        kwargs["max_output_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stream:
        kwargs["stream"] = True
    if request.tools:
        kwargs["tools"] = [_tool_to_responses(tool) for tool in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = _tool_choice_to_responses(request.tool_choice)
    if request.metadata:
        kwargs["metadata"] = request.metadata
    if request.service_tier and request.service_tier != "standard_only":
        kwargs["service_tier"] = request.service_tier

    return kwargs


def from_external(
    kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
    *,
    strict: bool = True,
) -> llm_request.LLMRequest:
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
    filtered_messages: list[llm_request.Message] = []

    if isinstance(input_param, str):
        filtered_messages = [llm_request.UserMessage(role="user", content=input_param)]
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
                        llm_request.Message,
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
                    filtered_messages.append(cast(llm_request.Message, {"role": "tool", "content": output_str}))
                else:
                    # Create tool message with call_id
                    filtered_messages.append(
                        cast(llm_request.Message, {"role": "tool", "content": output_str, "tool_call_id": call_id})
                    )

            # Handle regular messages
            elif item_type == "message":
                role = item.get("role")

                # Validate role is supported
                if role not in llm_request._VALID_ROLES:
                    if strict:
                        raise ValueError(f"Unsupported message role: {role}. Must be one of {llm_request._VALID_ROLES}")
                    _LOGGER.warning("Skipping message with unsupported role: %s", role)
                    continue

                # Extract content - use helper to detect unsupported block formats
                content_param = item.get("content", "")
                content_str = llm_request._extract_string_content(
                    content_param, strict=strict, context=f"Message content (role={role})"
                )

                # Build message dict with required fields
                message_dict: dict[str, Any] = {"role": role, "content": content_str}

                # Include optional name field if present
                if "name" in item:
                    message_dict["name"] = item["name"]

                # Cast to Message after validating role and building dict
                filtered_messages.append(cast(llm_request.Message, message_dict))

    # Convert tools from Responses format to Claude format
    tools_param = kwargs.get("tools")
    converted_tools: list[llm_request.Tool] | None = None
    if tools_param:
        temp_tools: list[llm_request.Tool] = []
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

    # Convert tool_choice from Responses flat format to Chat nested format
    # Responses: {"type": "function", "name": "..."}
    # Chat: {"type": "function", "function": {"name": "..."}}
    tool_choice_param = kwargs.get("tool_choice")
    if (
        isinstance(tool_choice_param, dict)
        and tool_choice_param.get("type") == "function"
        and "name" in tool_choice_param
    ):
        tool_choice_param = {"type": "function", "function": {"name": tool_choice_param["name"]}}

    resolved_tool_choice = _tool_choice_from_openai(cast(str | dict[str, Any] | None, tool_choice_param))

    return llm_request.LLMRequest(
        messages=filtered_messages,
        max_output_tokens=kwargs.get("max_output_tokens"),
        temperature=kwargs.get("temperature"),
        top_p=kwargs.get("top_p"),
        stream=bool(kwargs.get("stream", False)),
        tools=converted_tools,
        tool_choice=resolved_tool_choice,
        metadata=cast(dict[str, Any] | None, kwargs.get("metadata")),
        service_tier=kwargs.get("service_tier"),
        system_prompt=kwargs.get("instructions"),
    )
