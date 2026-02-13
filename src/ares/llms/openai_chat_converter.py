"""Converter for OpenAI Chat Completions API format.

This module provides bidirectional conversion between ARES's internal LLMRequest format
and the OpenAI Chat Completions API format. The module itself conforms to the
RequestConverter Protocol through its to_external and from_external functions.

Conversion Notes:
    - top_k is not supported (Claude-specific)
    - service_tier="standard_only" is not supported
    - stop_sequences truncated to 4 (OpenAI limit)
    - system_prompt is converted to/from system message in messages list
    - ToolCallMessage flattened into AssistantMessage.tool_calls
"""

import logging
from typing import Any, cast

import openai.types.chat
import openai.types.chat.completion_create_params

from ares.llms import request as llm_request

_LOGGER = logging.getLogger(__name__)


def _tool_to_chat_completions(tool: llm_request.Tool) -> openai.types.chat.ChatCompletionToolParam:
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


def _tool_from_chat_completions(chat_completions_tool: openai.types.chat.ChatCompletionToolParam) -> llm_request.Tool:
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

    return llm_request.Tool(
        name=function["name"],
        description=function.get("description", ""),
        input_schema=cast(llm_request.JSONSchema, parameters),
    )


def _tool_choice_to_openai(tool_choice: llm_request.ToolChoice | None) -> str | dict[str, Any] | None:
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
            function_data = tool_choice.get("function", {})
            if isinstance(function_data, dict) and "name" in function_data:
                return llm_request.ToolChoiceTool(type="tool", name=function_data["name"])

    return None


def to_external(request: llm_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert ARES LLMRequest to OpenAI Chat Completions format.

    Args:
        request: ARES internal request format
        strict: If True, raise ValueError on information loss. If False, log warnings.

    Returns:
        Dictionary of kwargs for openai.ChatCompletion.create() (without model)

    Raises:
        ValueError: If strict=True and information would be lost in conversion

    Note:
        Model parameter is NOT included - it should be added by the LLMClient
    """
    # Check for information loss
    lost_info = []
    if request.top_k is not None:
        lost_info.append(f"top_k={request.top_k} (Claude-specific, not supported)")
    if request.service_tier == "standard_only":
        lost_info.append("service_tier='standard_only' (not supported by Chat API)")
    if request.stop_sequences and len(request.stop_sequences) > 4:
        lost_info.append(
            f"stop_sequences truncated from {len(request.stop_sequences)} to 4 "
            f"(Chat API limit: {request.stop_sequences[4:]} will be dropped)"
        )

    if lost_info:
        msg = f"Converting to Chat Completions will lose information: {'; '.join(lost_info)}"
        if strict:
            raise ValueError(msg)
        _LOGGER.warning(msg)

    # Convert messages, flattening ToolCallMessage into AssistantMessage.tool_calls
    chat_messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    for msg in request.messages:
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
                        raise ValueError(f"ToolCallMessage found but previous message is not assistant (role={role})")
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
    if request.system_prompt:
        kwargs["messages"] = [
            {"role": "system", "content": request.system_prompt},
            *kwargs["messages"],
        ]

    # Add optional parameters (filter None values)
    if request.max_output_tokens is not None:
        kwargs["max_completion_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        kwargs["top_p"] = request.top_p
    if request.stream:
        kwargs["stream"] = True
    if request.tools:
        kwargs["tools"] = [_tool_to_chat_completions(tool) for tool in request.tools]
    if request.tool_choice is not None:
        kwargs["tool_choice"] = _tool_choice_to_openai(request.tool_choice)
    if request.metadata:
        kwargs["metadata"] = request.metadata
    if request.service_tier and request.service_tier != "standard_only":
        kwargs["service_tier"] = request.service_tier
    if request.stop_sequences:
        # OpenAI Chat supports up to 4 stop sequences
        kwargs["stop"] = request.stop_sequences[:4]

    return kwargs


def from_external(
    kwargs: openai.types.chat.completion_create_params.CompletionCreateParams,
    *,
    strict: bool = True,
) -> llm_request.LLMRequest:
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
    filtered_messages: list[llm_request.Message] = []

    for msg in kwargs["messages"]:
        role = msg.get("role")

        # Extract system/developer messages as system_prompt (use first one)
        if role in ("system", "developer"):
            if system_prompt is None:
                content = msg.get("content", "")
                system_prompt = llm_request._extract_string_content(
                    content, strict=strict, context=f"System/developer message content (role={role})"
                )
            continue

        # Validate role is supported
        if role not in llm_request._VALID_ROLES:
            if strict:
                raise ValueError(f"Unsupported message role: {role}. Must be one of {llm_request._VALID_ROLES}")
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
                assistant_msg["content"] = llm_request._extract_string_content(
                    assistant_msg["content"], strict=strict, context="Assistant message content"
                )

            filtered_messages.append(cast(llm_request.Message, assistant_msg))

            # Create separate ToolCallMessage for each tool call
            if isinstance(tool_calls_list, list):
                for tool_call in tool_calls_list:
                    if tool_call.get("type") == "function":
                        function = tool_call.get("function", {})
                        filtered_messages.append(
                            cast(
                                llm_request.Message,
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
                message_dict["content"] = llm_request._extract_string_content(
                    message_dict["content"], strict=strict, context=f"Message content (role={role})"
                )
            filtered_messages.append(cast(llm_request.Message, message_dict))

    # Convert tools from OpenAI to Claude format
    tools_param = kwargs.get("tools")
    converted_tools: list[llm_request.Tool] | None = None
    if tools_param:
        converted_tools = []
        for tool in tools_param:
            tool_type = tool.get("type")
            if tool_type != "function":
                if strict:
                    raise ValueError(f"Unsupported tool type: {tool_type}. Only 'function' tools are supported.")
                _LOGGER.warning("Skipping tool with unsupported type: %s", tool_type)
                continue
            converted_tools.append(_tool_from_chat_completions(cast(openai.types.chat.ChatCompletionToolParam, tool)))

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
        final_system_prompt = llm_request._extract_string_content(system_prompt, strict=strict, context="System prompt")

    return llm_request.LLMRequest(
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
