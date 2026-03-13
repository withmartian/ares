"""Helpers for ARES's canonical Open Responses request type."""

from collections.abc import Sequence
import dataclasses
import enum
import logging
from typing import Any, cast

import linguafranca as lf
from linguafranca.generated import open_responses_request as request_types
from linguafranca.generated import open_responses_stream_event as response_types

from ares.llms import request as legacy_request

_LOGGER = logging.getLogger(__name__)

MODEL_STUB = "__ARES_MODEL_UNSET__"

type OpenResponsesRequest = request_types.OpenResponsesRequest
type OpenResponsesResponse = response_types.OpenResponsesResponse
type Request = request_types.OpenResponsesRequest
type Response = response_types.OpenResponsesResponse
type InputItem = request_types.InputItem
type InputItemMessage = request_types.InputItemMessage
type InputItemFunctionCall = request_types.InputItemFunctionCall
type InputItemFunctionCallOutput = request_types.InputItemFunctionCallOutput
type Tool = request_types.Tool
type ToolChoice = request_types.ToolChoice


def user_message(content: str) -> InputItemMessage:
    return request_types.InputItemMessage(content=content, role=request_types.MessageRole.user, type="message")


def assistant_message(content: str) -> InputItemMessage:
    return request_types.InputItemMessage(content=content, role=request_types.MessageRole.assistant, type="message")


def function_call(*, call_id: str, name: str, arguments: str) -> InputItemFunctionCall:
    return request_types.InputItemFunctionCall(arguments=arguments, call_id=call_id, name=name, type="function_call")


def function_call_output(*, call_id: str, output: str) -> InputItemFunctionCallOutput:
    return request_types.InputItemFunctionCallOutput(call_id=call_id, output=output, type="function_call_output")


def function_tool(
    *, name: str, description: str | None = None, parameters: Any | None = None, strict: bool | None = None
) -> Tool:
    return request_types.ToolFunction(
        name=name,
        type="function",
        description=description,
        parameters=parameters,
        strict=strict,
    )


def specific_tool_choice(name: str) -> request_types.SpecificToolChoiceFunction:
    return request_types.SpecificToolChoiceFunction(type="function", name=name)


def make_request(
    items: str | Sequence[InputItem],
    *,
    model: str = MODEL_STUB,
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    tools: Sequence[Tool] | None = None,
    tool_choice: ToolChoice | None = None,
    metadata: dict[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    service_tier: request_types.ServiceTier | None = None,
) -> Request:
    input_value = items if isinstance(items, str) else list(items)
    return request_types.OpenResponsesRequest(
        input=input_value,
        model=model,
        instructions=instructions,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        tools=list(tools) if tools is not None else None,
        tool_choice=tool_choice,
        metadata=metadata,
        parallel_tool_calls=parallel_tool_calls,
        service_tier=service_tier,
    )


def with_model(request: Request, model: str) -> Request:
    return dataclasses.replace(request, model=model)


def input_items(request: Request) -> list[InputItem]:
    if isinstance(request.input, str):
        return [user_message(request.input)]
    return list(request.input)


def message_items(request: Request) -> list[InputItemMessage]:
    return [item for item in input_items(request) if isinstance(item, request_types.InputItemMessage)]


def extract_text_content(content: request_types.InputContent, *, strict: bool = True, context: str = "content") -> str:
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []

    for part in content:
        if isinstance(part, request_types.InputContentPartInputText):
            text_parts.append(part.text)
            continue
        unsupported_types.append(getattr(part, "type", type(part).__name__))

    if unsupported_types:
        msg = f"{context} contains unsupported parts: {', '.join(unsupported_types)}"
        if strict:
            raise ValueError(msg)
        _LOGGER.warning(msg)

    return "".join(text_parts)


def message_text(message: InputItemMessage, *, strict: bool = True) -> str:
    return extract_text_content(message.content, strict=strict, context=f"{message.role} message content")


def handle_conversion_warnings(
    warnings: Sequence[lf.ConversionWarning],
    *,
    strict: bool,
    context: str,
) -> None:
    if not warnings:
        return

    formatted = "; ".join(f"{warning.field}: {warning.message}" for warning in warnings)
    if strict:
        raise ValueError(f"Lossy conversion during {context}: {formatted}")

    for warning in warnings:
        _LOGGER.warning("%s warning for %s: %s", context, warning.field, warning.message)


def _raise_or_log_messages(messages: Sequence[str], *, strict: bool, context: str) -> None:
    if not messages:
        return

    joined = "; ".join(messages)
    if strict:
        raise ValueError(f"{context}: {joined}")

    for message in messages:
        _LOGGER.warning("%s: %s", context, message)


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, enum.Enum):
        return str(value.value)
    return str(value)


def _item_field(item: Any, field: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(field, default)
    return getattr(item, field, default)


def _request_field(request: Request | dict[str, Any], field: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(field, default)
    return getattr(request, field, default)


def _content_to_legacy_string(content: Any, *, strict: bool, context: str) -> str:
    return legacy_request._extract_string_content(content, strict=strict, context=context)


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def request_to_jsonable(request: Request) -> dict[str, Any]:
    return cast(dict[str, Any], to_jsonable(request))


def _legacy_tool_choice_to_open_responses(
    tool_choice: legacy_request.ToolChoice | None,
) -> ToolChoice | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return request_types.ToolChoiceMode.auto
    if tool_choice == "any":
        return request_types.ToolChoiceMode.required
    if tool_choice == "none":
        return request_types.ToolChoiceMode.none
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        return specific_tool_choice(tool_choice["name"])
    return None


def _legacy_service_tier_to_open_responses(
    service_tier: str | None,
    *,
    strict: bool,
) -> request_types.ServiceTier | None:
    if service_tier is None:
        return None
    try:
        return request_types.ServiceTier(service_tier)
    except ValueError:
        _raise_or_log_messages(
            [f"service_tier='{service_tier}' is not supported by Open Responses and will be dropped"],
            strict=strict,
            context="Legacy -> Open Responses conversion",
        )
        return None


def from_legacy_request(request: legacy_request.LLMRequest, *, strict: bool = True) -> Request:
    conversion_notes: list[str] = []

    input_items_list: list[InputItem] = []
    for message in request.messages:
        message_dict = dict(message)

        if "call_id" in message_dict and "name" in message_dict and "arguments" in message_dict:
            input_items_list.append(
                function_call(
                    call_id=str(message_dict["call_id"]),
                    name=str(message_dict["name"]),
                    arguments=str(message_dict["arguments"]),
                )
            )
            continue

        role = message_dict.get("role")
        if role == "tool":
            tool_call_id = message_dict.get("tool_call_id")
            if not tool_call_id:
                _raise_or_log_messages(
                    ["tool messages require tool_call_id; using empty call_id in non-strict mode"],
                    strict=strict,
                    context="Legacy -> Open Responses conversion",
                )
                tool_call_id = ""
            input_items_list.append(
                function_call_output(call_id=str(tool_call_id), output=str(message_dict.get("content", "")))
            )
            continue

        if role not in {"user", "assistant"}:
            _raise_or_log_messages(
                [f"unsupported message role '{role}'"],
                strict=strict,
                context="Legacy -> Open Responses conversion",
            )
            continue

        if "name" in message_dict:
            conversion_notes.append(f"message name '{message_dict['name']}' is not supported by Open Responses")

        content = legacy_request._extract_string_content(
            message_dict.get("content", ""),
            strict=strict,
            context=f"Legacy message content (role={role})",
        )
        if role == "user":
            input_items_list.append(user_message(content))
        else:
            input_items_list.append(assistant_message(content))

    _raise_or_log_messages(conversion_notes, strict=strict, context="Legacy -> Open Responses conversion")

    tools: list[Tool] | None = None
    if request.tools:
        tools = [
            function_tool(
                name=tool["name"],
                description=tool.get("description") or None,
                parameters=cast(dict[str, object], tool["input_schema"]),
                strict=True,
            )
            for tool in request.tools
        ]

    return make_request(
        input_items_list,
        model=MODEL_STUB,
        instructions=request.system_prompt,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=tools,
        tool_choice=_legacy_tool_choice_to_open_responses(request.tool_choice),
        metadata=request.metadata,
        service_tier=_legacy_service_tier_to_open_responses(request.service_tier, strict=strict),
    )


def to_legacy_request(request: Request | dict[str, Any], *, strict: bool = True) -> legacy_request.LLMRequest:
    unsupported_fields = [
        "background",
        "frequency_penalty",
        "include",
        "max_tool_calls",
        "parallel_tool_calls",
        "presence_penalty",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "store",
        "stream_options",
        "text",
        "top_logprobs",
        "truncation",
    ]
    conversion_notes = [
        f"{field} is not representable in legacy LLMRequest"
        for field in unsupported_fields
        if _request_field(request, field) is not None
    ]
    _raise_or_log_messages(conversion_notes, strict=strict, context="Open Responses -> legacy conversion")

    system_parts: list[str] = []
    instructions = _request_field(request, "instructions")
    if instructions:
        system_parts.append(str(instructions))

    messages: list[legacy_request.Message] = []
    input_value = _request_field(request, "input")
    if isinstance(input_value, str):
        messages.append(legacy_request.UserMessage(role="user", content=input_value))
    elif isinstance(input_value, list):
        for item in input_value:
            item_type = _item_field(item, "type")
            if item_type == "message":
                role = _enum_value(_item_field(item, "role"))
                content = _content_to_legacy_string(
                    _item_field(item, "content", ""),
                    strict=strict,
                    context=f"Open Responses message content (role={role})",
                )
                if role in {"system", "developer"}:
                    system_parts.append(content)
                    continue
                if role not in legacy_request._VALID_ROLES:
                    _raise_or_log_messages(
                        [f"unsupported message role '{role}'"],
                        strict=strict,
                        context="Open Responses -> legacy conversion",
                    )
                    continue
                messages.append(cast(legacy_request.Message, {"role": role, "content": content}))
                continue

            if item_type == "function_call":
                call_id = str(_item_field(item, "call_id", ""))
                name = str(_item_field(item, "name", ""))
                arguments = _item_field(item, "arguments", "")
                if not call_id or not name:
                    _raise_or_log_messages(
                        ["function_call items require call_id and name"],
                        strict=strict,
                        context="Open Responses -> legacy conversion",
                    )
                    continue
                messages.append(
                    cast(
                        legacy_request.Message,
                        {
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments if isinstance(arguments, str) else str(arguments),
                        },
                    )
                )
                continue

            if item_type == "function_call_output":
                call_id = _item_field(item, "call_id")
                output = _content_to_legacy_string(
                    _item_field(item, "output", ""),
                    strict=strict,
                    context="Open Responses function_call_output",
                )
                if not call_id:
                    _raise_or_log_messages(
                        ["function_call_output items require call_id"],
                        strict=strict,
                        context="Open Responses -> legacy conversion",
                    )
                    messages.append(cast(legacy_request.Message, {"role": "tool", "content": output}))
                else:
                    messages.append(
                        cast(
                            legacy_request.Message,
                            {"role": "tool", "content": output, "tool_call_id": str(call_id)},
                        )
                    )
                continue

            _raise_or_log_messages(
                [f"unsupported input item type '{item_type}'"],
                strict=strict,
                context="Open Responses -> legacy conversion",
            )

    tools: list[legacy_request.Tool] | None = None
    tools_param = _request_field(request, "tools")
    if tools_param:
        converted_tools: list[legacy_request.Tool] = []
        for tool in tools_param:
            tool_type = _item_field(tool, "type")
            if tool_type != "function":
                _raise_or_log_messages(
                    [f"unsupported tool type '{tool_type}'"],
                    strict=strict,
                    context="Open Responses -> legacy conversion",
                )
                continue
            parameters = _item_field(tool, "parameters") or {"type": "object", "properties": {}}
            if not isinstance(parameters, dict):
                _raise_or_log_messages(
                    [f"tool parameters for '{_item_field(tool, 'name', '')}' must be a dict"],
                    strict=strict,
                    context="Open Responses -> legacy conversion",
                )
                continue
            converted_tools.append(
                legacy_request.Tool(
                    name=str(_item_field(tool, "name", "")),
                    description=str(_item_field(tool, "description", "") or ""),
                    input_schema=cast(legacy_request.JSONSchema, parameters),
                )
            )
        tools = converted_tools or None

    resolved_tool_choice: legacy_request.ToolChoice | None = None
    tool_choice = _request_field(request, "tool_choice")
    tool_choice_value = _enum_value(tool_choice)
    if tool_choice_value == "auto":
        resolved_tool_choice = "auto"
    elif tool_choice_value == "required":
        resolved_tool_choice = "any"
    elif tool_choice_value == "none":
        resolved_tool_choice = "none"
    elif _item_field(tool_choice, "type") == "function":
        tool_name = _item_field(tool_choice, "name")
        if tool_name:
            resolved_tool_choice = legacy_request.ToolChoiceTool(type="tool", name=str(tool_name))
        else:
            _raise_or_log_messages(
                ["specific function tool_choice is missing a name"],
                strict=strict,
                context="Open Responses -> legacy conversion",
            )
    elif tool_choice is not None:
        _raise_or_log_messages(
            [f"unsupported tool_choice '{tool_choice}'"],
            strict=strict,
            context="Open Responses -> legacy conversion",
        )

    service_tier = _enum_value(_request_field(request, "service_tier"))
    if service_tier not in {None, "auto", "default", "flex", "scale", "priority", "standard_only"}:
        _raise_or_log_messages(
            [f"unsupported service_tier '{service_tier}'"],
            strict=strict,
            context="Open Responses -> legacy conversion",
        )
        service_tier = None

    return legacy_request.LLMRequest(
        messages=messages,
        max_output_tokens=_request_field(request, "max_output_tokens"),
        temperature=_request_field(request, "temperature"),
        top_p=_request_field(request, "top_p"),
        stream=bool(_request_field(request, "stream", False)),
        tools=tools,
        tool_choice=resolved_tool_choice,
        metadata=cast(dict[str, Any] | None, _request_field(request, "metadata")),
        service_tier=cast(Any, service_tier),
        system_prompt="\n\n".join(part for part in system_parts if part) or None,
    )


def to_chat_completions_kwargs(request: Request, *, model: str | None = None, strict: bool = True) -> dict[str, Any]:
    request_with_model = with_model(request, model) if model is not None else request
    result = lf.convert_request_json(
        request_to_jsonable(request_with_model),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPENAI_CHAT_COMPLETIONS,
    )
    handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Chat Completions")
    return cast(dict[str, Any], result.value)


def to_chat_messages(request: Request, *, model: str | None = None, strict: bool = True) -> list[dict[str, Any]]:
    kwargs = to_chat_completions_kwargs(request, model=model, strict=strict)
    return cast(list[dict[str, Any]], kwargs["messages"])


__all__ = [
    "MODEL_STUB",
    "InputItem",
    "InputItemFunctionCall",
    "InputItemFunctionCallOutput",
    "InputItemMessage",
    "OpenResponsesRequest",
    "OpenResponsesResponse",
    "Request",
    "Response",
    "Tool",
    "ToolChoice",
    "assistant_message",
    "extract_text_content",
    "from_legacy_request",
    "function_call",
    "function_call_output",
    "function_tool",
    "handle_conversion_warnings",
    "input_items",
    "make_request",
    "message_items",
    "message_text",
    "request_to_jsonable",
    "specific_tool_choice",
    "to_chat_completions_kwargs",
    "to_chat_messages",
    "to_jsonable",
    "to_legacy_request",
    "user_message",
    "with_model",
]
