"""Helpers for ARES's canonical Open Responses request type."""

from collections.abc import Sequence
import dataclasses
import enum
import logging
from typing import Any, cast

import linguafranca as lf
from linguafranca import types as lft

_LOGGER = logging.getLogger(__name__)

MODEL_STUB = "__ARES_MODEL_UNSET__"


def ensure_request(request: object) -> lft.OpenResponsesRequest:
    if not isinstance(request, lft.OpenResponsesRequest):
        raise TypeError(f"Expected OpenResponsesRequest, got {type(request).__name__}")
    return request


def user_message(content: str) -> lft.InputItemMessage:
    return lft.InputItemMessage(
        content=content,
        role=lft.MessageRole.user,  # type: ignore[arg-type]
        type="message",
    )


def assistant_message(content: str) -> lft.InputItemMessage:
    return lft.InputItemMessage(
        content=content,
        role=lft.MessageRole.assistant,  # type: ignore[arg-type]
        type="message",
    )


def function_call(*, call_id: str, name: str, arguments: str) -> lft.InputItemFunctionCall:
    return lft.InputItemFunctionCall(arguments=arguments, call_id=call_id, name=name, type="function_call")


def function_call_output(*, call_id: str, output: str) -> lft.InputItemFunctionCallOutput:
    return lft.InputItemFunctionCallOutput(call_id=call_id, output=output, type="function_call_output")


def function_tool(
    *, name: str, description: str | None = None, parameters: Any | None = None, strict: bool | None = None
) -> lft.Tool:
    return lft.ToolFunction(
        name=name,
        type="function",
        description=description,
        parameters=parameters,
        strict=strict,
    )


def specific_tool_choice(name: str) -> lft.SpecificToolChoiceFunction:
    return lft.SpecificToolChoiceFunction(type="function", name=name)


def make_request(
    items: str | Sequence[lft.InputItem],
    *,
    model: str = MODEL_STUB,
    instructions: str | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    tools: Sequence[lft.Tool] | None = None,
    tool_choice: lft.ToolChoice | None = None,
    metadata: dict[str, Any] | None = None,
    parallel_tool_calls: bool | None = None,
    service_tier: lft.ServiceTier | None = None,
) -> lft.OpenResponsesRequest:
    input_value = items if isinstance(items, str) else list(items)
    return lft.OpenResponsesRequest(
        input=input_value,  # type: ignore[arg-type]
        model=model,
        instructions=instructions,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        tools=list(tools) if tools is not None else None,  # type: ignore[arg-type]
        tool_choice=tool_choice,  # type: ignore[arg-type]
        metadata=metadata,
        parallel_tool_calls=parallel_tool_calls,
        service_tier=service_tier,  # type: ignore[arg-type]
    )


def with_model(request: lft.OpenResponsesRequest, model: str) -> lft.OpenResponsesRequest:
    request = ensure_request(request)
    return dataclasses.replace(request, model=model)


def input_items(request: lft.OpenResponsesRequest) -> list[lft.InputItem]:
    request = ensure_request(request)
    if isinstance(request.input, str):
        return [user_message(request.input)]
    return list(request.input)


def message_items(request: lft.OpenResponsesRequest) -> list[lft.InputItemMessage]:
    return [item for item in input_items(request) if isinstance(item, lft.InputItemMessage)]


def extract_text_content(content: lft.InputContent, *, strict: bool = True, context: str = "content") -> str:
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []

    for part in content:
        if isinstance(part, lft.InputContentPartInputText):
            text_parts.append(part.text)
            continue
        unsupported_types.append(getattr(part, "type", type(part).__name__))

    if unsupported_types:
        msg = f"{context} contains unsupported parts: {', '.join(unsupported_types)}"
        if strict:
            raise ValueError(msg)
        _LOGGER.warning(msg)

    return "".join(text_parts)


def message_text(message: lft.InputItemMessage, *, strict: bool = True) -> str:
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


def request_to_jsonable(request: lft.OpenResponsesRequest) -> dict[str, Any]:
    request = ensure_request(request)
    return cast(dict[str, Any], to_jsonable(request))


def validate_external_fields(
    payload: dict[str, Any],
    *,
    allowed_fields: frozenset[str],
    strict: bool,
    context: str,
) -> dict[str, Any]:
    unsupported_fields = sorted(field for field in payload if field not in allowed_fields)
    if not unsupported_fields:
        return payload

    _raise_or_log_messages(
        [f"unsupported parameters: {', '.join(unsupported_fields)}"],
        strict=strict,
        context=context,
    )
    return {field: value for field, value in payload.items() if field in allowed_fields}


def _flatten_chat_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if tool_calls and flattened and flattened[-1].get("role") == "assistant":
            flattened[-1]["tool_calls"] = tool_calls
            continue
        flattened.append(message)
    return flattened


def _strip_chat_tool_strict_flags(payload: dict[str, Any]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return

    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("strict", None)


def normalize_chat_completions_payload(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages")
    if isinstance(messages, list):
        payload["messages"] = _flatten_chat_tool_call_messages(cast(list[dict[str, Any]], messages))

    _strip_chat_tool_strict_flags(payload)

    if payload.get("stream") is False:
        payload.pop("stream", None)

    return payload


def to_chat_completions_kwargs(
    request: lft.OpenResponsesRequest, *, model: str | None = None, strict: bool = True
) -> dict[str, Any]:
    request = ensure_request(request)
    request_with_model = with_model(request, model) if model is not None else request
    result = lf.convert_request_json(
        request_to_jsonable(request_with_model),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPENAI_CHAT_COMPLETIONS,
    )
    handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Chat Completions")
    payload = cast(dict[str, Any], result.value)
    # Guard against the MODEL_STUB leaking into API calls.
    if payload.get("model") == MODEL_STUB:
        _LOGGER.warning("MODEL_STUB is still set on request; stripping from Chat Completions payload.")
        payload.pop("model", None)
    return normalize_chat_completions_payload(payload)


def to_chat_messages(
    request: lft.OpenResponsesRequest, *, model: str | None = None, strict: bool = True
) -> list[dict[str, Any]]:
    kwargs = to_chat_completions_kwargs(request, model=model, strict=strict)
    return cast(list[dict[str, Any]], kwargs["messages"])


# NOTE: The following helpers are intentionally excluded from __all__ because they are
# internal plumbing used only by the converter modules:
#   extract_text_content, handle_conversion_warnings, normalize_chat_completions_payload,
#   to_jsonable, validate_external_fields

__all__ = [
    "MODEL_STUB",
    "assistant_message",
    "ensure_request",
    "function_call",
    "function_call_output",
    "function_tool",
    "input_items",
    "make_request",
    "message_items",
    "message_text",
    "request_to_jsonable",
    "specific_tool_choice",
    "to_chat_completions_kwargs",
    "to_chat_messages",
    "user_message",
    "with_model",
]
