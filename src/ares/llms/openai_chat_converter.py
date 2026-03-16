"""Converter for OpenAI Chat Completions request format."""

from typing import Any, cast

import linguafranca as lf
import openai.types.chat.completion_create_params

from ares.llms import open_responses
from ares.llms import request as legacy_request

_SUPPORTED_CHAT_FIELDS = frozenset(
    {
        "max_completion_tokens",
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "service_tier",
        "stop",
        "stream",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
    }
)


def _raise_or_log(messages: list[str], *, strict: bool) -> None:
    if not messages:
        return
    joined = "; ".join(messages)
    if strict:
        raise ValueError(f"Converting to Chat Completions will lose information: {joined}")


def _filtered_warnings(warnings: list[lf.ConversionWarning]) -> list[lf.ConversionWarning]:
    return [warning for warning in warnings if warning.field not in {"stop"}]


def _flatten_tool_call_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for message in messages:
        tool_calls = message.get("tool_calls")
        if tool_calls and flattened and flattened[-1].get("role") == "assistant":
            flattened[-1]["tool_calls"] = tool_calls
            continue
        flattened.append(message)
    return flattened


def _strip_function_tool_strict_flag(payload: dict[str, Any]) -> None:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            function.pop("strict", None)


def to_external(request: legacy_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    loss_messages = []
    if request.top_k is not None:
        loss_messages.append(f"top_k={request.top_k} (Claude-specific, not supported)")
    if request.service_tier == "standard_only":
        loss_messages.append("service_tier='standard_only' (not supported by Chat API)")
    if request.stop_sequences and len(request.stop_sequences) > 4:
        loss_messages.append(f"stop_sequences truncated from {len(request.stop_sequences)} to 4 (Chat API limit)")
    _raise_or_log(loss_messages, strict=strict)

    canonical = open_responses.from_legacy_request(request, strict=strict)
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(canonical),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPENAI_CHAT_COMPLETIONS,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Chat")

    payload = cast(dict[str, Any], result.value)
    payload.pop("model", None)
    payload["messages"] = _flatten_tool_call_messages(cast(list[dict[str, Any]], payload["messages"]))
    _strip_function_tool_strict_flag(payload)
    if request.service_tier and request.service_tier != "standard_only":
        payload["service_tier"] = request.service_tier
    if request.stop_sequences:
        payload["stop"] = request.stop_sequences[:4]
    if payload.get("stream") is False:
        payload.pop("stream", None)
    return payload


def from_external(
    kwargs: openai.types.chat.completion_create_params.CompletionCreateParams,
    *,
    strict: bool = True,
) -> legacy_request.LLMRequest:
    payload = open_responses.validate_external_fields(
        dict(kwargs),
        allowed_fields=_SUPPORTED_CHAT_FIELDS,
        strict=strict,
        context="Chat -> Open Responses conversion",
    )
    payload.setdefault("model", open_responses.MODEL_STUB)

    result = lf.convert_request_json(
        payload,
        source_format=lf.FormatName.OPENAI_CHAT_COMPLETIONS,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(
        _filtered_warnings(result.warnings), strict=strict, context="Chat -> Open Responses"
    )

    request = open_responses.to_legacy_request(cast(dict[str, Any], result.value), strict=strict)

    stop = payload.get("stop")
    stop_sequences: list[str] | None = None
    if isinstance(stop, list):
        stop_sequences = cast(list[str], stop)
    elif isinstance(stop, str):
        stop_sequences = [stop]

    return legacy_request.LLMRequest(
        messages=request.messages,
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        metadata=request.metadata,
        service_tier=request.service_tier,
        stop_sequences=stop_sequences,
        system_prompt=request.system_prompt,
        top_k=request.top_k,
    )
