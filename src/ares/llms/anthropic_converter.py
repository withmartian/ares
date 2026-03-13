"""Converter for Anthropic Messages request format."""

from typing import Any, cast

import anthropic.types
import linguafranca as lf

from ares.llms import open_responses
from ares.llms import request as legacy_request


def _raise_or_log(messages: list[str], *, strict: bool) -> None:
    if not messages:
        return
    joined = "; ".join(messages)
    if strict:
        raise ValueError(f"Converting to Claude Messages will lose information: {joined}")


def _filtered_warnings(warnings: list[lf.ConversionWarning]) -> list[lf.ConversionWarning]:
    return [warning for warning in warnings if warning.field not in {"metadata", "stop_sequences", "top_k"}]


def _normalize_messages(
    messages: list[legacy_request.Message],
    *,
    strict: bool,
) -> list[legacy_request.Message]:
    normalized: list[legacy_request.Message] = []
    last_role: str | None = None

    for message in messages:
        role = cast(str | None, dict(message).get("role"))
        comparable_role = "user" if role in {"tool", "function"} else role

        if comparable_role in {"user", "assistant"} and last_role == comparable_role:
            if strict:
                raise ValueError("Messages must alternate between user and assistant roles for Claude API")
            continue

        normalized.append(message)
        if comparable_role in {"user", "assistant"}:
            last_role = comparable_role

    return normalized


def to_external(request: legacy_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    loss_messages = []
    if request.service_tier not in (None, "auto", "standard_only"):
        loss_messages.append(f"service_tier='{request.service_tier}' (Claude only supports 'auto' and 'standard_only')")
    _raise_or_log(loss_messages, strict=strict)

    request = legacy_request.LLMRequest(
        messages=_normalize_messages(request.messages, strict=strict),
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        metadata=request.metadata,
        service_tier=request.service_tier,
        stop_sequences=request.stop_sequences,
        system_prompt=request.system_prompt,
        top_k=request.top_k,
    )

    canonical = open_responses.from_legacy_request(request, strict=strict)
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(canonical),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.ANTHROPIC_MESSAGES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Anthropic")

    payload = cast(dict[str, Any], result.value)
    payload.pop("model", None)
    payload.pop("service_tier", None)
    if request.max_output_tokens is None:
        payload["max_tokens"] = 1024
    if request.temperature is not None:
        payload["temperature"] = min(request.temperature / 2.0, 1.0)
    if request.top_k is not None:
        payload["top_k"] = request.top_k
    if request.tools:
        payload["tools"] = [
            {
                "type": "custom",
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["input_schema"],
            }
            for tool in request.tools
        ]
    if request.metadata:
        payload["metadata"] = request.metadata
    if request.stop_sequences:
        payload["stop_sequences"] = request.stop_sequences
    if request.service_tier in {"auto", "standard_only"}:
        payload["service_tier"] = request.service_tier
    if payload.get("stream") is False:
        payload.pop("stream", None)
    return payload


def from_external(
    kwargs: anthropic.types.MessageCreateParams,
    *,
    strict: bool = True,
) -> legacy_request.LLMRequest:
    payload = dict(kwargs)
    payload.setdefault("model", open_responses.MODEL_STUB)
    if strict and isinstance(payload.get("system"), list):
        legacy_request._extract_string_content(payload["system"], strict=True, context="System prompt")

    result = lf.convert_request_json(
        payload,
        source_format=lf.FormatName.ANTHROPIC_MESSAGES,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(
        _filtered_warnings(result.warnings),
        strict=strict,
        context="Anthropic -> Open Responses",
    )

    request = open_responses.to_legacy_request(cast(dict[str, Any], result.value), strict=strict)
    return legacy_request.LLMRequest(
        messages=request.messages,
        max_output_tokens=cast(int, payload["max_tokens"]),
        temperature=(cast(float, payload["temperature"]) * 2.0) if payload.get("temperature") is not None else None,
        top_p=request.top_p,
        stream=request.stream,
        tools=request.tools,
        tool_choice=request.tool_choice,
        metadata=cast(dict[str, Any] | None, payload.get("metadata")),
        service_tier=cast(Any, payload.get("service_tier")),
        stop_sequences=cast(list[str] | None, payload.get("stop_sequences")),
        system_prompt=request.system_prompt,
        top_k=cast(int | None, payload.get("top_k")),
    )
