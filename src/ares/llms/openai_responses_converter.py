"""Converter for OpenAI Responses request format."""

import logging
from typing import Any, cast

import linguafranca as lf
import openai.types.responses.response_create_params

from ares.llms import open_responses
from ares.llms import request as legacy_request

_LOGGER = logging.getLogger(__name__)

_SUPPORTED_RESPONSES_FIELDS = frozenset(
    {
        "input",
        "instructions",
        "max_output_tokens",
        "metadata",
        "model",
        "service_tier",
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
        raise ValueError(f"Converting to Responses will lose information: {joined}")
    for message in messages:
        _LOGGER.warning("Open Responses identity warning: %s", message)


def _sanitize_payload_for_conversion(
    payload: dict[str, Any],
    *,
    strict: bool,
) -> tuple[dict[str, Any], list[legacy_request.Message]]:
    sanitized = dict(payload)
    fallback_messages: list[legacy_request.Message] = []

    tools = sanitized.get("tools")
    if isinstance(tools, list):
        valid_tools = []
        for tool in tools:
            tool_type = tool.get("type") if isinstance(tool, dict) else None
            if tool_type == "function":
                valid_tools.append(tool)
                continue
            message = f"Unsupported tool type for conversion: {tool_type}"
            if strict:
                raise ValueError(message)
        if valid_tools:
            sanitized["tools"] = valid_tools
        else:
            sanitized.pop("tools", None)

    input_value = sanitized.get("input")
    if isinstance(input_value, list):
        valid_inputs = []
        for item in input_value:
            if isinstance(item, dict) and item.get("type") == "function_call_output" and "call_id" not in item:
                output = item.get("output", "")
                output_str = output if isinstance(output, str) else str(output)
                if strict:
                    raise ValueError(
                        "Tool result (function_call_output) missing required 'call_id' field for routing. "
                        f"Output: {output_str[:50]}..."
                    )
                fallback_messages.append(cast(legacy_request.Message, {"role": "tool", "content": output_str}))
                continue
            valid_inputs.append(item)
        sanitized["input"] = valid_inputs

    return sanitized, fallback_messages


def to_external(request: legacy_request.LLMRequest, *, strict: bool = True) -> dict[str, Any]:
    loss_messages = []
    if request.stop_sequences:
        loss_messages.append(f"stop_sequences={request.stop_sequences} (not supported by Responses API)")
    if request.top_k is not None:
        loss_messages.append(f"top_k={request.top_k} (Claude-specific, not supported)")
    if request.service_tier == "standard_only":
        loss_messages.append("service_tier='standard_only' (not supported by Responses API)")
    _raise_or_log(loss_messages, strict=strict)

    canonical = open_responses.from_legacy_request(request, strict=strict)
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(canonical),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses identity")

    payload = cast(dict[str, Any], result.value)
    payload.pop("model", None)
    if request.service_tier and request.service_tier != "standard_only":
        payload["service_tier"] = request.service_tier
    return payload


def from_external(
    kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
    *,
    strict: bool = True,
) -> legacy_request.LLMRequest:
    payload = open_responses.validate_external_fields(
        dict(kwargs),
        allowed_fields=_SUPPORTED_RESPONSES_FIELDS,
        strict=strict,
        context="Open Responses identity",
    )
    payload.setdefault("model", open_responses.MODEL_STUB)
    payload, fallback_messages = _sanitize_payload_for_conversion(payload, strict=strict)

    result = lf.convert_request_json(
        payload,
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses identity")

    request = open_responses.to_legacy_request(cast(dict[str, Any], result.value), strict=strict)
    if not fallback_messages:
        return request

    return legacy_request.LLMRequest(
        messages=[*request.messages, *fallback_messages],
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
