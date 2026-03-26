"""Converter for OpenAI Responses request format."""

import logging
from typing import Any, cast

import linguafranca as lf
from linguafranca import types as lft
import openai.types.responses.response_create_params

from ares.llms import open_responses

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


def _sanitize_payload_for_conversion(
    payload: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    """Sanitize the payload by filtering unsupported tools and validating inputs."""
    sanitized = dict(payload)

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
            _LOGGER.warning("Skipping unsupported tool type: %s", tool_type)
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
                _LOGGER.warning("Skipping function_call_output without call_id")
                continue
            valid_inputs.append(item)
        sanitized["input"] = valid_inputs

    return sanitized


def to_external(request: lft.OpenResponsesRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert an Open Responses request to the external format (identity conversion)."""
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(request),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses identity")

    payload = cast(dict[str, Any], result.value)
    payload.pop("model", None)
    return payload


def from_external(
    kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
    *,
    strict: bool = True,
) -> lft.OpenResponsesRequest:
    """Convert an external Responses request to Open Responses format."""
    payload = open_responses.validate_external_fields(
        dict(kwargs),
        allowed_fields=_SUPPORTED_RESPONSES_FIELDS,
        strict=strict,
        context="Open Responses identity",
    )
    payload.setdefault("model", open_responses.MODEL_STUB)
    payload = _sanitize_payload_for_conversion(payload, strict=strict)

    result = lf.convert_request_json(
        payload,
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPEN_RESPONSES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses identity")

    return lft.OpenResponsesRequest(**cast(dict[str, Any], result.value))
