"""Converter for Anthropic Messages request format."""

from typing import Any, cast

import anthropic.types
import linguafranca as lf
from linguafranca import types as lft

from ares.llms import open_responses

_SUPPORTED_ANTHROPIC_FIELDS = frozenset(
    {
        "max_tokens",
        "messages",
        "metadata",
        "model",
        "service_tier",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)


def _filtered_warnings(warnings: list[lf.ConversionWarning]) -> list[lf.ConversionWarning]:
    return [warning for warning in warnings if warning.field not in {"metadata", "stop_sequences", "top_k"}]


def to_external(request: lft.OpenResponsesRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert an Open Responses request to Anthropic Messages format."""
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(request),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.ANTHROPIC_MESSAGES,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Anthropic")

    payload = cast(dict[str, Any], result.value)
    payload.pop("model", None)

    # linguafranca sets max_tokens to its own default; ARES defaults to 1024.
    if request.max_output_tokens is None:
        payload["max_tokens"] = 1024

    if payload.get("stream") is False:
        payload.pop("stream", None)
    return payload


def from_external(
    kwargs: anthropic.types.MessageCreateParams,
    *,
    strict: bool = True,
) -> lft.OpenResponsesRequest:
    """Convert an Anthropic Messages request to Open Responses format."""
    payload = open_responses.validate_external_fields(
        dict(kwargs),
        allowed_fields=_SUPPORTED_ANTHROPIC_FIELDS,
        strict=strict,
        context="Anthropic -> Open Responses",
    )
    payload.setdefault("model", open_responses.MODEL_STUB)
    if strict and isinstance(payload.get("system"), list):
        open_responses.extract_text_content(payload["system"], strict=True, context="System prompt")

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

    return lft.OpenResponsesRequest(**cast(dict[str, Any], result.value))
