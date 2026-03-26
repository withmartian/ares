"""Converter for OpenAI Chat Completions request format."""

from typing import Any, cast

import linguafranca as lf
from linguafranca import types as lft
import openai.types.chat.completion_create_params

from ares.llms import open_responses

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


def _filtered_warnings(warnings: list[lf.ConversionWarning]) -> list[lf.ConversionWarning]:
    return [warning for warning in warnings if warning.field not in {"stop"}]


def to_external(request: lft.OpenResponsesRequest, *, strict: bool = True) -> dict[str, Any]:
    """Convert an Open Responses request to Chat Completions format."""
    result = lf.convert_request_json(
        open_responses.request_to_jsonable(request),
        source_format=lf.FormatName.OPEN_RESPONSES,
        target_format=lf.FormatName.OPENAI_CHAT_COMPLETIONS,
    )
    open_responses.handle_conversion_warnings(result.warnings, strict=strict, context="Open Responses -> Chat")

    payload = open_responses.normalize_chat_completions_payload(cast(dict[str, Any], result.value))
    payload.pop("model", None)
    return payload


def from_external(
    kwargs: openai.types.chat.completion_create_params.CompletionCreateParams,
    *,
    strict: bool = True,
) -> lft.OpenResponsesRequest:
    """Convert a Chat Completions request to Open Responses format."""
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

    return lft.OpenResponsesRequest(**cast(dict[str, Any], result.value))
