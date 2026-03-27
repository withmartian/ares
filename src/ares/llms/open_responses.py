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
    """Validate and return an OpenResponsesRequest.

    Args:
        request: Object to validate.

    Returns:
        The validated request.

    Raises:
        TypeError: If request is not an OpenResponsesRequest.
    """
    if not isinstance(request, lft.OpenResponsesRequest):
        raise TypeError(f"Expected OpenResponsesRequest, got {type(request).__name__}")
    return request


def user_message(content: str) -> lft.InputItemMessage:
    """Create a user message input item.

    Args:
        content: The text content of the message.

    Returns:
        An InputItemMessage with role "user".
    """
    return lft.InputItemMessage(
        content=content,
        role=lft.MessageRole.user,  # type: ignore[arg-type]
        type="message",
    )


def assistant_message(content: str) -> lft.InputItemMessage:
    """Create an assistant message input item.

    Args:
        content: The text content of the message.

    Returns:
        An InputItemMessage with role "assistant".
    """
    return lft.InputItemMessage(
        content=content,
        role=lft.MessageRole.assistant,  # type: ignore[arg-type]
        type="message",
    )


def function_call(*, call_id: str, name: str, arguments: str) -> lft.InputItemFunctionCall:
    """Create a function call input item representing an assistant's tool invocation.

    Args:
        call_id: Unique identifier for this tool call.
        name: Name of the function being called.
        arguments: JSON-encoded arguments to the function.

    Returns:
        An InputItemFunctionCall representing the tool invocation.
    """
    return lft.InputItemFunctionCall(arguments=arguments, call_id=call_id, name=name, type="function_call")


def function_call_output(*, call_id: str, output: str) -> lft.InputItemFunctionCallOutput:
    """Create a function call output item representing the result of a tool invocation.

    Args:
        call_id: Identifier matching the original function_call.
        output: The string result of the function execution.

    Returns:
        An InputItemFunctionCallOutput containing the tool result.
    """
    return lft.InputItemFunctionCallOutput(call_id=call_id, output=output, type="function_call_output")


def function_tool(
    *, name: str, description: str | None = None, parameters: Any | None = None, strict: bool | None = None
) -> lft.Tool:
    """Create a function tool definition for use in requests.

    Args:
        name: The name of the function.
        description: Human-readable description of what the function does.
        parameters: JSON Schema object describing the function's parameters.
        strict: Whether to enforce strict schema validation.

    Returns:
        A ToolFunction that can be passed to make_request.
    """
    return lft.ToolFunction(
        name=name,
        type="function",
        description=description,
        parameters=parameters,
        strict=strict,
    )


def specific_tool_choice(name: str) -> lft.SpecificToolChoiceFunction:
    """Create a tool choice that forces the model to call a specific function.

    Args:
        name: Name of the function the model must call.

    Returns:
        A SpecificToolChoiceFunction for use as tool_choice in make_request.
    """
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
    """Create an OpenResponsesRequest from input items and optional parameters.

    Args:
        items: Either a string (treated as a user message) or a sequence of input items.
        model: Model identifier. Defaults to MODEL_STUB which must be replaced before API calls.
        instructions: System instructions for the model.
        max_output_tokens: Maximum tokens in the response.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling parameter.
        stream: Whether to stream the response.
        tools: List of tool definitions the model may call.
        tool_choice: How the model should choose tools ("auto", "none", or specific).
        metadata: Arbitrary metadata to attach to the request.
        parallel_tool_calls: Whether the model can make multiple tool calls in parallel.
        service_tier: Service tier for the request.

    Returns:
        A fully-formed OpenResponsesRequest ready for use with an LLMClient.
    """
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
    """Return a copy of the request with the model field replaced.

    Args:
        request: The original request.
        model: The new model identifier.

    Returns:
        A new request with the updated model field.
    """
    request = ensure_request(request)
    return dataclasses.replace(request, model=model)


def input_items(request: lft.OpenResponsesRequest) -> list[lft.InputItem]:
    """Extract input items from a request, normalizing string inputs to user messages.

    Args:
        request: The request to extract items from.

    Returns:
        List of input items. If the request input was a string, returns a single-element
        list containing a user message with that content.
    """
    request = ensure_request(request)
    if isinstance(request.input, str):
        return [user_message(request.input)]
    return list(request.input)


def message_items(request: lft.OpenResponsesRequest) -> list[lft.InputItemMessage]:
    """Extract only message items from a request, filtering out function calls and other types.

    Args:
        request: The request to extract messages from.

    Returns:
        List of InputItemMessage objects (user and assistant messages only).
    """
    return [item for item in input_items(request) if isinstance(item, lft.InputItemMessage)]


def extract_text_content(
    content: str | Sequence[lft.InputContentPart], *, strict: bool = True, context: str = "content"
) -> str:
    if isinstance(content, str):
        return content

    text_parts: list[str] = []
    unsupported_types: list[str] = []

    for part in content:
        if isinstance(part, lft.ContentPartInputText):
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
    """Extract text content from a message.

    Args:
        message: The message to extract text from.
        strict: If True, raise ValueError for unsupported content types. If False, log a
            warning and skip them.

    Returns:
        The concatenated text content of the message.

    Raises:
        ValueError: If strict=True and the message contains unsupported content parts.
    """
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
    """Convert an OpenResponsesRequest to a JSON-serializable dictionary.

    Args:
        request: The request to convert.

    Returns:
        A dictionary suitable for JSON serialization.
    """
    request = ensure_request(request)
    return cast(dict[str, Any], to_jsonable(request))


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
    """Convert an OpenResponsesRequest to OpenAI Chat Completions API kwargs.

    Args:
        request: The Open Responses request to convert.
        model: Optional model override. If provided, replaces the request's model field.
        strict: If True, raise ValueError on lossy conversions. If False, log warnings.

    Returns:
        Dictionary of kwargs suitable for passing to openai.chat.completions.create().

    Raises:
        ValueError: If strict=True and the conversion would lose information.
    """
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
    """Convert an OpenResponsesRequest to Chat Completions message format.

    This is a convenience wrapper around to_chat_completions_kwargs that returns
    only the messages array.

    Args:
        request: The Open Responses request to convert.
        model: Optional model override.
        strict: If True, raise ValueError on lossy conversions.

    Returns:
        List of message dictionaries in Chat Completions format.

    Raises:
        ValueError: If strict=True and the conversion would lose information.
    """
    kwargs = to_chat_completions_kwargs(request, model=model, strict=strict)
    return cast(list[dict[str, Any]], kwargs["messages"])

