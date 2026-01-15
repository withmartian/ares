"""Classes for making LLM requests."""

from collections.abc import Iterable
import dataclasses
from typing import Any, Protocol

from openai.types.chat import chat_completion as chat_completion_type
from openai.types.chat import chat_completion_message_param


# TODO: expand the request/response model for LLM reqs.
@dataclasses.dataclass(frozen=True)
class LLMRequest:
    messages: Iterable[chat_completion_message_param.ChatCompletionMessageParam]
    temperature: float | None = None

    def as_kwargs(self) -> dict[str, Any]:
        """Converts the request to a dictionary of kwargs, filtering out None values."""
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None}

    def __repr__(self) -> str:
        messages_list = list(self.messages)
        num_messages = len(messages_list)

        # Get preview of last message content
        last_msg_preview = ""
        if messages_list:
            last_msg = messages_list[-1]
            content = last_msg.get("content", "")
            if isinstance(content, str):
                last_msg_preview = content[:50]
                if len(content) > 50:
                    last_msg_preview += "..."
            elif isinstance(content, list):
                # Handle multi-part content (e.g., text + images)
                last_msg_preview = f"[{len(content)} parts]"

        # Build optional params string
        optional_parts = []
        if self.temperature is not None:
            optional_parts.append(f"temperature={self.temperature}")

        optional_str = f", {', '.join(optional_parts)}" if optional_parts else ""

        return f"LLMRequest(messages={num_messages}, last='{last_msg_preview}'{optional_str})"


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: chat_completion_type.ChatCompletion
    cost: float

    def __repr__(self) -> str:
        completion = self.chat_completion_response
        model = completion.model

        # Get token usage info
        usage = completion.usage
        if usage:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            tokens_str = f"tokens={prompt_tokens}+{completion_tokens}={total_tokens}"
        else:
            tokens_str = "tokens=unknown"

        # Get finish reason
        finish_reason = "unknown"
        if completion.choices:
            finish_reason = completion.choices[0].finish_reason or "unknown"

        # Get preview of response content
        response_preview = ""
        if completion.choices:
            content = completion.choices[0].message.content
            if content:
                response_preview = content[:50]
                if len(content) > 50:
                    response_preview += "..."

        return (
            f"LLMResponse(model='{model}', {tokens_str}, "
            f"finish='{finish_reason}', cost=${self.cost:.4f}, response='{response_preview}')"
        )


class LLMClient(Protocol):
    async def __call__(self, request: LLMRequest) -> LLMResponse: ...
