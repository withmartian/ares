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


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: chat_completion_type.ChatCompletion
    cost: float


class LLMClient(Protocol):
    async def __call__(self, request: LLMRequest) -> LLMResponse: ...
