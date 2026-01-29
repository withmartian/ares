"""LLM response model."""

import dataclasses

import openai.types.chat.chat_completion


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: openai.types.chat.chat_completion.ChatCompletion
    cost: float
