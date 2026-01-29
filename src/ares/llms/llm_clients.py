"""LLM client protocols and response types."""

from typing import Protocol

from ares.llms import request
from ares.llms import response


class LLMClient(Protocol):
    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse: ...
