"""LLM client protocols and response types."""

from typing import Protocol

from ares.llms import request
from ares.llms import response


class OutputLengthExceededError(Exception):
    """Raised when LLM output exceeds the maximum allowed length.

    This matches the terminal-bench reference implementation's error handling.
    """

    def __init__(self, message: str, truncated_response: str | None = None):
        super().__init__(message)
        self.truncated_response = truncated_response


class LLMClient(Protocol):
    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse: ...
