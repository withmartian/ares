"""Mock LLM client implementation for testing."""

from collections.abc import Callable
import dataclasses

from ares.llms import request
from ares.llms import response


@dataclasses.dataclass
class MockLLMClient:
    """Mock LLM client for testing without real API calls.

    This mock allows tests to verify LLM interactions without making actual
    API requests. It records all requests and allows configuring responses.

    Attributes:
        requests: List of all LLMRequest objects received.
        responses: List of response strings to return (cycles through them).
        response_handler: Optional function to dynamically generate responses.
        default_response: Default response if no responses configured.
        call_count: Number of times the client has been called.
    """

    requests: list[request.LLMRequest] = dataclasses.field(default_factory=list)
    responses: list[str] = dataclasses.field(default_factory=list)
    response_handler: Callable[[request.LLMRequest], str] | None = None
    default_response: str = "Mock LLM response"
    call_count: int = 0

    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse:
        """Process LLM request and return mock response.

        Args:
            request: The LLM request to process.

        Returns:
            LLMResponse with mock data.
        """
        self.requests.append(request)
        self.call_count += 1

        # Generate response content
        if self.response_handler:
            response_text = self.response_handler(request)
        elif self.responses:
            # Cycle through configured responses
            response_text = self.responses[(self.call_count - 1) % len(self.responses)]
        else:
            response_text = self.default_response

        return response.LLMResponse(
            data=[response.TextData(content=response_text)],
            cost=0.0,
            usage=response.Usage(prompt_tokens=100, generation_tokens=50),
        )

    def get_last_request(self) -> request.LLMRequest | None:
        """Get the most recent request, or None if no requests."""
        return self.requests[-1] if self.requests else None

    def get_request_messages(self, index: int = -1) -> list[request.Message]:
        """Get messages from a specific request (default: last request)."""
        if not self.requests:
            return []
        return self.requests[index].messages

    def reset(self) -> None:
        """Clear all recorded data."""
        self.requests.clear()
        self.call_count = 0
