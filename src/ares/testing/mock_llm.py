"""Mock LLM client implementation for testing."""

from collections.abc import Callable
import dataclasses
import time
import uuid

import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage

from ares.llms import llm_clients


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

    requests: list[llm_clients.LLMRequest] = dataclasses.field(default_factory=list)
    responses: list[str] = dataclasses.field(default_factory=list)
    response_handler: Callable[[llm_clients.LLMRequest], str] | None = None
    default_response: str = "Mock LLM response"
    call_count: int = 0

    async def __call__(self, request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
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

        # Build mock ChatCompletion response
        chat_completion = openai.types.chat.chat_completion.ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                openai.types.chat.chat_completion.Choice(
                    message=openai.types.chat.chat_completion_message.ChatCompletionMessage(
                        content=response_text,
                        role="assistant",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=int(time.time()),
            model="mock-llm",
            object="chat.completion",
            usage=openai.types.completion_usage.CompletionUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )

        return llm_clients.LLMResponse(
            chat_completion_response=chat_completion,
            cost=0.0,
        )

    def get_last_request(self) -> llm_clients.LLMRequest | None:
        """Get the most recent request, or None if no requests."""
        return self.requests[-1] if self.requests else None

    def get_request_messages(self, index: int = -1) -> list[dict]:
        """Get messages from a specific request (default: last request)."""
        if not self.requests:
            return []
        return self.requests[index].messages

    def reset(self) -> None:
        """Clear all recorded data."""
        self.requests.clear()
        self.call_count = 0
