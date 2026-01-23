"""Tests for mock LLM client implementation."""

import pytest

from ares.llms import llm_clients
from ares.testing import mock_llm


@pytest.mark.asyncio
async def test_mock_llm_client_records_requests():
    """Test that mock LLM client records all requests."""
    client = mock_llm.MockLLMClient()

    request1 = llm_clients.LLMRequest(messages=[{"role": "user", "content": "Hello"}])
    request2 = llm_clients.LLMRequest(messages=[{"role": "user", "content": "World"}])

    await client(request1)
    await client(request2)

    assert len(client.requests) == 2
    assert client.requests[0] == request1
    assert client.requests[1] == request2


@pytest.mark.asyncio
async def test_mock_llm_client_default_response():
    """Test that mock LLM client returns default response."""
    client = mock_llm.MockLLMClient()

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])
    response = await client(request)

    assert response.chat_completion_response.choices[0].message.content == "Mock LLM response"
    assert response.cost == 0.0


@pytest.mark.asyncio
async def test_mock_llm_client_configured_responses():
    """Test that mock LLM client cycles through configured responses."""
    client = mock_llm.MockLLMClient(responses=["First", "Second", "Third"])

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])

    response1 = await client(request)
    response2 = await client(request)
    response3 = await client(request)
    response4 = await client(request)  # Should cycle back to first

    assert response1.chat_completion_response.choices[0].message.content == "First"
    assert response2.chat_completion_response.choices[0].message.content == "Second"
    assert response3.chat_completion_response.choices[0].message.content == "Third"
    assert response4.chat_completion_response.choices[0].message.content == "First"


@pytest.mark.asyncio
async def test_mock_llm_client_response_handler():
    """Test that mock LLM client uses custom response handler."""

    def handler(req: llm_clients.LLMRequest) -> str:
        # Echo back the user's message
        user_msg = req.messages[-1]["content"]
        return f"You said: {user_msg}"

    client = mock_llm.MockLLMClient(response_handler=handler)

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "Hello AI"}])
    response = await client(request)

    assert response.chat_completion_response.choices[0].message.content == "You said: Hello AI"


@pytest.mark.asyncio
async def test_mock_llm_client_call_count():
    """Test that mock LLM client tracks call count."""
    client = mock_llm.MockLLMClient()

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])

    assert client.call_count == 0

    await client(request)
    assert client.call_count == 1

    await client(request)
    assert client.call_count == 2


@pytest.mark.asyncio
async def test_mock_llm_client_get_last_request():
    """Test getting the last request from client."""
    client = mock_llm.MockLLMClient()

    assert client.get_last_request() is None

    request1 = llm_clients.LLMRequest(messages=[{"role": "user", "content": "First"}])
    request2 = llm_clients.LLMRequest(messages=[{"role": "user", "content": "Second"}])

    await client(request1)
    assert client.get_last_request() == request1

    await client(request2)
    assert client.get_last_request() == request2


@pytest.mark.asyncio
async def test_mock_llm_client_get_request_messages():
    """Test getting messages from specific requests."""
    client = mock_llm.MockLLMClient()

    request = llm_clients.LLMRequest(
        messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
    )

    await client(request)

    messages = client.get_request_messages()
    assert len(messages) == 2
    assert messages[0]["content"] == "You are helpful"
    assert messages[1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_mock_llm_client_reset():
    """Test that reset() clears all data."""
    client = mock_llm.MockLLMClient()

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])
    await client(request)
    await client(request)

    assert len(client.requests) == 2
    assert client.call_count == 2

    client.reset()

    assert len(client.requests) == 0
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_mock_llm_response_structure():
    """Test that mock response has correct structure."""
    client = mock_llm.MockLLMClient()

    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])
    response = await client(request)

    # Check response structure
    assert hasattr(response, "chat_completion_response")
    assert hasattr(response, "cost")

    # Check chat completion structure
    completion = response.chat_completion_response
    assert hasattr(completion, "id")
    assert hasattr(completion, "choices")
    assert hasattr(completion, "model")
    assert hasattr(completion, "usage")

    # Check choice structure
    assert len(completion.choices) == 1
    choice = completion.choices[0]
    assert hasattr(choice, "message")
    assert choice.message.role == "assistant"
    assert choice.finish_reason == "stop"

    # Check usage structure
    assert completion.usage.prompt_tokens == 100
    assert completion.usage.completion_tokens == 50
    assert completion.usage.total_tokens == 150
