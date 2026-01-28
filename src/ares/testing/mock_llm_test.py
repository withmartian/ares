"""Tests for mock LLM client implementation."""

import pytest

from ares.llms import request
from ares.testing import mock_llm


@pytest.mark.asyncio
async def test_mock_llm_client_records_requests():
    """Test that mock LLM client records all requests."""
    client = mock_llm.MockLLMClient()

    request1 = request.LLMRequest(messages=[{"role": "user", "content": "Hello"}])
    request2 = request.LLMRequest(messages=[{"role": "user", "content": "World"}])

    await client(request1)
    await client(request2)

    assert len(client.requests) == 2
    assert client.requests[0] == request1
    assert client.requests[1] == request2


@pytest.mark.asyncio
async def test_mock_llm_client_default_response():
    """Test that mock LLM client returns default response."""
    client = mock_llm.MockLLMClient()

    req = request.LLMRequest(messages=[{"role": "user", "content": "test"}])
    response = await client(req)

    assert response.chat_completion_response.choices[0].message.content == "Mock LLM response"
    assert response.cost == 0.0


@pytest.mark.asyncio
async def test_mock_llm_client_configured_responses():
    """Test that mock LLM client cycles through configured responses."""
    client = mock_llm.MockLLMClient(responses=["First", "Second", "Third"])

    req = request.LLMRequest(messages=[{"role": "user", "content": "test"}])

    response1 = await client(req)
    response2 = await client(req)
    response3 = await client(req)
    response4 = await client(req)  # Should cycle back to first

    assert response1.chat_completion_response.choices[0].message.content == "First"
    assert response2.chat_completion_response.choices[0].message.content == "Second"
    assert response3.chat_completion_response.choices[0].message.content == "Third"
    assert response4.chat_completion_response.choices[0].message.content == "First"


@pytest.mark.asyncio
async def test_mock_llm_client_response_handler():
    """Test that mock LLM client uses custom response handler."""

    def handler(req: request.LLMRequest) -> str:
        # Echo back the user's message
        user_msg = req.messages[-1].get("content", "")
        return f"You said: {user_msg}"

    client = mock_llm.MockLLMClient(response_handler=handler)

    req = request.LLMRequest(messages=[{"role": "user", "content": "Hello AI"}])
    response = await client(req)

    assert response.chat_completion_response.choices[0].message.content == "You said: Hello AI"


@pytest.mark.asyncio
async def test_mock_llm_client_call_count():
    """Test that mock LLM client tracks call count."""
    client = mock_llm.MockLLMClient()

    req = request.LLMRequest(messages=[{"role": "user", "content": "test"}])

    assert client.call_count == 0

    await client(req)
    assert client.call_count == 1

    await client(req)
    assert client.call_count == 2


@pytest.mark.asyncio
async def test_mock_llm_client_get_last_request():
    """Test getting the last request from client."""
    client = mock_llm.MockLLMClient()

    assert client.get_last_request() is None

    request1 = request.LLMRequest(messages=[{"role": "user", "content": "First"}])
    request2 = request.LLMRequest(messages=[{"role": "user", "content": "Second"}])

    await client(request1)
    assert client.get_last_request() == request1

    await client(request2)
    assert client.get_last_request() == request2


@pytest.mark.asyncio
async def test_mock_llm_client_get_request_messages():
    """Test getting messages from specific requests."""
    client = mock_llm.MockLLMClient()

    req = request.LLMRequest(
        messages=[
            {"role": "user", "content": "Hello"},
        ],
    )

    await client(req)

    messages = client.get_request_messages()
    assert len(messages) == 1
    assert messages[0].get("content", "") == "Hello"


@pytest.mark.asyncio
async def test_mock_llm_client_reset():
    """Test that reset() clears all data."""
    client = mock_llm.MockLLMClient()

    req = request.LLMRequest(messages=[{"role": "user", "content": "test"}])
    await client(req)
    await client(req)

    assert len(client.requests) == 2
    assert client.call_count == 2

    client.reset()

    assert len(client.requests) == 0
    assert client.call_count == 0


@pytest.mark.asyncio
async def test_mock_llm_response_structure():
    """Test that mock response has correct structure."""
    client = mock_llm.MockLLMClient()

    req = request.LLMRequest(messages=[{"role": "user", "content": "test"}])
    response = await client(req)

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
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 100
    assert completion.usage.completion_tokens == 50
    assert completion.usage.total_tokens == 150
