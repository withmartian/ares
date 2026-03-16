"""Tests for mock LLM client implementation."""

from typing import cast

import pytest

from ares.llms import open_responses
from ares.testing import mock_llm


@pytest.mark.asyncio
async def test_mock_llm_client_records_requests():
    """Test that mock LLM client records all requests."""
    client = mock_llm.MockLLMClient()

    request1 = open_responses.make_request([open_responses.user_message("Hello")])
    request2 = open_responses.make_request([open_responses.user_message("World")])

    await client(request1)
    await client(request2)

    assert len(client.requests) == 2
    assert client.requests[0] == request1
    assert client.requests[1] == request2


@pytest.mark.asyncio
async def test_mock_llm_client_default_response():
    """Test that mock LLM client returns default response."""
    client = mock_llm.MockLLMClient()

    req = open_responses.make_request([open_responses.user_message("test")])
    response = await client(req)

    assert response.data[0].content == "Mock LLM response"
    assert response.cost == 0.0


@pytest.mark.asyncio
async def test_mock_llm_client_configured_responses():
    """Test that mock LLM client cycles through configured responses."""
    client = mock_llm.MockLLMClient(responses=["First", "Second", "Third"])

    req = open_responses.make_request([open_responses.user_message("test")])

    response1 = await client(req)
    response2 = await client(req)
    response3 = await client(req)
    response4 = await client(req)  # Should cycle back to first

    assert response1.data[0].content == "First"
    assert response2.data[0].content == "Second"
    assert response3.data[0].content == "Third"
    assert response4.data[0].content == "First"


@pytest.mark.asyncio
async def test_mock_llm_client_response_handler():
    """Test that mock LLM client uses custom response handler."""

    def handler(req: open_responses.Request) -> str:
        # Echo back the user's message
        user_msg = open_responses.message_text(open_responses.message_items(req)[-1])
        return f"You said: {user_msg}"

    client = mock_llm.MockLLMClient(response_handler=handler)

    req = open_responses.make_request([open_responses.user_message("Hello AI")])
    response = await client(req)

    assert response.data[0].content == "You said: Hello AI"


@pytest.mark.asyncio
async def test_mock_llm_client_call_count():
    """Test that mock LLM client tracks call count."""
    client = mock_llm.MockLLMClient()

    req = open_responses.make_request([open_responses.user_message("test")])

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

    request1 = open_responses.make_request([open_responses.user_message("First")])
    request2 = open_responses.make_request([open_responses.user_message("Second")])

    await client(request1)
    assert client.get_last_request() == request1

    await client(request2)
    assert client.get_last_request() == request2


@pytest.mark.asyncio
async def test_mock_llm_client_get_request_messages():
    """Test getting messages from specific requests."""
    client = mock_llm.MockLLMClient()

    req = open_responses.make_request([open_responses.user_message("Hello")])

    await client(req)

    messages = client.get_request_messages()
    assert len(messages) == 1
    assert open_responses.message_text(cast(open_responses.InputItemMessage, messages[0])) == "Hello"


@pytest.mark.asyncio
async def test_mock_llm_client_get_request_messages_includes_tool_items():
    """Test that request item helper returns tool call items too."""
    client = mock_llm.MockLLMClient()

    req = open_responses.make_request(
        [
            open_responses.function_call(call_id="call_1", name="search", arguments="{}"),
            open_responses.function_call_output(call_id="call_1", output="done"),
        ]
    )

    await client(req)

    items = client.get_request_messages()
    assert len(items) == 2
    assert items[0].type == "function_call"
    assert items[1].type == "function_call_output"


@pytest.mark.asyncio
async def test_mock_llm_client_reset():
    """Test that reset() clears all data."""
    client = mock_llm.MockLLMClient()

    req = open_responses.make_request([open_responses.user_message("test")])
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

    req = open_responses.make_request([open_responses.user_message("test")])
    response = await client(req)

    # Check response structure
    assert hasattr(response, "data")
    assert hasattr(response, "cost")
    assert hasattr(response, "usage")

    # Check data structure
    assert len(response.data) == 1
    assert hasattr(response.data[0], "content")
    assert response.data[0].content == "Mock LLM response"

    # Check usage structure
    assert response.usage.prompt_tokens == 100
    assert response.usage.generated_tokens == 50
    assert response.usage.total_tokens == 150
