"""Tests for the PubSub proxy FastAPI application."""

import asyncio

import httpx
import pytest


@pytest.mark.asyncio
async def test_proxy_health_endpoint():
    """Test that the proxy health endpoint works."""
    from ares.pubsub_proxy import container

    proxy = container.PubSubContainer(name="test-proxy-health", port=8001)
    await proxy.start()

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8001/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
    finally:
        await proxy.stop()


@pytest.mark.asyncio
async def test_proxy_request_response_flow():
    """Test the full request/response flow through the proxy."""
    from ares.pubsub_proxy import client, container

    proxy = container.PubSubContainer(name="test-proxy-flow", port=8002)
    await proxy.start()

    mediated_client = client.PubSubMediatedLLMClient(proxy_url="http://localhost:8002")
    await mediated_client.start_polling()

    try:
        # Simulate agent making a request
        async def agent_request():
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    "http://localhost:8002/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "temperature": 0.0,
                    },
                    timeout=30.0,
                )
                return response.json()

        # Start agent request (it will block waiting for response)
        agent_task = asyncio.create_task(agent_request())

        # Get request from queue (like Environment would)
        req_and_future = await mediated_client.q.get()
        assert req_and_future.value.messages[0]["content"] == "Hello"

        # Create a mock response
        from openai.types.chat import chat_completion as cc

        mock_response_data = cc.ChatCompletion(
            id="test-id",
            choices=[
                cc.Choice(
                    finish_reason="stop",
                    index=0,
                    message=cc.ChatCompletionMessage(
                        content="Response from test",
                        role="assistant",
                    ),
                )
            ],
            created=1234567890,
            model="test-model",
            object="chat.completion",
        )

        from ares.llms import llm_clients

        mock_response = llm_clients.LLMResponse(
            chat_completion_response=mock_response_data,
            cost=0.0,
        )

        # Provide response (like Environment.step(action) would)
        req_and_future.future.set_result(mock_response)

        # Agent should receive response
        result = await agent_task
        assert result["choices"][0]["message"]["content"] == "Response from test"

    finally:
        await mediated_client.stop_polling()
        await mediated_client.close()
        await proxy.stop()


@pytest.mark.asyncio
async def test_proxy_concurrent_requests():
    """Test that proxy can handle multiple concurrent agent requests."""
    from ares.pubsub_proxy import client, container

    proxy = container.PubSubContainer(name="test-proxy-concurrent", port=8003)
    await proxy.start()

    mediated_client = client.PubSubMediatedLLMClient(proxy_url="http://localhost:8003")
    await mediated_client.start_polling()

    try:
        num_requests = 5

        # Simulate multiple agents making requests
        async def agent_request(agent_id: int):
            async with httpx.AsyncClient() as http_client:
                response = await http_client.post(
                    "http://localhost:8003/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": f"Hello from agent {agent_id}"}],
                    },
                    timeout=30.0,
                )
                return response.json()

        # Start all agent requests
        agent_tasks = [asyncio.create_task(agent_request(i)) for i in range(num_requests)]

        # Handle all requests
        for i in range(num_requests):
            req_and_future = await mediated_client.q.get()

            # Create mock response
            from openai.types.chat import chat_completion as cc

            from ares.llms import llm_clients

            mock_response = llm_clients.LLMResponse(
                chat_completion_response=cc.ChatCompletion(
                    id=f"test-id-{i}",
                    choices=[
                        cc.Choice(
                            finish_reason="stop",
                            index=0,
                            message=cc.ChatCompletionMessage(
                                content=f"Response {i}",
                                role="assistant",
                            ),
                        )
                    ],
                    created=1234567890,
                    model="test-model",
                    object="chat.completion",
                ),
                cost=0.0,
            )

            req_and_future.future.set_result(mock_response)

        # All agents should receive responses
        results = await asyncio.gather(*agent_tasks)
        assert len(results) == num_requests
        for i, result in enumerate(results):
            assert "Response" in result["choices"][0]["message"]["content"]

    finally:
        await mediated_client.stop_polling()
        await mediated_client.close()
        await proxy.stop()
