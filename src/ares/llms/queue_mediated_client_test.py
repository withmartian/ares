"""Tests for the queue-mediated LLM client."""

import asyncio

import pytest

from ares.llms import open_responses
from ares.llms import queue_mediated_client
from ares.llms import response


@pytest.mark.asyncio
async def test_queue_mediated_client_roundtrips_canonical_requests():
    client = queue_mediated_client.QueueMediatedLLMClient()
    request = open_responses.make_request([open_responses.user_message("Hello")])

    async def answer_request() -> None:
        queued = await client.q.get()
        assert queued.value == request
        queued.future.set_result(
            response.LLMResponse(
                data=[response.TextData(content="Hi")],
                cost=0.0,
                usage=response.Usage(prompt_tokens=1, generated_tokens=1),
            )
        )

    answer_task = asyncio.create_task(answer_request())
    llm_response = await client(request)
    await answer_task

    assert llm_response.data[0].content == "Hi"
