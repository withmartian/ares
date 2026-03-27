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
        lf_response = response.make_response("Hi", input_tokens=1, output_tokens=1)
        queued.future.set_result(response.InferenceResult(response=lf_response, cost=0.0))

    answer_task = asyncio.create_task(answer_request())
    llm_response = await client(request)
    await answer_task

    assert response.extract_text_content(llm_response.response) == "Hi"
