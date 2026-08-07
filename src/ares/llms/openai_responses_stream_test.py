"""Tests for OpenAI Responses stream serialization."""

import json

from ares.llms import openai_responses_stream
from ares.llms import response


def _parse_sse(stream: str) -> list[dict]:
    events = []
    for block in stream.strip().split("\n\n"):
        lines = block.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        events.append(json.loads(lines[1].removeprefix("data: ")))
    return events


def test_to_sse_serializes_text_and_tool_calls() -> None:
    llm_response = response.LLMResponse(
        data=[response.TextData(content="Checking the repository.")],
        cost=0.0,
        usage=response.Usage(prompt_tokens=12, generated_tokens=7),
        tool_calls=[
            response.ToolCallData(
                call_id="call_123",
                name="bash",
                arguments='{"command":"pwd"}',
            )
        ],
    )

    events = _parse_sse(openai_responses_stream.to_sse(llm_response, model="ares"))

    event_types = [event["type"] for event in events]
    assert event_types[0:2] == ["response.created", "response.in_progress"]
    assert "response.output_text.delta" in event_types
    assert "response.function_call_arguments.delta" in event_types
    assert event_types[-1] == "response.completed"

    function_call = next(
        event["item"]
        for event in events
        if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"
    )
    assert function_call["call_id"] == "call_123"
    assert function_call["name"] == "bash"
    assert function_call["arguments"] == '{"command":"pwd"}'
    assert events[-1]["response"]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
    }
