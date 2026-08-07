"""Serialize atomic ARES responses as OpenAI Responses SSE streams."""

import json
import time
from typing import Any
import uuid

from ares.llms import response


def _sse_event(event: dict[str, Any]) -> str:
    """Serialize one Responses event in SSE wire format."""
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


def to_sse(llm_response: response.LLMResponse, *, model: str) -> str:
    """Convert one atomic ARES response into a complete Responses API stream."""
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sequence_number = 0

    def add_event(event_type: str, **values: Any) -> None:
        """Append one event with the next sequence number."""
        nonlocal sequence_number
        events.append({"type": event_type, "sequence_number": sequence_number, **values})
        sequence_number += 1

    response_base: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": model,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    add_event("response.created", response=response_base)
    add_event("response.in_progress", response=response_base)

    text = "".join(part.content for part in llm_response.data)
    if text or not llm_response.tool_calls:
        output_index = len(output)
        item_id = f"msg_{uuid.uuid4().hex}"
        empty_item = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        add_event("response.output_item.added", output_index=output_index, item=empty_item)
        empty_part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        add_event(
            "response.content_part.added",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            part=empty_part,
        )
        add_event(
            "response.output_text.delta",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            delta=text,
            logprobs=[],
        )
        add_event(
            "response.output_text.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            text=text,
            logprobs=[],
        )
        completed_part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        add_event(
            "response.content_part.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            part=completed_part,
        )
        completed_item = {**empty_item, "status": "completed", "content": [completed_part]}
        add_event("response.output_item.done", output_index=output_index, item=completed_item)
        output.append(completed_item)

    for tool_call in llm_response.tool_calls:
        output_index = len(output)
        item_id = f"fc_{uuid.uuid4().hex}"
        empty_item = {
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": "",
        }
        add_event("response.output_item.added", output_index=output_index, item=empty_item)
        add_event(
            "response.function_call_arguments.delta",
            item_id=item_id,
            output_index=output_index,
            delta=tool_call.arguments,
        )
        add_event(
            "response.function_call_arguments.done",
            item_id=item_id,
            output_index=output_index,
            arguments=tool_call.arguments,
        )
        completed_item = {**empty_item, "status": "completed", "arguments": tool_call.arguments}
        add_event("response.output_item.done", output_index=output_index, item=completed_item)
        output.append(completed_item)

    usage = {
        "input_tokens": llm_response.usage.prompt_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": llm_response.usage.generated_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": llm_response.usage.total_tokens,
    }
    add_event(
        "response.completed",
        response={**response_base, "status": "completed", "output": output, "usage": usage},
    )
    return "".join(_sse_event(event) for event in events)
