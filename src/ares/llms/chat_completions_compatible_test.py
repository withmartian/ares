"""Tests for the Chat Completions-compatible LLM client."""

import openai.types.chat.chat_completion
import pytest

from ares.llms import chat_completions_compatible


@pytest.mark.asyncio
async def test_client_preserves_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    completion = openai.types.chat.chat_completion.ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }
    )

    async def query(*args, **kwargs):
        del args, kwargs
        return completion

    def get_cost(*args, **kwargs):
        del args, kwargs
        return 0.0

    monkeypatch.setattr(chat_completions_compatible, "_query_llm_with_retry", query)
    monkeypatch.setattr(chat_completions_compatible.accounting, "get_llm_cost", get_cost)
    monkeypatch.setattr(chat_completions_compatible.accounting, "martian_cost_list", lambda: {})

    client = chat_completions_compatible.ChatCompletionCompatibleLLMClient(
        model="test-model",
        base_url="http://unused",
        api_key="unused",
    )
    result = await client(
        chat_completions_compatible.request.LLMRequest(messages=[{"role": "user", "content": "Run pwd"}])
    )

    assert result.data[0].content == ""
    assert result.tool_calls[0].call_id == "call_123"
    assert result.tool_calls[0].name == "bash"
    assert result.tool_calls[0].arguments == '{"command":"pwd"}'
