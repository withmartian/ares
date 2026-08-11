"""Tests for the Chat Completions-compatible LLM client."""

import decimal

import frozendict
import openai
import openai.types.chat.chat_completion
import pytest

from ares.llms import accounting
from ares.llms import chat_completions_compatible
from ares.llms import request


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

    async def query(
        llm_client: openai.AsyncClient,
        model: str,
        req: request.LLMRequest,
    ) -> openai.types.chat.chat_completion.ChatCompletion:
        del llm_client, model, req
        return completion

    def get_cost(
        model_id: str,
        completion: openai.types.chat.chat_completion.ChatCompletion,
        *,
        cost_mapping: frozendict.frozendict[str, accounting.ModelCost],
    ) -> decimal.Decimal:
        del model_id, completion, cost_mapping
        return decimal.Decimal(0)

    def cost_list() -> frozendict.frozendict[str, accounting.ModelCost]:
        return frozendict.frozendict()

    monkeypatch.setattr(chat_completions_compatible, "_query_llm_with_retry", query)
    monkeypatch.setattr(chat_completions_compatible.accounting, "get_llm_cost", get_cost)
    monkeypatch.setattr(chat_completions_compatible.accounting, "martian_cost_list", cost_list)

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
