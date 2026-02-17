"""Unit tests for OpenAI Chat Completions converter."""

import decimal
import json
from pathlib import Path

import frozendict
import openai.types.chat
import openai.types.chat.chat_completion_content_part_image_param
import openai.types.chat.completion_create_params
import openai.types.shared_params
import pytest

from ares.llms import accounting
from ares.llms import openai_chat_converter
from ares.llms import request as request_lib

# Mock cost mapping for tests
_MOCK_COST_MAPPING = frozendict.frozendict(
    {
        "gpt-5": accounting.ModelCost(
            id="gpt-5",
            pricing=accounting.ModelPricing(
                prompt=decimal.Decimal("0.00003"),
                completion=decimal.Decimal("0.00006"),
                image=None,
                request=None,
                web_search=None,
                internal_reasoning=None,
            ),
        )
    }
)


def _load_test_completion(filename: str) -> openai.types.chat.ChatCompletion:
    """Load a test JSON file and convert to OpenAI ChatCompletion."""
    test_data_dir = Path(__file__).parent / "test_data"
    with open(test_data_dir / filename) as f:
        data = json.load(f)
    return openai.types.chat.ChatCompletion(**data)


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Chat Completions API conversions."""

    def test_from_chat_completion_with_structured_content_strict(self):
        """Test that structured content in chat messages raises error in strict mode."""
        kwargs = openai.types.chat.completion_create_params.CompletionCreateParamsNonStreaming(
            model="gpt-5",
            messages=[
                openai.types.chat.ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        openai.types.chat.ChatCompletionContentPartTextParam(type="text", text="What's in this image?"),
                        openai.types.chat.ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=openai.types.chat.chat_completion_content_part_image_param.ImageURL(
                                url="https://example.com/image.png"
                            ),
                        ),
                    ],
                )
            ],
        )

        with pytest.raises(ValueError, match=r"list of blocks.*structured content"):
            openai_chat_converter.ares_request_from_external(kwargs, strict=True)


class TestLLMRequestChatCompletionConversion:
    """Tests for Chat Completions API conversion."""

    def test_to_chat_completion_minimal(self):
        """Test minimal conversion to Chat Completions format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = openai_chat_converter.ares_request_to_external(request)

        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert "temperature" not in kwargs
        assert "stream" not in kwargs

    def test_to_chat_completion_all_params(self):
        """Test conversion with all common parameters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            tools=[
                {
                    "name": "test",
                    "description": "A test function",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="auto",
            metadata={"user_id": "123"},
            service_tier="default",
            stop_sequences=["STOP", "END"],
        )
        kwargs = openai_chat_converter.ares_request_to_external(request)

        assert kwargs["max_completion_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True
        # Tools should be converted to OpenAI format
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["metadata"] == {"user_id": "123"}
        assert kwargs["service_tier"] == "default"
        assert kwargs["stop"] == ["STOP", "END"]

    def test_to_chat_completion_with_system_prompt(self):
        """Test system prompt is added as first message."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful assistant.",
        )
        kwargs = openai_chat_converter.ares_request_to_external(request)

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert kwargs["messages"][1] == {"role": "user", "content": "Hello"}

    def test_to_chat_completion_stop_sequences_truncated(self):
        """Test that stop sequences are truncated to 4 (OpenAI limit)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["A", "B", "C", "D", "E", "F"],
        )
        kwargs = openai_chat_converter.ares_request_to_external(request, strict=False)

        assert kwargs["stop"] == ["A", "B", "C", "D"]

    def test_to_chat_completion_excludes_top_k(self):
        """Test that top_k (Claude-specific) is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        kwargs = openai_chat_converter.ares_request_to_external(request, strict=False)

        assert "top_k" not in kwargs

    def test_to_chat_completion_excludes_standard_only_tier(self):
        """Test that standard_only service tier is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="standard_only",
        )
        kwargs = openai_chat_converter.ares_request_to_external(request, strict=False)

        assert "service_tier" not in kwargs

    def test_from_chat_completion_minimal(self):
        """Test parsing minimal Chat Completions request."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = openai_chat_converter.ares_request_from_external(kwargs)

        assert list(request.messages) == [{"role": "user", "content": "Hello"}]
        assert request.max_output_tokens is None
        assert request.temperature is None

    def test_from_chat_completion_all_params(self):
        """Test parsing Chat Completions with all parameters."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "tools": [
                openai.types.chat.ChatCompletionToolParam(
                    type="function",
                    function=openai.types.shared_params.FunctionDefinition(
                        name="test",
                        description="A test function",
                        parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
                    ),
                )
            ],
            "tool_choice": "auto",
            "metadata": {"user_id": "123"},
            "service_tier": "default",
            "stop": ["STOP", "END"],
        }
        request = openai_chat_converter.ares_request_from_external(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        # Tools are converted from OpenAI to Claude format internally
        assert request.tools == [
            {
                "name": "test",
                "description": "A test function",
                "input_schema": {"type": "object", "properties": {"arg": {"type": "string"}}},
            }
        ]
        assert request.tool_choice == "auto"
        assert request.metadata == {"user_id": "123"}
        assert request.service_tier == "default"
        assert request.stop_sequences == ["STOP", "END"]

    def test_from_chat_completion_extracts_system_prompt(self):
        """Test that system message is extracted to system_prompt."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = openai_chat_converter.ares_request_from_external(kwargs)

        assert request.system_prompt == "You are helpful."
        assert list(request.messages) == [{"role": "user", "content": "Hello"}]

    def test_from_chat_completion_handles_max_tokens_fallback(self):
        """Test that deprecated max_tokens is used as fallback."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = openai_chat_converter.ares_request_from_external(kwargs)

        assert request.max_output_tokens == 100

    def test_to_chat_completion_flattens_tool_calls(self):
        """Test that ToolCallMessage is flattened into AssistantMessage.tool_calls."""
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="What's the weather?"),
                request_lib.AssistantMessage(role="assistant", content=""),
                request_lib.ToolCallMessage(call_id="call_123", name="get_weather", arguments='{"location":"LA"}'),
            ],
        )
        kwargs = openai_chat_converter.ares_request_to_external(request)

        # Should have 2 messages (user + assistant with tool_calls)
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][1]["role"] == "assistant"
        assert "tool_calls" in kwargs["messages"][1]
        assert len(kwargs["messages"][1]["tool_calls"]) == 1

        tool_call = kwargs["messages"][1]["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == '{"location":"LA"}'

    def test_from_chat_completion_extracts_tool_calls(self):
        """Test that AssistantMessage.tool_calls are extracted as separate ToolCallMessage."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_789",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"Seattle"}'},
                        }
                    ],
                },
            ],
        }
        request = openai_chat_converter.ares_request_from_external(kwargs)

        # Should have 3 messages: user, assistant, tool_call
        assert len(request.messages) == 3
        assert request.messages[0].get("role") == "user"
        assert request.messages[1].get("role") == "assistant"

        # Third message should be ToolCallMessage
        tool_call_msg = request.messages[2]
        assert tool_call_msg.get("call_id") == "call_789"
        assert tool_call_msg.get("name") == "get_weather"
        assert tool_call_msg.get("arguments") == '{"location":"Seattle"}'

    def test_roundtrip_chat_completion(self):
        """Test that Chat Completions roundtrip preserves data."""
        original: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "tools": [
                openai.types.chat.ChatCompletionToolParam(
                    type="function",
                    function=openai.types.shared_params.FunctionDefinition(
                        name="test",
                        description="Test tool",
                        parameters={"type": "object", "properties": {"x": {"type": "number"}}},
                    ),
                )
            ],
            "tool_choice": "auto",
            "metadata": {"user_id": "123"},
        }
        request = openai_chat_converter.ares_request_from_external(original)
        converted = openai_chat_converter.ares_request_to_external(request)

        assert converted["messages"] == original["messages"]
        assert converted["max_completion_tokens"] == original["max_completion_tokens"]
        assert converted["temperature"] == original["temperature"]
        assert converted["top_p"] == original["top_p"]
        assert converted["stream"] == original["stream"]
        assert converted["tools"] == original["tools"]
        assert converted["tool_choice"] == original["tool_choice"]
        assert converted["metadata"] == original["metadata"]


class TestLLMResponseConversion:
    """Test converting OpenAI responses to/from ARES format."""

    def test_response_text_only_roundtrip(self):
        """Round-trip: OpenAI text response → ARES → OpenAI."""
        completion = _load_test_completion("openai_text_only.json")

        # Convert to ARES format
        ares_response = openai_chat_converter.ares_response_from_external(
            completion,
            model="gpt-5",
            cost_mapping=_MOCK_COST_MAPPING,
        )

        # Verify ARES format
        assert len(ares_response.data) == 1
        assert isinstance(ares_response.data[0], openai_chat_converter.llm_response.TextData)
        assert ares_response.data[0].content == "Hello! How can I help you today?"
        assert ares_response.usage.prompt_tokens == 10
        assert ares_response.usage.generated_tokens == 9

        # Convert back to OpenAI format
        message = openai_chat_converter.ares_response_to_external(ares_response)

        # Verify round-trip
        assert message["role"] == "assistant"
        assert message["content"] == "Hello! How can I help you today?"
        assert "tool_calls" not in message

    def test_response_tool_call_single_roundtrip(self):
        """Round-trip: OpenAI single tool call → ARES → OpenAI."""
        completion = _load_test_completion("openai_tool_call_single.json")

        # Convert to ARES format
        ares_response = openai_chat_converter.ares_response_from_external(
            completion,
            model="gpt-5",
            cost_mapping=_MOCK_COST_MAPPING,
        )

        # Verify ARES format
        assert len(ares_response.data) == 1
        assert isinstance(ares_response.data[0], openai_chat_converter.llm_response.ToolUseData)
        assert ares_response.data[0].id == "call_abc123"
        assert ares_response.data[0].name == "bash"
        assert ares_response.data[0].input == {"command": "ls -la"}

        # Convert back to OpenAI format
        message = openai_chat_converter.ares_response_to_external(ares_response)

        # Verify round-trip
        assert message["role"] == "assistant"
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["id"] == "call_abc123"
        assert message["tool_calls"][0]["type"] == "function"
        assert message["tool_calls"][0]["function"]["name"] == "bash"
        assert message["tool_calls"][0]["function"]["arguments"] == '{"command": "ls -la"}'

    def test_response_tool_call_parallel_roundtrip(self):
        """Round-trip: OpenAI parallel tool calls → ARES → OpenAI."""
        completion = _load_test_completion("openai_tool_call_parallel.json")

        # Convert to ARES format
        ares_response = openai_chat_converter.ares_response_from_external(
            completion,
            model="gpt-5",
            cost_mapping=_MOCK_COST_MAPPING,
        )

        # Verify ARES format
        assert len(ares_response.data) == 3
        assert all(isinstance(d, openai_chat_converter.llm_response.ToolUseData) for d in ares_response.data)

        tool_call_0 = ares_response.data[0]
        assert isinstance(tool_call_0, openai_chat_converter.llm_response.ToolUseData)
        assert tool_call_0.id == "call_def456"
        assert tool_call_0.input == {"command": "pwd"}

        tool_call_1 = ares_response.data[1]
        assert isinstance(tool_call_1, openai_chat_converter.llm_response.ToolUseData)
        assert tool_call_1.id == "call_ghi789"
        assert tool_call_1.input == {"command": "whoami"}

        tool_call_2 = ares_response.data[2]
        assert isinstance(tool_call_2, openai_chat_converter.llm_response.ToolUseData)
        assert tool_call_2.id == "call_jkl012"
        assert tool_call_2.input == {"command": "date"}

        # Convert back to OpenAI format
        message = openai_chat_converter.ares_response_to_external(ares_response)

        # Verify round-trip
        assert message["role"] == "assistant"
        assert message["content"] is None
        assert len(message["tool_calls"]) == 3
        assert message["tool_calls"][0]["id"] == "call_def456"
        assert message["tool_calls"][1]["id"] == "call_ghi789"
        assert message["tool_calls"][2]["id"] == "call_jkl012"

    def test_response_mixed_content_roundtrip(self):
        """Round-trip: OpenAI mixed text + tool call → ARES → OpenAI."""
        completion = _load_test_completion("openai_tool_call_mixed.json")

        # Convert to ARES format
        ares_response = openai_chat_converter.ares_response_from_external(
            completion,
            model="gpt-5",
            cost_mapping=_MOCK_COST_MAPPING,
        )

        # Verify ARES format
        assert len(ares_response.data) == 2  # 1 text + 1 tool call
        assert isinstance(ares_response.data[0], openai_chat_converter.llm_response.TextData)
        assert ares_response.data[0].content == "Let me check the current directory for you."
        assert isinstance(ares_response.data[1], openai_chat_converter.llm_response.ToolUseData)
        assert ares_response.data[1].id == "call_mixed123"

        # Convert back to OpenAI format
        message = openai_chat_converter.ares_response_to_external(ares_response)

        # Verify round-trip
        assert message["role"] == "assistant"
        assert message["content"] == "Let me check the current directory for you."
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["id"] == "call_mixed123"
