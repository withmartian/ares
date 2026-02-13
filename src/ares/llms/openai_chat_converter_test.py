"""Unit tests for OpenAI Chat Completions converter."""

import openai.types.chat
import openai.types.chat.chat_completion_content_part_image_param
import openai.types.chat.completion_create_params
import openai.types.shared_params
import pytest

from ares.llms import openai_chat_converter
from ares.llms import request as request_lib


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Chat Completions API conversions."""

    def test_from_chat_completion_with_structured_content_strict(self):
        """Test that structured content in chat messages raises error in strict mode."""
        kwargs = openai.types.chat.completion_create_params.CompletionCreateParamsNonStreaming(
            model="gpt-4",
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
            openai_chat_converter.from_external(kwargs, strict=True)


class TestLLMRequestChatCompletionConversion:
    """Tests for Chat Completions API conversion."""

    def test_to_chat_completion_minimal(self):
        """Test minimal conversion to Chat Completions format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = openai_chat_converter.to_external(request)

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
        kwargs = openai_chat_converter.to_external(request)

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
        kwargs = openai_chat_converter.to_external(request)

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert kwargs["messages"][1] == {"role": "user", "content": "Hello"}

    def test_to_chat_completion_stop_sequences_truncated(self):
        """Test that stop sequences are truncated to 4 (OpenAI limit)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["A", "B", "C", "D", "E", "F"],
        )
        kwargs = openai_chat_converter.to_external(request, strict=False)

        assert kwargs["stop"] == ["A", "B", "C", "D"]

    def test_to_chat_completion_excludes_top_k(self):
        """Test that top_k (Claude-specific) is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        kwargs = openai_chat_converter.to_external(request, strict=False)

        assert "top_k" not in kwargs

    def test_to_chat_completion_excludes_standard_only_tier(self):
        """Test that standard_only service tier is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="standard_only",
        )
        kwargs = openai_chat_converter.to_external(request, strict=False)

        assert "service_tier" not in kwargs

    def test_from_chat_completion_minimal(self):
        """Test parsing minimal Chat Completions request."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = openai_chat_converter.from_external(kwargs)

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
        request = openai_chat_converter.from_external(kwargs)

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
        request = openai_chat_converter.from_external(kwargs)

        assert request.system_prompt == "You are helpful."
        assert list(request.messages) == [{"role": "user", "content": "Hello"}]

    def test_from_chat_completion_handles_max_tokens_fallback(self):
        """Test that deprecated max_tokens is used as fallback."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = openai_chat_converter.from_external(kwargs)

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
        kwargs = openai_chat_converter.to_external(request)

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
        request = openai_chat_converter.from_external(kwargs)

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
        request = openai_chat_converter.from_external(original)
        converted = openai_chat_converter.to_external(request)

        assert converted["messages"] == original["messages"]
        assert converted["max_completion_tokens"] == original["max_completion_tokens"]
        assert converted["temperature"] == original["temperature"]
        assert converted["top_p"] == original["top_p"]
        assert converted["stream"] == original["stream"]
        assert converted["tools"] == original["tools"]
        assert converted["tool_choice"] == original["tool_choice"]
        assert converted["metadata"] == original["metadata"]
