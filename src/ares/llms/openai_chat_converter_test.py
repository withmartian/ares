"""Unit tests for OpenAI Chat Completions converter."""

from linguafranca import types as lft
import openai.types.chat
import openai.types.chat.chat_completion_content_part_image_param
import openai.types.chat.completion_create_params
import openai.types.shared_params
import pytest

from ares.llms import open_responses
from ares.llms import openai_chat_converter


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Chat Completions API conversions."""

    def test_from_chat_completion_with_structured_content_parses(self):
        """Test that structured content in chat messages is passed through."""
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

        # Linguafranca passes structured content through to Open Responses format
        request = openai_chat_converter.from_external(kwargs, strict=False)
        items = open_responses.input_items(request)
        assert len(items) == 1


class TestOpenResponsesChatCompletionConversion:
    """Tests for Chat Completions API conversion using Open Responses requests."""

    def test_to_chat_completion_minimal(self):
        """Test minimal conversion to Chat Completions format."""
        request = open_responses.make_request([open_responses.user_message("Hello")])
        kwargs = openai_chat_converter.to_external(request)

        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert "temperature" not in kwargs
        assert "stream" not in kwargs

    def test_to_chat_completion_all_params(self):
        """Test conversion with all common parameters."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            tools=[
                open_responses.function_tool(
                    name="test",
                    description="A test function",
                    parameters={"type": "object", "properties": {}},
                )
            ],
            tool_choice=lft.ToolChoiceMode.auto,
            metadata={"user_id": "123"},
            service_tier="default",  # type: ignore[arg-type]
        )
        kwargs = openai_chat_converter.to_external(request)

        assert kwargs["max_completion_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True
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

    def test_to_chat_completion_with_instructions(self):
        """Test instructions is added as system message."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            instructions="You are a helpful assistant.",
        )
        kwargs = openai_chat_converter.to_external(request)

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert kwargs["messages"][1] == {"role": "user", "content": "Hello"}

    def test_from_chat_completion_minimal(self):
        """Test parsing minimal Chat Completions request."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = openai_chat_converter.from_external(kwargs)

        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 1
        assert jsonable["input"][0]["content"] == "Hello"
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
        }
        request = openai_chat_converter.from_external(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.metadata == {"user_id": "123"}

    def test_from_chat_completion_preserves_system_message(self):
        """Test that system message is preserved in input."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = openai_chat_converter.from_external(kwargs)

        # Linguafranca preserves system messages in input rather than extracting to instructions
        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 2
        assert jsonable["input"][0]["role"] == "system"
        assert jsonable["input"][0]["content"] == "You are helpful."
        assert jsonable["input"][1]["content"] == "Hello"

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
        """Test that function_call items are flattened into assistant tool_calls."""
        request = open_responses.make_request(
            [
                open_responses.user_message("What's the weather?"),
                open_responses.assistant_message(""),
                open_responses.function_call(call_id="call_123", name="get_weather", arguments='{"location":"LA"}'),
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

    def test_from_chat_completion_rejects_unknown_params_in_strict_mode(self):
        """Test that strict mode rejects unhandled chat parameters."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "unexpected_flag": 123,  # type: ignore[typeddict-item]
        }

        with pytest.raises(ValueError, match=r"unsupported parameters: unexpected_flag"):
            openai_chat_converter.from_external(kwargs, strict=True)

    def test_from_chat_completion_extracts_tool_calls(self):
        """Test that AssistantMessage.tool_calls are extracted as separate function_call items."""
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

        items = open_responses.input_items(request)
        # Should have 3 items: user message, assistant message, function_call
        assert len(items) == 3

        # Verify the function_call item
        jsonable = open_responses.request_to_jsonable(request)
        input_items = jsonable["input"]
        assert input_items[2]["type"] == "function_call"
        assert input_items[2]["call_id"] == "call_789"
        assert input_items[2]["name"] == "get_weather"

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
