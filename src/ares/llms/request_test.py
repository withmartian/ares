"""Unit tests for request_lib.LLMRequest conversion methods."""

from typing import cast

import anthropic.types
import openai.types.chat
import openai.types.chat.completion_create_params
import openai.types.responses.response_create_params
import openai.types.shared_params
import pytest

from ares.llms import request as request_lib


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in API conversions."""

    def test_from_responses_with_structured_content_strict(self):
        """Test that structured content raises error in strict mode."""
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsBase(
            model="gpt-4",
            input=[
                openai.types.responses.EasyInputMessageParam(
                    type="message",
                    role="user",
                    content=[
                        openai.types.responses.ResponseInputTextParam(type="input_text", text="Hello"),
                        openai.types.responses.ResponseInputImageParam(
                            type="input_image", detail="auto", image_url="data:image/png;base64,..."
                        ),
                    ],
                )
            ],
        )

        with pytest.raises(ValueError, match=r"list of blocks.*structured content"):
            request_lib.LLMRequest.from_responses(kwargs, strict=True)

    def test_from_responses_with_structured_content_non_strict(self):
        """Test that structured content returns empty string in non-strict mode."""
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsBase(
            model="gpt-4",
            input=[
                openai.types.responses.EasyInputMessageParam(
                    type="message",
                    role="user",
                    content=[
                        openai.types.responses.ResponseInputTextParam(type="input_text", text="Hello"),
                        openai.types.responses.ResponseInputImageParam(
                            type="input_image", detail="auto", image_url="data:image/png;base64,..."
                        ),
                    ],
                )
            ],
        )

        # Should not raise, but content will be empty
        request = request_lib.LLMRequest.from_responses(kwargs, strict=False)
        assert len(request.messages) == 1
        assert request.messages[0].get("content") == ""

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
            request_lib.LLMRequest.from_chat_completion(kwargs, strict=True)

    def test_from_messages_with_structured_content_strict(self):
        """Test that structured content in Claude messages raises error in strict mode."""
        kwargs = anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            model="claude-3-opus",
            max_tokens=100,
            messages=[
                anthropic.types.MessageParam(
                    role="user",
                    content=[
                        anthropic.types.TextBlockParam(type="text", text="Analyze this image"),
                        anthropic.types.ImageBlockParam(
                            type="image",
                            source=anthropic.types.base64_image_source_param.Base64ImageSourceParam(
                                type="base64", media_type="image/png", data="..."
                            ),
                        ),
                    ],
                )
            ],
        )

        with pytest.raises(ValueError, match=r"list of blocks.*structured content"):
            request_lib.LLMRequest.from_messages(kwargs, strict=True)

    def test_system_prompt_with_structured_content_strict(self):
        """Test that structured system prompt raises error in strict mode."""
        kwargs = anthropic.types.message_create_params.MessageCreateParamsNonStreaming(
            model="claude-3-opus",
            max_tokens=100,
            system=[
                anthropic.types.TextBlockParam(type="text", text="You are a helpful assistant."),
                anthropic.types.TextBlockParam(type="text", text="Always be concise."),
            ],
            messages=[anthropic.types.MessageParam(role="user", content="Hello")],
        )

        with pytest.raises(ValueError, match=r"list of blocks.*structured content"):
            request_lib.LLMRequest.from_messages(kwargs, strict=True)


class TestLLMRequestChatCompletionConversion:
    """Tests for Chat Completions API conversion."""

    def test_to_chat_completion_minimal(self):
        """Test minimal conversion to Chat Completions format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = request.to_chat_completion_kwargs()

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
        kwargs = request.to_chat_completion_kwargs()

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
        kwargs = request.to_chat_completion_kwargs()

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert kwargs["messages"][1] == {"role": "user", "content": "Hello"}

    def test_to_chat_completion_stop_sequences_truncated(self):
        """Test that stop sequences are truncated to 4 (OpenAI limit)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["A", "B", "C", "D", "E", "F"],
        )
        kwargs = request.to_chat_completion_kwargs(strict=False)

        assert kwargs["stop"] == ["A", "B", "C", "D"]

    def test_to_chat_completion_excludes_top_k(self):
        """Test that top_k (Claude-specific) is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        kwargs = request.to_chat_completion_kwargs(strict=False)

        assert "top_k" not in kwargs

    def test_to_chat_completion_excludes_standard_only_tier(self):
        """Test that standard_only service tier is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="standard_only",
        )
        kwargs = request.to_chat_completion_kwargs(strict=False)

        assert "service_tier" not in kwargs

    def test_from_chat_completion_minimal(self):
        """Test parsing minimal Chat Completions request."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request = request_lib.LLMRequest.from_chat_completion(kwargs)

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
        request = request_lib.LLMRequest.from_chat_completion(kwargs)

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
        request = request_lib.LLMRequest.from_chat_completion(kwargs)

        assert request.system_prompt == "You are helpful."
        assert list(request.messages) == [{"role": "user", "content": "Hello"}]

    def test_from_chat_completion_handles_max_tokens_fallback(self):
        """Test that deprecated max_tokens is used as fallback."""
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = request_lib.LLMRequest.from_chat_completion(kwargs)

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
        kwargs = request.to_chat_completion_kwargs()

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
        request = request_lib.LLMRequest.from_chat_completion(kwargs)

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
        request = request_lib.LLMRequest.from_chat_completion(original)
        converted = request.to_chat_completion_kwargs()

        assert converted["messages"] == original["messages"]
        assert converted["max_completion_tokens"] == original["max_completion_tokens"]
        assert converted["temperature"] == original["temperature"]
        assert converted["top_p"] == original["top_p"]
        assert converted["stream"] == original["stream"]
        assert converted["tools"] == original["tools"]
        assert converted["tool_choice"] == original["tool_choice"]
        assert converted["metadata"] == original["metadata"]


class TestLLMRequestResponsesConversion:
    """Tests for Responses API conversion."""

    def test_to_responses_minimal(self):
        """Test minimal conversion to Responses format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = request.to_responses_kwargs()

        assert kwargs["input"] == [{"type": "message", "role": "user", "content": "Hello"}]
        assert "temperature" not in kwargs

    def test_to_responses_all_params(self):
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
        )
        kwargs = request.to_responses_kwargs()

        assert kwargs["max_output_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True
        # Tools should be converted to Responses format (flat structure)
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "test",
                "description": "A test function",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ]
        assert kwargs["tool_choice"] == "auto"
        assert kwargs["metadata"] == {"user_id": "123"}
        assert kwargs["service_tier"] == "default"

    def test_to_responses_with_instructions(self):
        """Test system prompt is mapped to instructions."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful assistant.",
        )
        kwargs = request.to_responses_kwargs()

        assert kwargs["instructions"] == "You are a helpful assistant."

    def test_to_responses_excludes_stop_sequences(self):
        """Test that stop_sequences are not included (not supported)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP"],
        )
        kwargs = request.to_responses_kwargs(strict=False)

        assert "stop_sequences" not in kwargs
        assert "stop" not in kwargs

    def test_to_responses_excludes_top_k(self):
        """Test that top_k is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        kwargs = request.to_responses_kwargs(strict=False)

        assert "top_k" not in kwargs

    def test_from_responses_minimal(self):
        """Test parsing minimal Responses request."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        request = request_lib.LLMRequest.from_responses(kwargs)

        assert list(request.messages) == [{"role": "user", "content": "Hello"}]

    def test_from_responses_all_params(self):
        """Test parsing Responses with all parameters."""
        tool: openai.types.responses.FunctionToolParam = openai.types.responses.FunctionToolParam(
            type="function",
            name="test",
            description="A test function",
            parameters={"type": "object", "properties": {"x": {"type": "number"}}},
            strict=None,
        )
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "max_output_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "tools": [tool],
            "tool_choice": "auto",
            "metadata": {"user_id": "123"},
            "service_tier": "default",
            "instructions": "You are helpful.",
        }
        request = request_lib.LLMRequest.from_responses(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        # Tools should be converted from OpenAI to Claude format
        assert request.tools == [
            {
                "name": "test",
                "description": "A test function",
                "input_schema": {"type": "object", "properties": {"x": {"type": "number"}}},
            }
        ]
        assert request.tool_choice == "auto"
        assert request.metadata == {"user_id": "123"}
        assert request.service_tier == "default"
        assert request.system_prompt == "You are helpful."

    def test_from_responses_string_input(self):
        """Test parsing Responses with string input."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "input": "Hello, world!",
        }
        request = request_lib.LLMRequest.from_responses(kwargs)

        assert list(request.messages) == [{"role": "user", "content": "Hello, world!"}]

    def test_from_responses_list_input(self):
        """Test parsing Responses with list input."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi!"},
            ],
        }
        request = request_lib.LLMRequest.from_responses(kwargs)

        assert list(request.messages) == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_to_responses_includes_tool_call_id(self):
        """Test that tool messages are converted to function_call_output format with call_id."""
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="What's the weather?"),
                request_lib.AssistantMessage(role="assistant", content="Let me check..."),
                request_lib.ToolCallResponseMessage(role="tool", content="Sunny, 72°F", tool_call_id="call_123"),
            ],
        )
        kwargs = request.to_responses_kwargs()

        # Tool message should be converted to function_call_output format
        tool_output = kwargs["input"][2]
        assert tool_output["type"] == "function_call_output"
        assert tool_output["output"] == "Sunny, 72°F"
        assert tool_output["call_id"] == "call_123"

    def test_from_responses_reads_tool_call_id(self):
        """Test that tool_call_id is read from Responses input items (function_call_output format)."""
        inputs: list[openai.types.responses.ResponseInputItemParam] = [
            openai.types.responses.EasyInputMessageParam(type="message", role="user", content="What's the weather?"),
            openai.types.responses.response_input_item_param.FunctionCallOutput(
                type="function_call_output", call_id="call_456", output="Sunny, 72°F"
            ),
        ]
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming(
            model="gpt-4o",
            input=inputs,
        )
        request = request_lib.LLMRequest.from_responses(kwargs)

        tool_msg = request.messages[1]
        assert tool_msg.get("role") == "tool"
        assert tool_msg.get("content") == "Sunny, 72°F"
        assert tool_msg.get("tool_call_id") == "call_456"

    def test_from_responses_tool_missing_id_strict(self):
        """Test that missing call_id raises error in strict mode."""
        # Note: This tests a malformed input - function_call_output requires call_id
        # We construct a dict to simulate a malformed payload
        inputs: list[dict[str, str]] = [
            {"type": "function_call_output", "output": "Result"},  # Missing required call_id
        ]
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming(
            model="gpt-4o",
            input=inputs,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match=r"Tool .*missing required.*call_id"):
            request_lib.LLMRequest.from_responses(kwargs, strict=True)

    def test_from_responses_tool_missing_id_non_strict(self):
        """Test that missing call_id logs warning in non-strict mode."""
        # Note: This tests a malformed input - function_call_output requires call_id
        inputs: list[dict[str, str]] = [
            {"type": "function_call_output", "output": "Result"},  # Missing required call_id
        ]
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming(
            model="gpt-4o",
            input=inputs,  # type: ignore[arg-type]
        )
        request = request_lib.LLMRequest.from_responses(kwargs, strict=False)

        # Should still parse the message but without tool_call_id
        assert len(request.messages) == 1
        assert request.messages[0].get("role") == "tool"
        assert request.messages[0].get("content") == "Result"
        assert "tool_call_id" not in request.messages[0]

    def test_to_responses_includes_tool_calls(self):
        """Test that ToolCallMessage is converted to function_call format."""
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="What's the weather?"),
                request_lib.AssistantMessage(role="assistant", content="Let me check..."),
                request_lib.ToolCallMessage(call_id="call_123", name="get_weather", arguments='{"location":"SF"}'),
            ],
        )
        kwargs = request.to_responses_kwargs()

        # ToolCallMessage should be converted to function_call format
        tool_call = kwargs["input"][2]
        assert tool_call["type"] == "function_call"
        assert tool_call["call_id"] == "call_123"
        assert tool_call["name"] == "get_weather"
        assert tool_call["arguments"] == '{"location":"SF"}'

    def test_from_responses_reads_tool_calls(self):
        """Test that function_call items are read as ToolCallMessage."""
        inputs: list[openai.types.responses.ResponseInputItemParam] = [
            openai.types.responses.EasyInputMessageParam(type="message", role="user", content="What's the weather?"),
            openai.types.responses.response_function_tool_call_param.ResponseFunctionToolCallParam(
                type="function_call",
                call_id="call_456",
                name="get_weather",
                arguments='{"location":"NYC"}',
            ),
        ]
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming(
            model="gpt-4o",
            input=inputs,
        )
        request = request_lib.LLMRequest.from_responses(kwargs)

        tool_call_msg = request.messages[1]
        assert "call_id" in tool_call_msg
        assert tool_call_msg["call_id"] == "call_456"  # type: ignore[typeddict-item]
        assert tool_call_msg["name"] == "get_weather"  # type: ignore[typeddict-item]
        assert tool_call_msg["arguments"] == '{"location":"NYC"}'  # type: ignore[typeddict-item]

    def test_responses_roundtrip_preserves_tool_call_id(self):
        """Test that tool_call_id is preserved in roundtrip conversion (via function_call_output)."""
        original = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="Check weather"),
                request_lib.ToolCallResponseMessage(role="tool", content="Sunny", tool_call_id="call_789"),
            ],
        )
        # Convert to Responses format (should use function_call_output)
        responses_kwargs = original.to_responses_kwargs()

        # Verify it uses function_call_output format
        assert responses_kwargs["input"][1]["type"] == "function_call_output"
        assert responses_kwargs["input"][1]["call_id"] == "call_789"

        # Convert back and verify tool_call_id is preserved
        request2 = request_lib.LLMRequest.from_responses(
            cast(openai.types.responses.response_create_params.ResponseCreateParamsBase, responses_kwargs)
        )

        tool_msg = cast(request_lib.ToolCallResponseMessage, request2.messages[1])
        assert tool_msg["tool_call_id"] == "call_789"

    def test_from_responses_unsupported_tool_type_strict(self):
        """Test that unsupported tool type raises error in strict mode."""
        # Create a tool with unsupported type (not "function")
        unsupported_tool: dict[str, str] = {"type": "browser_tool", "name": "search_web"}

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [unsupported_tool],  # type: ignore[list-item]
        }

        with pytest.raises(ValueError, match=r"Unsupported tool type for conversion"):
            request_lib.LLMRequest.from_responses(kwargs, strict=True)

    def test_from_responses_unsupported_tool_type_non_strict(self):
        """Test that unsupported tool type is skipped in non-strict mode."""
        # Mix of supported and unsupported tools
        supported_tool: openai.types.responses.FunctionToolParam = openai.types.responses.FunctionToolParam(
            type="function",
            name="valid_tool",
            description="A valid function tool",
            parameters={"type": "object", "properties": {}},
            strict=None,
        )
        unsupported_tool: dict[str, str] = {"type": "browser_tool", "name": "search_web"}

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [supported_tool, unsupported_tool],  # type: ignore[list-item]
        }

        # Should not raise, but only include the supported tool
        request = request_lib.LLMRequest.from_responses(kwargs, strict=False)

        assert request.tools is not None
        assert len(request.tools) == 1
        assert request.tools[0]["name"] == "valid_tool"

    def test_from_responses_all_unsupported_tools_non_strict(self):
        """Test that tools is None when all tools are unsupported in non-strict mode."""
        # Only unsupported tools
        unsupported_tool1: dict[str, str] = {"type": "browser_tool", "name": "search_web"}
        unsupported_tool2: dict[str, str] = {"type": "computer_tool", "name": "run_code"}

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [unsupported_tool1, unsupported_tool2],  # type: ignore[list-item]
        }

        # Should not raise, but tools should be None since no valid tools were converted
        request = request_lib.LLMRequest.from_responses(kwargs, strict=False)

        assert request.tools is None

    def test_to_responses_specific_tool_choice_flat_format(self):
        """Test that specific tool choice uses flat format for Responses API."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "search"},
        )
        kwargs = request.to_responses_kwargs()

        # Responses API should use flat format: {"type": "function", "name": "search"}
        assert kwargs["tool_choice"] == {"type": "function", "name": "search"}

    def test_to_chat_completion_specific_tool_choice_nested_format(self):
        """Test that specific tool choice uses nested format for Chat Completions API."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "search",
                    "description": "Search the web",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "search"},
        )
        kwargs = request.to_chat_completion_kwargs()

        # Chat Completions API should use nested format: {"type": "function", "function": {"name": "search"}}
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "search"}}


class TestLLMRequestMessagesConversion:
    """Tests for Claude Messages API conversion."""

    def test_to_messages_minimal(self):
        """Test minimal conversion to Messages format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = request.to_messages_kwargs()

        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["max_tokens"] == 1024  # Default required by Claude

    def test_to_messages_all_params(self):
        """Test conversion with all common parameters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
            temperature=1.4,  # Will be converted to 0.7
            top_p=0.9,
            top_k=40,
            stream=True,
            tools=[
                {
                    "name": "test",
                    "description": "A test function",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice="auto",  # Internal format
            metadata={"user_id": "123"},
            service_tier="auto",
            stop_sequences=["STOP", "END"],
        )
        kwargs = request.to_messages_kwargs()

        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.7  # Converted from 1.4
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40
        assert kwargs["stream"] is True
        # Tools are converted to Anthropic format (explicit type: "custom")
        assert kwargs["tools"] == [
            {
                "type": "custom",
                "name": "test",
                "description": "A test function",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        assert kwargs["tool_choice"] == {"type": "auto"}
        assert kwargs["metadata"] == {"user_id": "123"}
        assert kwargs["service_tier"] == "auto"
        assert kwargs["stop_sequences"] == ["STOP", "END"]

    def test_to_messages_temperature_conversion(self):
        """Test temperature conversion from OpenAI (0-2) to Claude (0-1) range."""
        test_cases = [
            (0.0, 0.0),
            (1.0, 0.5),
            (2.0, 1.0),
            (0.5, 0.25),
            (1.5, 0.75),
        ]
        for openai_temp, claude_temp in test_cases:
            request = request_lib.LLMRequest(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=openai_temp,
            )
            kwargs = request.to_messages_kwargs()
            assert kwargs["temperature"] == claude_temp

    def test_to_messages_with_system_prompt(self):
        """Test system prompt is mapped to system parameter."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful assistant.",
        )
        kwargs = request.to_messages_kwargs()

        assert kwargs["system"] == "You are a helpful assistant."

    def test_to_messages_excludes_invalid_service_tier(self):
        """Test that non-Claude service tiers are excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="flex",  # Not supported by Claude
        )
        kwargs = request.to_messages_kwargs(strict=False)

        assert "service_tier" not in kwargs

    def test_to_messages_filters_system_messages(self):
        """Test that system/developer messages are filtered out."""
        # Note: system messages are not valid in our Message type but we want to test filtering
        # so we pass them as-is (they'll be filtered by to_messages_kwargs)
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="Hello"),
                request_lib.AssistantMessage(role="assistant", content="Hi"),
            ],
            system_prompt="System message",
        )
        kwargs = request.to_messages_kwargs(strict=False)

        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "user"
        assert kwargs["messages"][1]["role"] == "assistant"

    def test_to_messages_maps_tool_to_user(self):
        """Test that tool/function roles are mapped to user."""
        messages: list[request_lib.Message] = [
            request_lib.UserMessage(role="user", content="What's the weather?"),
            request_lib.AssistantMessage(role="assistant", content="Let me check..."),
            request_lib.ToolCallResponseMessage(role="tool", content="Sunny, 72°F", tool_call_id="test"),
        ]
        request = request_lib.LLMRequest(
            messages=messages,
        )
        kwargs = request.to_messages_kwargs()

        assert kwargs["messages"][2]["role"] == "user"

    def test_from_messages_minimal(self):
        """Test parsing minimal Messages request."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = request_lib.LLMRequest.from_messages(kwargs)

        assert list(request.messages) == [{"role": "user", "content": "Hello"}]
        assert request.max_output_tokens == 100

    def test_from_messages_all_params(self):
        """Test parsing Messages with all parameters."""
        tool: anthropic.types.ToolParam = {
            "type": "custom",
            "name": "test",
            "input_schema": {"type": "object", "properties": {}},
        }
        tool_choice: anthropic.types.ToolChoiceAutoParam = {"type": "auto"}
        msg: anthropic.types.MessageParam = {"role": "user", "content": "Hello"}
        metadata: anthropic.types.MetadataParam = {"user_id": "123"}

        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [msg],
            "max_tokens": 100,
            "temperature": 0.7,  # Will be converted to 1.4
            "top_p": 0.9,
            "top_k": 40,
            "stream": True,
            "tools": [tool],
            "tool_choice": tool_choice,
            "metadata": metadata,
            "service_tier": "auto",
            "stop_sequences": ["STOP"],
            "system": "You are helpful.",
        }
        request = request_lib.LLMRequest.from_messages(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 1.4  # Converted from 0.7
        assert request.top_p == 0.9
        assert request.top_k == 40
        assert request.stream is True
        # Tools are converted to internal format (name, description, input_schema)
        assert request.tools == [
            {"name": "test", "description": "", "input_schema": {"type": "object", "properties": {}}}
        ]
        assert request.tool_choice == "auto"  # Internal format
        assert request.metadata == {"user_id": "123"}
        assert request.service_tier == "auto"
        assert request.stop_sequences == ["STOP"]
        assert request.system_prompt == "You are helpful."

    def test_from_messages_temperature_conversion(self):
        """Test temperature conversion from Claude (0-1) to OpenAI (0-2) range."""
        test_cases = [
            (0.0, 0.0),
            (0.5, 1.0),
            (1.0, 2.0),
            (0.25, 0.5),
            (0.75, 1.5),
        ]
        for claude_temp, openai_temp in test_cases:
            kwargs: anthropic.types.MessageCreateParams = {
                "model": "claude-sonnet-4-5-20250929",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "temperature": claude_temp,
            }
            request = request_lib.LLMRequest.from_messages(kwargs)
            assert request.temperature == openai_temp

    def test_to_messages_strict_alternation_error(self):
        """Test that non-alternating messages raise error in strict mode."""
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="Hello"),
                request_lib.UserMessage(role="user", content="How are you?"),
            ],
        )
        with pytest.raises(ValueError, match="Messages must alternate"):
            request.to_messages_kwargs(strict=True)

    def test_to_messages_non_strict_drops_duplicates(self):
        """Test that non-alternating messages are dropped in non-strict mode."""
        request = request_lib.LLMRequest(
            messages=[
                request_lib.UserMessage(role="user", content="Hello"),
                request_lib.UserMessage(role="user", content="How are you?"),
                request_lib.AssistantMessage(role="assistant", content="I'm fine"),
                request_lib.AssistantMessage(role="assistant", content="Thanks"),
                request_lib.UserMessage(role="user", content="Great"),
            ],
        )
        kwargs = request.to_messages_kwargs(strict=False)

        # Should keep first of each consecutive group
        assert len(kwargs["messages"]) == 3
        assert kwargs["messages"][0] == {"role": "user", "content": "Hello"}
        assert kwargs["messages"][1] == {"role": "assistant", "content": "I'm fine"}
        assert kwargs["messages"][2] == {"role": "user", "content": "Great"}


class TestLLMRequestCrossAPIConversion:
    """Tests for converting between different APIs."""

    def test_chat_to_responses_to_chat(self):
        """Test Chat -> Responses -> Chat roundtrip."""
        original_chat: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 100,
            "temperature": 0.7,
        }
        request = request_lib.LLMRequest.from_chat_completion(original_chat)
        responses_kwargs = request.to_responses_kwargs()
        request2 = request_lib.LLMRequest.from_responses(
            cast(openai.types.responses.response_create_params.ResponseCreateParamsBase, responses_kwargs)
        )
        final_chat = request2.to_chat_completion_kwargs()

        # Note: model will be dropped in the conversion.
        assert "model" not in final_chat
        assert final_chat["max_completion_tokens"] == original_chat["max_completion_tokens"]
        assert final_chat["temperature"] == original_chat["temperature"]

    def test_chat_to_messages_temperature_conversion(self):
        """Test Chat -> Messages converts temperature correctly."""
        chat_params: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 1.0,
        }
        request = request_lib.LLMRequest.from_chat_completion(chat_params)
        claude_kwargs = request.to_messages_kwargs()

        assert claude_kwargs["temperature"] == 0.5  # 1.0 / 2

    def test_messages_to_chat_temperature_conversion(self):
        """Test Messages -> Chat converts temperature correctly."""
        messages_params: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5,
        }
        request = request_lib.LLMRequest.from_messages(messages_params)
        chat_kwargs = request.to_chat_completion_kwargs()

        assert chat_kwargs["temperature"] == 1.0  # 0.5 * 2

    def test_all_apis_preserve_core_params(self):
        """Test that core parameters are preserved across all conversions."""
        # Create a request with all common parameters
        original = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
            temperature=1.0,
            top_p=0.9,
            stream=True,
            metadata={"user_id": "123"},
        )

        # Convert to all formats
        chat_kwargs = original.to_chat_completion_kwargs()
        responses_kwargs = original.to_responses_kwargs()
        messages_kwargs = original.to_messages_kwargs()

        # Verify core params are present in all
        assert chat_kwargs["max_completion_tokens"] == 100
        assert responses_kwargs["max_output_tokens"] == 100
        assert messages_kwargs["max_tokens"] == 100

        assert chat_kwargs["temperature"] == 1.0
        assert responses_kwargs["temperature"] == 1.0
        assert messages_kwargs["temperature"] == 0.5  # Converted

        assert chat_kwargs["top_p"] == 0.9
        assert responses_kwargs["top_p"] == 0.9
        assert messages_kwargs["top_p"] == 0.9

        assert chat_kwargs["stream"] is True
        assert responses_kwargs["stream"] is True
        assert messages_kwargs["stream"] is True

        assert chat_kwargs["metadata"] == {"user_id": "123"}
        assert responses_kwargs["metadata"] == {"user_id": "123"}
        assert messages_kwargs["metadata"] == {"user_id": "123"}
