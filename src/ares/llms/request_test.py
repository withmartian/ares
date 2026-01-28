"""Unit tests for request_lib.LLMRequest conversion methods."""

from typing import cast

import anthropic.types
import openai.types.chat
import openai.types.chat.completion_create_params
import openai.types.responses.response_create_params
import openai.types.shared_params

from ares.llms import request as request_lib


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
            tool_choice={"type": "auto"},
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
        # Tools stay in Claude format (no conversion needed)
        assert kwargs["tools"] == [
            {
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
            request_lib.ToolMessage(role="tool", content="Sunny, 72°F", tool_call_id="test"),
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
        assert request.tool_choice == {"type": "auto"}
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
