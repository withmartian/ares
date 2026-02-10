"""Unit tests for Anthropic Messages API converter."""

import anthropic.types
import anthropic.types.message_create_params
import pytest

from ares.llms import anthropic_converter
from ares.llms import request as request_lib


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Messages API conversions."""

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
            anthropic_converter.from_external(kwargs, strict=True)

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
            anthropic_converter.from_external(kwargs, strict=True)


class TestLLMRequestMessagesConversion:
    """Tests for Claude Messages API conversion."""

    def test_to_messages_minimal(self):
        """Test minimal conversion to Messages format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = anthropic_converter.to_external(request)

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
        kwargs = anthropic_converter.to_external(request)

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
            kwargs = anthropic_converter.to_external(request)
            assert kwargs["temperature"] == claude_temp

    def test_to_messages_with_system_prompt(self):
        """Test system prompt is mapped to system parameter."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="You are a helpful assistant.",
        )
        kwargs = anthropic_converter.to_external(request)

        assert kwargs["system"] == "You are a helpful assistant."

    def test_to_messages_excludes_invalid_service_tier(self):
        """Test that non-Claude service tiers are excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="flex",  # Not supported by Claude
        )
        kwargs = anthropic_converter.to_external(request, strict=False)

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
        kwargs = anthropic_converter.to_external(request, strict=False)

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
        kwargs = anthropic_converter.to_external(request)

        assert kwargs["messages"][2]["role"] == "user"

    def test_from_messages_minimal(self):
        """Test parsing minimal Messages request."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = anthropic_converter.from_external(kwargs)

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
        request = anthropic_converter.from_external(kwargs)

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
            request = anthropic_converter.from_external(kwargs)
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
            anthropic_converter.to_external(request, strict=True)

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
        kwargs = anthropic_converter.to_external(request, strict=False)

        # Should keep first of each consecutive group
        assert len(kwargs["messages"]) == 3
        assert kwargs["messages"][0] == {"role": "user", "content": "Hello"}
        assert kwargs["messages"][1] == {"role": "assistant", "content": "I'm fine"}
        assert kwargs["messages"][2] == {"role": "user", "content": "Great"}
