"""Unit tests for Anthropic Messages API converter."""

import anthropic.types
import anthropic.types.message_create_params
import pytest

from ares.llms import anthropic_converter
from ares.llms import open_responses


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Messages API conversions."""

    def test_from_messages_with_structured_content_parses(self):
        """Test that structured content in Claude messages is passed through."""
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

        # Linguafranca passes structured content through
        request = anthropic_converter.from_external(kwargs, strict=False)
        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 1

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

        with pytest.raises(ValueError, match=r"contains unsupported parts"):
            anthropic_converter.from_external(kwargs, strict=True)


class TestOpenResponsesMessagesConversion:
    """Tests for Claude Messages API conversion using Open Responses requests."""

    def test_to_messages_minimal(self):
        """Test minimal conversion to Messages format."""
        request = open_responses.make_request([open_responses.user_message("Hello")])
        kwargs = anthropic_converter.to_external(request)

        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["max_tokens"] == 1024  # Default required by Claude

    def test_to_messages_with_params(self):
        """Test conversion with common parameters."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )
        kwargs = anthropic_converter.to_external(request)

        assert kwargs["max_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True

    def test_to_messages_with_instructions(self):
        """Test instructions is mapped to system parameter."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            instructions="You are a helpful assistant.",
        )
        kwargs = anthropic_converter.to_external(request)

        assert kwargs["system"] == "You are a helpful assistant."

    def test_from_messages_minimal(self):
        """Test parsing minimal Messages request."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        request = anthropic_converter.from_external(kwargs)

        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 1
        assert jsonable["input"][0]["content"] == "Hello"
        assert request.max_output_tokens == 100

    def test_from_messages_with_params(self):
        """Test parsing Messages with parameters."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        }
        request = anthropic_converter.from_external(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True

    def test_from_messages_with_system(self):
        """Test parsing Messages with system prompt (preserved as system message)."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "system": "You are helpful.",
        }
        request = anthropic_converter.from_external(kwargs)

        # Linguafranca preserves system as a system message rather than extracting to instructions
        jsonable = open_responses.request_to_jsonable(request)
        assert jsonable["input"][0]["role"] == "system"
        assert jsonable["input"][0]["content"] == "You are helpful."

    def test_from_messages_rejects_unknown_params_in_strict_mode(self):
        """Test that strict mode rejects unhandled Anthropic parameters."""
        kwargs: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "unexpected_flag": True,  # type: ignore[typeddict-item]
        }

        with pytest.raises(ValueError, match=r"unsupported parameters: unexpected_flag"):
            anthropic_converter.from_external(kwargs, strict=True)
