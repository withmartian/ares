"""Comprehensive edge case tests for LLMRequest conversions across all APIs.

This test suite addresses Issue #64 by providing exhaustive coverage of:
- Tool/function calling edge cases
- Message content edge cases (unicode, special chars, empty/null)
- Complete roundtrip conversions between all API pairs
- Error handling and validation (strict/non-strict modes)
- Parameter boundary conditions
- API-specific conversion edge cases

Test categories are organized to systematically cover the ~200+ edge cases
identified in the test coverage gap analysis.
"""

import pytest

from ares.llms import anthropic_converter
from ares.llms import openai_chat_converter
from ares.llms import openai_responses_converter
from ares.llms import request as request_lib


class TestToolEdgeCases:
    """Test edge cases for tool definitions and tool choice conversions."""

    def test_empty_tools_list(self):
        """Test that empty tools list [] is handled differently from None."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],  # Empty list, not None
        )

        # All APIs should handle empty list gracefully
        chat_kwargs = openai_chat_converter.to_external(request)
        responses_kwargs = openai_responses_converter.to_external(request)
        messages_kwargs = anthropic_converter.to_external(request)

        # Empty list should be omitted or passed as is (API dependent)
        assert chat_kwargs.get("tools", []) == []
        assert responses_kwargs.get("tools", []) == []
        assert messages_kwargs.get("tools", []) == []

    def test_tool_with_empty_description(self):
        """Test tool with empty string description."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "test_tool",
                    "description": "",  # Empty description
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        # Should handle empty description gracefully
        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tools"][0]["function"]["description"] == ""

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["tools"][0]["description"] == ""

    def test_tool_with_minimal_schema(self):
        """Test tool with minimal input_schema (no properties)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "simple_tool",
                    "description": "A simple tool",
                    "input_schema": {"type": "object"},  # No properties field
                }
            ],
        )

        # Should preserve minimal schema
        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tools"][0]["function"]["parameters"]["type"] == "object"

    def test_tool_with_complex_nested_schema(self):
        """Test tool with deeply nested JSON schema."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "complex_tool",
                    "description": "Tool with nested schema",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "user": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "address": {
                                        "type": "object",
                                        "properties": {
                                            "street": {"type": "string"},
                                            "city": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            ],
        )

        # All converters should preserve nested structure
        chat_kwargs = openai_chat_converter.to_external(request)
        schema = chat_kwargs["tools"][0]["function"]["parameters"]
        assert schema["properties"]["user"]["properties"]["address"]["properties"]["city"]["type"] == "string"

    def test_tool_choice_any_maps_to_required(self):
        """Test that tool_choice 'any' maps to OpenAI's 'required'."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "test", "description": "Test", "input_schema": {"type": "object", "properties": {}}}],
            tool_choice="any",  # Model must use at least one tool
        )

        # Chat Completions should map to "required"
        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tool_choice"] == "required"

        # Responses should map to "required"
        responses_kwargs = openai_responses_converter.to_external(request)
        assert responses_kwargs["tool_choice"] == "required"

        # Messages should map to {"type": "any"}
        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["tool_choice"] == {"type": "any"}

    def test_tool_choice_none_disables_tools(self):
        """Test that tool_choice 'none' prevents tool use."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "test", "description": "Test", "input_schema": {"type": "object", "properties": {}}}],
            tool_choice="none",  # Model must not use tools
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tool_choice"] == "none"

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["tool_choice"] == {"type": "none"}

    def test_tool_with_special_characters_in_name(self):
        """Test tool names with underscores, hyphens, numbers."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "get_user_data_v2",  # Underscores, numbers
                    "description": "Get user data",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tools"][0]["function"]["name"] == "get_user_data_v2"

    def test_tool_schema_with_required_field(self):
        """Test tool schema with 'required' field specifying mandatory parameters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "create_user",
                    "description": "Create a user",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "email"],  # Required fields
                    },
                }
            ],
        )

        # Should preserve required field
        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["tools"][0]["function"]["parameters"]["required"] == ["name", "email"]


class TestMessageContentEdgeCases:
    """Test edge cases for message content (empty, unicode, special chars)."""

    def test_message_with_empty_string_content(self):
        """Test message with empty string content."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": ""}],  # Empty content
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == ""

    def test_message_with_only_whitespace(self):
        """Test message with only whitespace characters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "   \t\n  "}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "   \t\n  "

    def test_message_with_newlines(self):
        """Test message with newline characters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Line 1\nLine 2\nLine 3"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "Line 1\nLine 2\nLine 3"

    def test_message_with_emoji(self):
        """Test message with emoji characters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello 👋 World 🌍"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "Hello 👋 World 🌍"

    def test_message_with_mathematical_symbols(self):
        """Test message with mathematical symbols and Greek letters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Calculate: ∑(x²) + ∫f(x)dx = π"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "Calculate: ∑(x²) + ∫f(x)dx = π"

    def test_message_with_cjk_characters(self):
        """Test message with Chinese, Japanese, Korean characters."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "你好 こんにちは 안녕하세요"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "你好 こんにちは 안녕하세요"

    def test_message_with_rtl_text(self):
        """Test message with right-to-left text (Arabic, Hebrew)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "مرحبا שלום"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "مرحبا שלום"

    def test_message_with_combined_emoji(self):
        """Test message with combined emoji (skin tones, zero-width joiners)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "👨‍👩‍👧‍👦 👋🏽"}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["messages"][0]["content"] == "👨‍👩‍👧‍👦 👋🏽"

    def test_very_long_message_content(self):
        """Test message with very long content (thousands of characters)."""
        long_content = "A" * 10000  # 10k characters
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": long_content}],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert len(chat_kwargs["messages"][0]["content"]) == 10000


class TestRoundtripConversions:
    """Test complete roundtrip conversions between all API pairs."""

    def test_messages_to_responses_to_messages_roundtrip(self):
        """Test Messages → Responses → Messages preserves all parameters."""
        original = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
            temperature=1.0,  # Will be converted
            top_p=0.9,
            stream=True,
            metadata={"user_id": "123"},
        )

        # Convert to Responses
        responses_kwargs = openai_responses_converter.to_external(original)

        # Convert back to internal
        request2 = openai_responses_converter.from_external(responses_kwargs)  # type: ignore[arg-type]

        # Convert to Messages
        messages_kwargs = anthropic_converter.to_external(request2)

        # Convert back to internal
        request3 = anthropic_converter.from_external(messages_kwargs)  # type: ignore[arg-type]

        # Verify parameters preserved (temperature will be scaled)
        assert request3.max_output_tokens == 100
        assert request3.top_p == 0.9
        assert request3.stream is True
        assert request3.metadata == {"user_id": "123"}

    def test_chat_to_messages_to_chat_with_tools_roundtrip(self):
        """Test Chat → Messages → Chat preserves tool definitions."""
        original = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
            tool_choice="auto",
        )

        # Chat → internal (already internal)
        messages_kwargs = anthropic_converter.to_external(original)
        request2 = anthropic_converter.from_external(messages_kwargs)  # type: ignore[arg-type]
        chat_kwargs = openai_chat_converter.to_external(request2)

        # Verify tool preserved
        assert len(chat_kwargs["tools"]) == 1
        assert chat_kwargs["tools"][0]["function"]["name"] == "get_weather"
        assert "location" in chat_kwargs["tools"][0]["function"]["parameters"]["properties"]
        assert chat_kwargs["tool_choice"] == "auto"

    def test_responses_to_chat_to_responses_roundtrip(self):
        """Test Responses → Chat → Responses preserves parameters."""
        original = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Test"}],
            max_output_tokens=200,
            system_prompt="You are helpful.",
        )

        # Responses → Chat → Responses
        responses_kwargs1 = openai_responses_converter.to_external(original)
        request2 = openai_responses_converter.from_external(responses_kwargs1)  # type: ignore[arg-type]
        chat_kwargs = openai_chat_converter.to_external(request2)
        request3 = openai_chat_converter.from_external(chat_kwargs)  # type: ignore[arg-type]
        responses_kwargs2 = openai_responses_converter.to_external(request3)

        # Verify preserved
        assert responses_kwargs2["max_output_tokens"] == 200
        assert responses_kwargs2["instructions"] == "You are helpful."


class TestParameterBoundaries:
    """Test parameter boundary conditions and edge values."""

    def test_temperature_zero(self):
        """Test temperature = 0.0 (greedy decoding)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.0,
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["temperature"] == 0.0

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["temperature"] == 0.0  # 0.0 / 2 = 0.0

    def test_temperature_two_point_zero(self):
        """Test temperature = 2.0 (max OpenAI value)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            temperature=2.0,
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["temperature"] == 2.0

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["temperature"] == 1.0  # Clamped to Claude max

    def test_top_p_zero(self):
        """Test top_p = 0.0 edge case."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_p=0.0,
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["top_p"] == 0.0

    def test_top_p_one(self):
        """Test top_p = 1.0 (no filtering)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_p=1.0,
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["top_p"] == 1.0

    def test_max_output_tokens_one(self):
        """Test max_output_tokens = 1 (minimal)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=1,
        )

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["max_tokens"] == 1

    def test_stop_sequences_at_chat_limit(self):
        """Test stop sequences with exactly 4 items (Chat API limit)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP1", "STOP2", "STOP3", "STOP4"],
        )

        chat_kwargs = openai_chat_converter.to_external(request, strict=False)
        assert len(chat_kwargs["stop"]) == 4

    def test_stop_sequences_exceeds_chat_limit(self):
        """Test stop sequences with 5+ items truncated for Chat API."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP1", "STOP2", "STOP3", "STOP4", "STOP5"],
        )

        chat_kwargs = openai_chat_converter.to_external(request, strict=False)
        assert len(chat_kwargs["stop"]) == 4  # Truncated to 4

    def test_stop_sequences_with_empty_strings(self):
        """Test stop sequences containing empty strings."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["", "STOP", ""],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["stop"] == ["", "STOP", ""]  # Preserved as-is


class TestErrorHandlingStrictMode:
    """Test strict mode error handling and validation."""

    def test_standard_only_service_tier_filtered_by_chat(self):
        """Test that service_tier='standard_only' is filtered by Chat API in strict mode."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            service_tier="standard_only",  # Not supported by Chat API
        )

        # Chat API should error on standard_only in strict mode
        with pytest.raises(ValueError, match="service_tier"):
            openai_chat_converter.to_external(request, strict=True)

    def test_top_k_unsupported_by_chat_strict(self):
        """Test that top_k raises error for Chat API in strict mode."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,  # Claude-specific parameter
        )

        # Should raise in strict mode
        with pytest.raises(ValueError, match="top_k"):
            openai_chat_converter.to_external(request, strict=True)

    def test_stop_sequences_unsupported_by_responses_strict(self):
        """Test that stop_sequences raises error for Responses API in strict mode."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP"],
        )

        # Should raise in strict mode
        with pytest.raises(ValueError, match="stop_sequences"):
            openai_responses_converter.to_external(request, strict=True)

    def test_non_alternating_messages_strict_mode(self):
        """Test that non-alternating messages raise error for Messages API in strict mode."""
        request = request_lib.LLMRequest(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Are you there?"},  # Two user messages in a row
            ],
        )

        # Should raise in strict mode
        with pytest.raises(ValueError, match="alternate"):
            anthropic_converter.to_external(request, strict=True)


class TestErrorHandlingNonStrictMode:
    """Test non-strict mode graceful degradation."""

    def test_top_k_filtered_non_strict(self):
        """Test that top_k is silently filtered in non-strict mode."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )

        # Should not raise, just filter out top_k
        chat_kwargs = openai_chat_converter.to_external(request, strict=False)
        assert "top_k" not in chat_kwargs

    def test_stop_sequences_filtered_non_strict(self):
        """Test that stop_sequences are filtered for Responses API in non-strict mode."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP"],
        )

        # Should not raise, just filter out
        responses_kwargs = openai_responses_converter.to_external(request, strict=False)
        assert "stop_sequences" not in responses_kwargs
        assert "stop" not in responses_kwargs

    def test_non_alternating_messages_dropped_non_strict(self):
        """Test that duplicate role messages are dropped in non-strict mode."""
        request = request_lib.LLMRequest(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "Are you there?"},  # Duplicate user
                {"role": "assistant", "content": "Yes"},
            ],
        )

        # Should drop the second user message
        messages_kwargs = anthropic_converter.to_external(request, strict=False)
        assert len(messages_kwargs["messages"]) == 2  # Only 2 messages (first user + assistant)


class TestAPISpecificConversions:
    """Test API-specific conversion behaviors."""

    def test_responses_string_input_conversion(self):
        """Test Responses API string input converts to user message."""
        import openai.types.responses.response_create_params

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "input": "Hello, world!",  # String input (not list)
        }

        request = openai_responses_converter.from_external(kwargs)
        assert len(request.messages) == 1
        assert request.messages[0]["role"] == "user"
        assert request.messages[0]["content"] == "Hello, world!"

    def test_chat_system_message_extraction(self):
        """Test that Chat API extracts first system message to system_prompt."""
        import openai.types.chat.completion_create_params

        kwargs: openai.types.chat.completion_create_params.CompletionCreateParams = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }

        request = openai_chat_converter.from_external(kwargs)
        assert request.system_prompt == "You are helpful."
        assert len(request.messages) == 1  # System message removed from messages
        assert request.messages[0]["role"] == "user"

    def test_messages_instructions_to_system_prompt(self):
        """Test that Responses instructions field maps to system_prompt."""
        import openai.types.responses.response_create_params

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "input": "Hello",
            "instructions": "Be concise.",
        }

        request = openai_responses_converter.from_external(kwargs)
        assert request.system_prompt == "Be concise."

    def test_anthropic_max_tokens_required_default(self):
        """Test that Anthropic converter defaults max_tokens to 1024."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            # No max_output_tokens specified
        )

        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["max_tokens"] == 1024  # Default


class TestMetadataEdgeCases:
    """Test metadata handling edge cases."""

    def test_metadata_with_null_values(self):
        """Test metadata with null values."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"key1": "value1", "key2": None},  # type: ignore[typeddict-item]
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["metadata"] == {"key1": "value1", "key2": None}

    def test_metadata_with_nested_dicts(self):
        """Test metadata with nested dictionary structures."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"user": {"id": "123", "name": "Test"}},  # type: ignore[typeddict-item]
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["metadata"]["user"]["id"] == "123"  # type: ignore[index, typeddict-item]

    def test_metadata_with_array_values(self):
        """Test metadata with array values."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            metadata={"tags": ["tag1", "tag2", "tag3"]},  # type: ignore[typeddict-item]
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert chat_kwargs["metadata"]["tags"] == ["tag1", "tag2", "tag3"]  # type: ignore[index, typeddict-item]

    def test_empty_metadata_dict(self):
        """Test empty metadata dictionary is omitted from output."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            metadata={},
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        # Empty metadata should be omitted (not included in kwargs)
        assert "metadata" not in chat_kwargs


class TestToolChoiceRoundtrips:
    """Test tool_choice preservation across all API pairs."""

    def test_tool_choice_auto_roundtrip_all_apis(self):
        """Test tool_choice 'auto' preserved across all conversions."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "test", "description": "Test", "input_schema": {"type": "object", "properties": {}}}],
            tool_choice="auto",
        )

        # Chat → Messages → Chat
        messages_kwargs = anthropic_converter.to_external(request)
        assert messages_kwargs["tool_choice"] == {"type": "auto"}

        request2 = anthropic_converter.from_external(messages_kwargs)  # type: ignore[arg-type]
        chat_kwargs = openai_chat_converter.to_external(request2)
        assert chat_kwargs["tool_choice"] == "auto"

    def test_tool_choice_specific_tool_roundtrip(self):
        """Test tool_choice with specific tool name preserved in roundtrip."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"name": "search", "description": "Search", "input_schema": {"type": "object", "properties": {}}}],
            tool_choice={"type": "tool", "name": "search"},
        )

        # Messages → Chat → Messages
        messages_kwargs1 = anthropic_converter.to_external(request)
        request2 = anthropic_converter.from_external(messages_kwargs1)  # type: ignore[arg-type]
        chat_kwargs = openai_chat_converter.to_external(request2)
        request3 = openai_chat_converter.from_external(chat_kwargs)  # type: ignore[arg-type]
        messages_kwargs2 = anthropic_converter.to_external(request3)

        # Verify specific tool preserved
        assert messages_kwargs2["tool_choice"] == {"type": "tool", "name": "search"}


class TestMultipleConsecutiveMessages:
    """Test handling of multiple consecutive messages with same role."""

    def test_multiple_empty_messages(self):
        """Test multiple messages with empty content."""
        request = request_lib.LLMRequest(
            messages=[
                {"role": "user", "content": ""},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "Hello"},
            ],
        )

        chat_kwargs = openai_chat_converter.to_external(request)
        assert len(chat_kwargs["messages"]) == 3
        assert all(msg["content"] == "" or msg["content"] == "Hello" for msg in chat_kwargs["messages"])

    def test_alternating_user_assistant_preserved(self):
        """Test that properly alternating messages are preserved."""
        request = request_lib.LLMRequest(
            messages=[
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
        )

        messages_kwargs = anthropic_converter.to_external(request, strict=True)
        assert len(messages_kwargs["messages"]) == 4  # All preserved
