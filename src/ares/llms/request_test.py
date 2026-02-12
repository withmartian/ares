"""Integration tests for cross-API conversions between different LLM request formats."""

from typing import cast

import anthropic.types
import openai.types.chat.completion_create_params
import openai.types.responses.response_create_params

from ares.llms import anthropic_converter
from ares.llms import openai_chat_converter
from ares.llms import openai_responses_converter
from ares.llms import request as request_lib


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
        request = openai_chat_converter.from_external(original_chat)
        responses_kwargs = openai_responses_converter.to_external(request)
        request2 = openai_responses_converter.from_external(
            cast(openai.types.responses.response_create_params.ResponseCreateParamsBase, responses_kwargs)
        )
        final_chat = openai_chat_converter.to_external(request2)

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
        request = openai_chat_converter.from_external(chat_params)
        claude_kwargs = anthropic_converter.to_external(request)

        assert claude_kwargs["temperature"] == 0.5  # 1.0 / 2

    def test_messages_to_chat_temperature_conversion(self):
        """Test Messages -> Chat converts temperature correctly."""
        messages_params: anthropic.types.MessageCreateParams = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.5,
        }
        request = anthropic_converter.from_external(messages_params)
        chat_kwargs = openai_chat_converter.to_external(request)

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
        chat_kwargs = openai_chat_converter.to_external(original)
        responses_kwargs = openai_responses_converter.to_external(original)
        messages_kwargs = anthropic_converter.to_external(original)

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

    def test_responses_tool_choice_flat_format(self):
        """Test that Responses API uses flat tool_choice format."""
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
        kwargs = openai_responses_converter.to_external(request)

        # Responses API uses flat format: {"type": "function", "name": "search"}
        assert kwargs["tool_choice"] == {"type": "function", "name": "search"}

    def test_chat_tool_choice_nested_format(self):
        """Test that Chat Completions API uses nested tool_choice format."""
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
        kwargs = openai_chat_converter.to_external(request)

        # Chat Completions API uses nested format: {"type": "function", "function": {"name": "search"}}
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "search"}}
