"""Tests for LLM response parsing, including tool calls.

This module tests parsing of OpenAI API responses into our internal LLMResponse format.
It uses hand-crafted JSON fixtures representing realistic API responses.
"""

import json
from pathlib import Path

import openai.types.chat
import openai.types.chat.chat_completion
import pytest

from ares.llms import response

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "test_data"


def load_openai_response(filename: str) -> openai.types.chat.chat_completion.ChatCompletion:
    """Load a JSON fixture and convert to OpenAI ChatCompletion object."""
    with open(FIXTURES_DIR / filename) as f:
        fixture_data = json.load(f)
    return openai.types.chat.chat_completion.ChatCompletion(**fixture_data)


class TestTextOnlyResponses:
    """Test parsing responses with only text content."""

    def test_parse_text_only(self):
        """Parse response with simple text content."""
        completion = load_openai_response("openai_text_only.json")

        # Extract data as we would in the LLM client
        message = completion.choices[0].message
        data = []
        if message.content:
            data.append(response.TextData(content=message.content))

        assert len(data) == 1
        assert isinstance(data[0], response.TextData)
        assert data[0].content == "Hello! How can I help you today?"

    def test_parse_empty_response(self):
        """Parse response with empty content."""
        completion = load_openai_response("openai_empty.json")

        message = completion.choices[0].message
        data = []
        if message.content:
            data.append(response.TextData(content=message.content))

        # Empty content should still create TextData
        if not data:
            data.append(response.TextData(content=""))

        assert len(data) == 1
        assert data[0].content == ""


class TestToolCallResponses:
    """Test parsing responses with tool calls."""

    def test_parse_single_tool_call(self):
        """Parse response with one tool call."""
        completion = load_openai_response("openai_tool_call_single.json")

        message = completion.choices[0].message

        # This test currently only validates the fixture structure
        # Once we implement ToolUseData, we'll test actual parsing
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
        assert tool_call.function.name == "bash"
        assert tool_call.id == "call_abc123"

    def test_parse_parallel_tool_calls(self):
        """Parse response with multiple tool calls."""
        completion = load_openai_response("openai_tool_call_parallel.json")

        message = completion.choices[0].message

        assert message.tool_calls is not None
        assert len(message.tool_calls) == 3

        # Verify all tool calls have required fields
        for tool_call in message.tool_calls:
            assert tool_call.id
            assert tool_call.type == "function"
            assert tool_call.function.name == "bash"
            assert tool_call.function.arguments

    def test_parse_mixed_content(self):
        """Parse response with both text and tool calls."""
        completion = load_openai_response("openai_tool_call_mixed.json")

        message = completion.choices[0].message

        # Has both text content and tool calls
        assert message.content == "Let me check the current directory for you."
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

    def test_parse_malformed_arguments(self):
        """Parse response with malformed JSON in tool arguments."""
        completion = load_openai_response("openai_tool_call_malformed.json")

        message = completion.choices[0].message

        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1

        # The arguments string itself is malformed JSON
        # We should handle this gracefully when parsing
        tool_call = message.tool_calls[0]
        assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
        assert tool_call.function.arguments == '{"command": "ls -la", invalid_json_here'

        # Test that we can handle malformed JSON
        try:
            json.loads(tool_call.function.arguments)
            pytest.fail("Expected JSON parsing to fail")
        except json.JSONDecodeError:
            # This is expected - we should handle this in our parser
            pass


class TestUsageInfo:
    """Test extracting usage information from responses."""

    @pytest.mark.parametrize(
        "fixture_name,expected_prompt_tokens,expected_completion_tokens",
        [
            ("openai_text_only.json", 10, 9),
            ("openai_tool_call_single.json", 50, 15),
            ("openai_tool_call_parallel.json", 50, 30),
            ("openai_empty.json", 10, 1),
        ],
    )
    def test_extract_usage_info(self, fixture_name, expected_prompt_tokens, expected_completion_tokens):
        """Verify we correctly extract token usage from all response types."""
        completion = load_openai_response(fixture_name)

        usage = response.Usage(
            prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
            generated_tokens=completion.usage.completion_tokens if completion.usage else 0,
        )

        assert usage.prompt_tokens == expected_prompt_tokens
        assert usage.generated_tokens == expected_completion_tokens
        assert usage.total_tokens == expected_prompt_tokens + expected_completion_tokens


class TestToolUseDataParsing:
    """Test parsing tool calls into ToolUseData."""

    def test_parse_single_tool_use(self):
        """Parse single tool call into ToolUseData."""
        completion = load_openai_response("openai_tool_call_single.json")

        message = completion.choices[0].message
        data = []

        # Parse tool calls into ToolUseData
        if message.tool_calls:
            for tool_call in message.tool_calls:
                assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
                data.append(
                    response.ToolUseData(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                )

        assert len(data) == 1
        assert isinstance(data[0], response.ToolUseData)
        assert data[0].id == "call_abc123"
        assert data[0].name == "bash"
        assert data[0].input == {"command": "ls -la"}

    def test_parse_parallel_tool_uses(self):
        """Parse multiple parallel tool calls into ToolUseData list."""
        completion = load_openai_response("openai_tool_call_parallel.json")

        message = completion.choices[0].message
        data = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
                data.append(
                    response.ToolUseData(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                )

        assert len(data) == 3
        assert all(isinstance(d, response.ToolUseData) for d in data)

        # Verify each tool call
        assert data[0].id == "call_def456"
        assert data[0].name == "bash"
        assert data[0].input == {"command": "pwd"}

        assert data[1].id == "call_ghi789"
        assert data[1].name == "bash"
        assert data[1].input == {"command": "whoami"}

        assert data[2].id == "call_jkl012"
        assert data[2].name == "bash"
        assert data[2].input == {"command": "date"}

    def test_parse_mixed_text_and_tool_use(self):
        """Parse response with both TextData and ToolUseData."""
        completion = load_openai_response("openai_tool_call_mixed.json")

        message = completion.choices[0].message
        data = []

        # Add text content if present
        if message.content:
            data.append(response.TextData(content=message.content))

        # Add tool calls if present
        if message.tool_calls:
            for tool_call in message.tool_calls:
                assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
                data.append(
                    response.ToolUseData(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                )

        assert len(data) == 2
        assert isinstance(data[0], response.TextData)
        assert isinstance(data[1], response.ToolUseData)

        assert data[0].content == "Let me check the current directory for you."
        assert data[1].id == "call_mixed123"
        assert data[1].name == "bash"
        assert data[1].input == {"command": "ls -la"}

    def test_parse_malformed_tool_arguments(self):
        """Handle malformed JSON in tool call arguments gracefully."""
        completion = load_openai_response("openai_tool_call_malformed.json")

        message = completion.choices[0].message
        data = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                assert isinstance(tool_call, openai.types.chat.ChatCompletionMessageFunctionToolCall)
                try:
                    input_dict = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    # Fall back to storing raw string as error field
                    input_dict = {"_raw_arguments": tool_call.function.arguments, "_parse_error": "Invalid JSON"}

                data.append(response.ToolUseData(id=tool_call.id, name=tool_call.function.name, input=input_dict))

        assert len(data) == 1
        assert isinstance(data[0], response.ToolUseData)
        assert data[0].id == "call_bad123"
        assert data[0].name == "bash"
        assert "_parse_error" in data[0].input
        assert data[0].input["_parse_error"] == "Invalid JSON"
        assert "_raw_arguments" in data[0].input
