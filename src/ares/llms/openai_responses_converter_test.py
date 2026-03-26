"""Unit tests for OpenAI Responses API converter."""

import openai.types.responses
import openai.types.responses.response_create_params
import pytest

from ares.llms import open_responses
from ares.llms import openai_responses_converter


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Responses API conversions."""

    def test_from_responses_with_structured_content_parses(self):
        """Test that structured content is passed through."""
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

        # Linguafranca passes structured content through
        request = openai_responses_converter.from_external(kwargs, strict=False)
        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 1


class TestOpenResponsesResponsesConversion:
    """Tests for Responses API conversion using Open Responses requests."""

    def test_to_responses_minimal(self):
        """Test minimal conversion to Responses format."""
        request = open_responses.make_request([open_responses.user_message("Hello")])
        kwargs = openai_responses_converter.to_external(request)

        # Check the essential fields
        assert kwargs["input"][0]["type"] == "message"
        assert kwargs["input"][0]["role"] == "user"
        assert kwargs["input"][0]["content"] == "Hello"
        assert "temperature" not in kwargs

    def test_to_responses_with_params(self):
        """Test conversion with common parameters."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            max_output_tokens=100,
            temperature=0.7,
            top_p=0.9,
            stream=True,
            metadata={"user_id": "123"},
        )
        kwargs = openai_responses_converter.to_external(request)

        assert kwargs["max_output_tokens"] == 100
        assert kwargs["temperature"] == 0.7
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True
        assert kwargs["metadata"] == {"user_id": "123"}

    def test_to_responses_with_instructions(self):
        """Test instructions is preserved."""
        request = open_responses.make_request(
            [open_responses.user_message("Hello")],
            instructions="You are a helpful assistant.",
        )
        kwargs = openai_responses_converter.to_external(request)

        assert kwargs["instructions"] == "You are a helpful assistant."

    def test_from_responses_minimal(self):
        """Test parsing minimal Responses request."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        request = openai_responses_converter.from_external(kwargs)

        # String input should be preserved
        assert request.input == "Hello"

    def test_from_responses_with_params(self):
        """Test parsing Responses with parameters."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "max_output_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "metadata": {"user_id": "123"},
            "instructions": "You are helpful.",
        }
        request = openai_responses_converter.from_external(kwargs)

        assert request.max_output_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.stream is True
        assert request.metadata == {"user_id": "123"}
        assert request.instructions == "You are helpful."

    def test_from_responses_string_input(self):
        """Test parsing Responses with string input."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "input": "Hello, world!",
        }
        request = openai_responses_converter.from_external(kwargs)

        assert request.input == "Hello, world!"

    def test_from_responses_list_input(self):
        """Test parsing Responses with list input."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi!"},
            ],
        }
        request = openai_responses_converter.from_external(kwargs)

        jsonable = open_responses.request_to_jsonable(request)
        assert len(jsonable["input"]) == 2
        assert jsonable["input"][0]["content"] == "Hello"
        assert jsonable["input"][1]["content"] == "Hi!"

    def test_to_responses_includes_function_call(self):
        """Test that function_call items are preserved."""
        request = open_responses.make_request(
            [
                open_responses.user_message("What's the weather?"),
                open_responses.assistant_message("Let me check..."),
                open_responses.function_call(call_id="call_123", name="get_weather", arguments='{"location":"SF"}'),
            ],
        )
        kwargs = openai_responses_converter.to_external(request)

        # function_call should be preserved
        assert kwargs["input"][2]["type"] == "function_call"
        assert kwargs["input"][2]["call_id"] == "call_123"
        assert kwargs["input"][2]["name"] == "get_weather"

    def test_to_responses_includes_function_call_output(self):
        """Test that function_call_output items are preserved."""
        request = open_responses.make_request(
            [
                open_responses.user_message("What's the weather?"),
                open_responses.function_call_output(call_id="call_123", output="Sunny, 72°F"),
            ],
        )
        kwargs = openai_responses_converter.to_external(request)

        assert kwargs["input"][1]["type"] == "function_call_output"
        assert kwargs["input"][1]["output"] == "Sunny, 72°F"
        assert kwargs["input"][1]["call_id"] == "call_123"

    def test_from_responses_reads_function_call(self):
        """Test that function_call items are read correctly."""
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
        request = openai_responses_converter.from_external(kwargs)

        jsonable = open_responses.request_to_jsonable(request)
        assert jsonable["input"][1]["type"] == "function_call"
        assert jsonable["input"][1]["call_id"] == "call_456"
        assert jsonable["input"][1]["name"] == "get_weather"

    def test_from_responses_reads_function_call_output(self):
        """Test that function_call_output items are read correctly."""
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
        request = openai_responses_converter.from_external(kwargs)

        jsonable = open_responses.request_to_jsonable(request)
        assert jsonable["input"][1]["type"] == "function_call_output"
        assert jsonable["input"][1]["call_id"] == "call_456"
        assert jsonable["input"][1]["output"] == "Sunny, 72°F"

    def test_from_responses_tool_missing_id_strict(self):
        """Test that missing call_id raises error in strict mode."""
        inputs: list[dict[str, str]] = [
            {"type": "function_call_output", "output": "Result"},  # Missing required call_id
        ]
        kwargs = openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming(
            model="gpt-4o",
            input=inputs,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match=r"function_call_output.*missing required.*call_id"):
            openai_responses_converter.from_external(kwargs, strict=True)

    def test_from_responses_unsupported_tool_type_strict(self):
        """Test that unsupported tool type raises error in strict mode."""
        unsupported_tool: dict[str, str] = {"type": "browser_tool", "name": "search_web"}

        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
            "tools": [unsupported_tool],  # type: ignore[list-item]
        }

        with pytest.raises(ValueError, match=r"Unsupported tool type for conversion"):
            openai_responses_converter.from_external(kwargs, strict=True)

    def test_from_responses_rejects_unknown_params_in_strict_mode(self):
        """Test that strict mode rejects unhandled responses parameters."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
            "unexpected_flag": True,  # type: ignore[typeddict-item]
        }

        with pytest.raises(ValueError, match=r"unsupported parameters: unexpected_flag"):
            openai_responses_converter.from_external(kwargs, strict=True)

    def test_roundtrip_responses(self):
        """Test that Responses roundtrip preserves data."""
        original: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": [{"type": "message", "role": "user", "content": "Hello"}],
            "max_output_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "metadata": {"user_id": "123"},
            "instructions": "Be helpful.",
        }
        request = openai_responses_converter.from_external(original)
        converted = openai_responses_converter.to_external(request)

        assert converted["max_output_tokens"] == original["max_output_tokens"]
        assert converted["temperature"] == original["temperature"]
        assert converted["top_p"] == original["top_p"]
        assert converted["metadata"] == original["metadata"]
        assert converted["instructions"] == original["instructions"]
