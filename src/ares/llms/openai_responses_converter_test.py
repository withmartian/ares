"""Unit tests for OpenAI Responses API converter."""

from typing import cast

import openai.types.responses
import openai.types.responses.response_create_params
import pytest

from ares.llms import openai_responses_converter
from ares.llms import request as request_lib


class TestStructuredContentHandling:
    """Tests for handling structured content (list of blocks) in Responses API conversions."""

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
            openai_responses_converter.from_external(kwargs, strict=True)

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
        request = openai_responses_converter.from_external(kwargs, strict=False)
        assert len(request.messages) == 1
        assert request.messages[0].get("content") == ""


class TestLLMRequestResponsesConversion:
    """Tests for Responses API conversion."""

    def test_to_responses_minimal(self):
        """Test minimal conversion to Responses format."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        kwargs = openai_responses_converter.to_external(request)

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
        kwargs = openai_responses_converter.to_external(request)

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
        kwargs = openai_responses_converter.to_external(request)

        assert kwargs["instructions"] == "You are a helpful assistant."

    def test_to_responses_excludes_stop_sequences(self):
        """Test that stop_sequences are not included (not supported)."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["STOP"],
        )
        kwargs = openai_responses_converter.to_external(request, strict=False)

        assert "stop_sequences" not in kwargs
        assert "stop" not in kwargs

    def test_to_responses_excludes_top_k(self):
        """Test that top_k is excluded."""
        request = request_lib.LLMRequest(
            messages=[{"role": "user", "content": "Hello"}],
            top_k=40,
        )
        kwargs = openai_responses_converter.to_external(request, strict=False)

        assert "top_k" not in kwargs

    def test_from_responses_minimal(self):
        """Test parsing minimal Responses request."""
        kwargs: openai.types.responses.response_create_params.ResponseCreateParams = {
            "model": "gpt-4o",
            "input": "Hello",
        }
        request = openai_responses_converter.from_external(kwargs)

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
        request = openai_responses_converter.from_external(kwargs)

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
        request = openai_responses_converter.from_external(kwargs)

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
        request = openai_responses_converter.from_external(kwargs)

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
        kwargs = openai_responses_converter.to_external(request)

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
        request = openai_responses_converter.from_external(kwargs)

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
            openai_responses_converter.from_external(kwargs, strict=True)

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
        request = openai_responses_converter.from_external(kwargs, strict=False)

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
        kwargs = openai_responses_converter.to_external(request)

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
        request = openai_responses_converter.from_external(kwargs)

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
        responses_kwargs = openai_responses_converter.to_external(original)

        # Verify it uses function_call_output format
        assert responses_kwargs["input"][1]["type"] == "function_call_output"
        assert responses_kwargs["input"][1]["call_id"] == "call_789"

        # Convert back and verify tool_call_id is preserved
        request2 = openai_responses_converter.from_external(
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
            openai_responses_converter.from_external(kwargs, strict=True)

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
        request = openai_responses_converter.from_external(kwargs, strict=False)

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
        request = openai_responses_converter.from_external(kwargs, strict=False)

        assert request.tools is None
