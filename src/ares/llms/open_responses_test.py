"""Tests for the canonical Open Responses helpers."""

from ares.llms import open_responses
from ares.llms import request as request_lib


def test_make_request_defaults_to_model_stub():
    request = open_responses.make_request([open_responses.user_message("Hello")])

    assert request.model == open_responses.MODEL_STUB
    assert len(open_responses.message_items(request)) == 1
    assert open_responses.message_text(open_responses.message_items(request)[0]) == "Hello"


def test_input_items_wraps_string_input_as_user_message():
    request = open_responses.make_request("Hello")

    messages = open_responses.message_items(request)
    assert len(messages) == 1
    assert messages[0].role.value == "user"
    assert open_responses.message_text(messages[0]) == "Hello"


def test_to_chat_completions_kwargs_maps_instructions_and_messages():
    request = open_responses.make_request(
        [open_responses.user_message("Hi"), open_responses.assistant_message("Hello")],
        model="test-model",
        instructions="Be concise.",
        temperature=0.25,
    )

    kwargs = open_responses.to_chat_completions_kwargs(request)

    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.25
    assert kwargs["messages"][0] == {"role": "system", "content": "Be concise."}
    assert kwargs["messages"][1] == {"role": "user", "content": "Hi"}
    assert kwargs["messages"][2] == {"role": "assistant", "content": "Hello"}


def test_from_legacy_request_preserves_embedded_assistant_tool_calls():
    legacy_request = request_lib.LLMRequest(
        messages=[
            request_lib.UserMessage(role="user", content="What is the weather?"),
            request_lib.AssistantMessage(
                role="assistant",
                content="Let me check.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
                    }
                ],
            ),
        ]
    )

    canonical_request = open_responses.from_legacy_request(legacy_request)
    items = open_responses.request_to_jsonable(canonical_request)["input"]

    assert len(items) == 3
    assert items[1]["type"] == "message"
    assert items[1]["content"] == "Let me check."
    assert items[2]["type"] == "function_call"
    assert items[2]["call_id"] == "call_123"
    assert items[2]["name"] == "get_weather"
    assert items[2]["arguments"] == '{"location":"SF"}'


def test_to_chat_completions_kwargs_flattens_tool_calls_and_strips_tool_strict():
    request = open_responses.make_request(
        [
            open_responses.user_message("What is the weather?"),
            open_responses.assistant_message("Let me check."),
            open_responses.function_call(call_id="call_123", name="get_weather", arguments='{"location":"SF"}'),
        ],
        model="test-model",
        tools=[
            open_responses.function_tool(
                name="get_weather",
                description="Look up weather.",
                parameters={"type": "object", "properties": {}},
                strict=True,
            )
        ],
    )

    kwargs = open_responses.to_chat_completions_kwargs(request)

    assert kwargs["messages"] == [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location":"SF"}'},
                }
            ],
        },
    ]
    assert kwargs["tools"][0]["function"]["name"] == "get_weather"
    assert "strict" not in kwargs["tools"][0]["function"]


def test_ensure_request_rejects_legacy_requests():
    legacy_request = request_lib.LLMRequest(messages=[request_lib.UserMessage(role="user", content="Hello")])

    try:
        open_responses.ensure_request(legacy_request)
    except TypeError as exc:
        assert "canonical Open Responses requests" in str(exc)
    else:
        raise AssertionError("Expected ensure_request to reject legacy LLMRequest inputs")


def test_ensure_request_accepts_canonical_request():
    request = open_responses.make_request([open_responses.user_message("Hello")])
    result = open_responses.ensure_request(request)
    assert result is request


def test_from_legacy_request_converts_tool_role_messages():
    legacy = request_lib.LLMRequest(
        messages=[
            request_lib.UserMessage(role="user", content="Search for cats"),
            request_lib.ToolCallResponseMessage(role="tool", content="Found 3 cats", tool_call_id="call_42"),
        ]
    )

    canonical = open_responses.from_legacy_request(legacy)
    items = open_responses.request_to_jsonable(canonical)["input"]

    assert len(items) == 2
    assert items[0]["type"] == "message"
    assert items[0]["role"] == "user"
    assert items[1]["type"] == "function_call_output"
    assert items[1]["call_id"] == "call_42"
    assert items[1]["output"] == "Found 3 cats"


def test_to_legacy_request_roundtrips_function_call_items():
    canonical = open_responses.make_request(
        [
            open_responses.user_message("What is the weather?"),
            open_responses.assistant_message("Let me check."),
            open_responses.function_call(call_id="call_1", name="get_weather", arguments='{"city":"SF"}'),
            open_responses.function_call_output(call_id="call_1", output="Sunny"),
        ],
        instructions="Be helpful.",
        temperature=0.5,
    )

    legacy = open_responses.to_legacy_request(canonical)
    assert legacy.system_prompt == "Be helpful."
    assert legacy.temperature == 0.5
    assert len(legacy.messages) == 4
    assert legacy.messages[0].get("role") == "user"
    assert legacy.messages[1].get("role") == "assistant"
    assert legacy.messages[2].get("call_id") == "call_1"
    assert legacy.messages[2].get("name") == "get_weather"
    assert legacy.messages[3].get("role") == "tool"
    assert legacy.messages[3].get("content") == "Sunny"


def test_to_chat_completions_kwargs_strips_model_stub(caplog):
    request = open_responses.make_request([open_responses.user_message("Hello")])
    assert request.model == open_responses.MODEL_STUB

    with caplog.at_level("WARNING"):
        kwargs = open_responses.to_chat_completions_kwargs(request)

    assert "model" not in kwargs
    assert "MODEL_STUB" in caplog.text


def test_to_chat_completions_kwargs_preserves_real_model():
    request = open_responses.make_request([open_responses.user_message("Hello")], model="gpt-4o")

    kwargs = open_responses.to_chat_completions_kwargs(request)

    assert kwargs["model"] == "gpt-4o"
