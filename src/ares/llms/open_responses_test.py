"""Tests for the canonical Open Responses helpers."""

from ares.llms import open_responses


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


def test_ensure_request_accepts_canonical_request():
    request = open_responses.make_request([open_responses.user_message("Hello")])
    result = open_responses.ensure_request(request)
    assert result is request


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
