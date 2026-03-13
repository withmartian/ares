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
