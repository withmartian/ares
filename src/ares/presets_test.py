"""Tests for built-in preset behavior."""

import pytest

import ares
from ares import presets
from ares import registry
from ares.llms import open_responses
from ares.llms import response


@pytest.mark.asyncio
async def test_make_twenty_questions_preset_uses_open_responses_observations():
    preset_name = "20q-open-responses-test"
    if preset_name in registry._list_presets():
        registry.unregister_preset(preset_name)

    registry.register_preset(preset_name, presets.TwentyQuestionsSpec(objects=("Basketball",)))

    try:
        async with ares.make(f"{preset_name}:0") as env:
            ts = await env.reset()
            assert ts.observation is not None
            initial_observation = open_responses.ensure_request(ts.observation)
            assert open_responses.request_to_jsonable(initial_observation)["input"][0]["role"] == "user"

            lf_response = response.make_response("Is it Basketball?", input_tokens=1, output_tokens=1)
            ts = await env.step(response.InferenceResult(response=lf_response, cost=0.0))

            assert ts.last()
            assert ts.reward == 0.0
            assert ts.observation is not None
            terminal_observation = open_responses.ensure_request(ts.observation)
            assert open_responses.request_to_jsonable(terminal_observation)["input"][0]["role"] == "user"
    finally:
        registry.unregister_preset(preset_name)
