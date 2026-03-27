"""Tests for built-in preset behavior."""

import pytest

from linguafranca import types as lft

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
            assert isinstance(ts.observation, lft.OpenResponsesRequest)
            assert open_responses.request_to_jsonable(ts.observation)["input"][0]["role"] == "user"
            lf_response = response.make_response("Is it Basketball?", input_tokens=1, output_tokens=1)
            ts = await env.step(response.InferenceResult(response=lf_response, cost=0.0))

            assert ts.last()
            assert ts.reward == 0.0
            assert isinstance(ts.observation, lft.OpenResponsesRequest)
            assert open_responses.request_to_jsonable(ts.observation)["input"][0]["role"] == "user"
    finally:
        registry.unregister_preset(preset_name)
