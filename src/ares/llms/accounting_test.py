"""Tests for LLM accounting."""

import dataclasses
import decimal

import frozendict

from ares.llms import accounting


@dataclasses.dataclass(frozen=True)
class _Usage:
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0


def _cost_mapping() -> frozendict.frozendict[str, accounting.ModelCost]:
    return frozendict.frozendict(
        {
            "openai/gpt-5.5": accounting.ModelCost(
                id="openai/gpt-5.5",
                pricing=accounting.ModelPricing(
                    prompt=decimal.Decimal("0.01"),
                    completion=decimal.Decimal("0.02"),
                    image=None,
                    request=decimal.Decimal("0.03"),
                    web_search=None,
                    internal_reasoning=None,
                ),
            )
        }
    )


def test_get_usage_cost_uses_base_model_for_racing_alias() -> None:
    cost = accounting.get_usage_cost(
            "openai/gpt-5.5:racing-8@4",
        _Usage(prompt_tokens=10, completion_tokens=20),
        cost_mapping=_cost_mapping(),
    )

    assert cost == decimal.Decimal("2.12")


def test_get_usage_cost_rejects_unknown_racing_base_model() -> None:
    try:
        accounting.get_usage_cost(
            "openai/not-real:racing-8@2",
            _Usage(prompt_tokens=1, completion_tokens=1),
            cost_mapping=_cost_mapping(),
        )
    except ValueError as e:
        assert str(e) == "Model openai/not-real not found in cost mapping."
    else:
        raise AssertionError("Expected ValueError")


def test_get_usage_cost_prices_cached_prompt_tokens_at_ten_percent() -> None:
    cost = accounting.get_usage_cost(
        "openai/gpt-5.5",
        _Usage(prompt_tokens=100, cached_prompt_tokens=60, completion_tokens=10),
        cost_mapping=_cost_mapping(),
    )

    assert cost == decimal.Decimal("0.69")
