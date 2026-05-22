"""Library for tracking LLM costs."""

import decimal
import functools
import re
from typing import Protocol

import frozendict
import httpx
from openai.types.chat import chat_completion as chat_completion_types
import pydantic

from ares import config


_RACING_MODEL_PATTERN = re.compile(r"^(?P<base_model>.+):racing-(?P<num_racers>[1-9][0-9]*)@(?P<multiplier>[1-9][0-9]*)$")


class Usage(Protocol):
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int


class ModelPricing(pydantic.BaseModel, frozen=True):
    """Model pricing information."""

    prompt: decimal.Decimal | None
    completion: decimal.Decimal | None
    image: decimal.Decimal | None
    request: decimal.Decimal | None
    web_search: decimal.Decimal | None
    internal_reasoning: decimal.Decimal | None


class ModelCost(pydantic.BaseModel, frozen=True):
    """Complete model cost information."""

    id: str
    pricing: ModelPricing


class ModelsResponse(pydantic.BaseModel, frozen=True):
    """Response from the models API endpoint."""

    data: list[ModelCost]


@functools.lru_cache(maxsize=1)
def martian_cost_list(
    client: httpx.Client | None = None,
) -> frozendict.frozendict[str, ModelCost]:
    """Get the cost of LLM calls for Martian.

    Note that we decidedly make this non-async to make it easier to work with.
    This is because it is cached after the first call, so although it has
    some setup time, it doesn't do any I/O after the first call.

    Args:
        client: Optional httpx client for making requests. If None, creates a new client.

    Returns:
        A frozendict mapping model_id to ModelCost objects with pricing information.
    """
    if client is None:
        client = httpx.Client()

    with client:
        models_response = client.get(f"{config.CONFIG.chat_completion_api_base_url}/models")
        models_response.raise_for_status()

    # Parse the response using Pydantic
    response_data = models_response.json()
    models_response_obj = ModelsResponse.model_validate(response_data)

    # Create the frozendict mapping
    cost_mapping = {model.id: model for model in models_response_obj.data}
    return frozendict.frozendict(cost_mapping)


def get_llm_cost(
    model_id: str,
    completion: chat_completion_types.ChatCompletion,
    *,
    cost_mapping: frozendict.frozendict[str, ModelCost],
) -> decimal.Decimal:
    """Get the cost of an LLM call."""
    usage = completion.usage
    if usage is None:
        raise ValueError("Cannot compute cost of a completion with no usage.")

    return get_usage_cost(model_id, usage, cost_mapping=cost_mapping)


def get_usage_cost(
    model_id: str,
    usage: Usage,
    *,
    cost_mapping: frozendict.frozendict[str, ModelCost],
) -> decimal.Decimal:
    """Get model cost from token usage.

    Racing model aliases use `<base-model>:racing-<m>@<k>` and cost k times the base model call cost.
    """
    cost_multiplier = decimal.Decimal(1)
    priced_model_id = model_id
    racing_match = _RACING_MODEL_PATTERN.match(model_id)
    if racing_match is not None:
        priced_model_id = racing_match.group("base_model")
        cost_multiplier = decimal.Decimal(racing_match.group("multiplier"))

    if priced_model_id not in cost_mapping:
        raise ValueError(f"Model {priced_model_id} not found in cost mapping.")
    model_pricing = cost_mapping[priced_model_id].pricing

    # Note: This doesn't take into account:
    # - completion_tokens_details.reasoning_tokens
    # - prompt_tokens_details.cached_tokens
    # It seems for now that the Martian API doesn't include internal reasoning tokens in cost,
    # and just considers them all output tokens.

    zero = decimal.Decimal("0")
    cached_prompt_tokens = min(usage.cached_prompt_tokens, usage.prompt_tokens)
    uncached_prompt_tokens = usage.prompt_tokens - cached_prompt_tokens

    return cost_multiplier * (
        (model_pricing.request or zero)
        + (model_pricing.prompt or zero) * uncached_prompt_tokens
        + decimal.Decimal("0.1") * (model_pricing.prompt or zero) * cached_prompt_tokens
        + (model_pricing.completion or zero) * usage.completion_tokens
    )
