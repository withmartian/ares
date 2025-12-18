"""Library for tracking LLM costs."""

import decimal
import functools
import os

import frozendict
import httpx
from openai.types.chat import chat_completion as chat_completion_types
import pydantic

# TODO: Put this in a config, rather than using a raw env var.
_MARTIAN_API_URL = os.getenv("MARTIAN_API_URL", "https://api.withmartian.com")


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
        models_response = client.get(f"{_MARTIAN_API_URL}/v1/models")
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
    if model_id not in cost_mapping:
        raise ValueError(f"Model {model_id} not found in cost mapping.")
    model_pricing = cost_mapping[model_id].pricing

    usage = completion.usage
    if usage is None:
        raise ValueError("Cannot compute cost of a completion with no usage.")

    # Note: This doesn't take into account:
    # - completion_tokens_details.reasoning_tokens
    # - prompt_tokens_details.cached_tokens
    # It seems for now that the Martian API doesn't include internal reasoning tokens in cost,
    # and just considers them all output tokens.

    zero = decimal.Decimal("0")

    return (
        (model_pricing.request or zero)
        + (model_pricing.prompt or zero) * usage.prompt_tokens
        + (model_pricing.completion or zero) * usage.completion_tokens
    )
