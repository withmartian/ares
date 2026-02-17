"""Library for tracking LLM costs."""

from collections.abc import Mapping
import decimal
import functools
from typing import assert_never

import anthropic.types
import frozendict
import httpx
import openai.types.chat.chat_completion
import openai.types.responses
import pydantic

from ares import config


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
    completion: (
        openai.types.chat.chat_completion.ChatCompletion
        | openai.types.responses.Response
        | anthropic.types.Message
    ),
    *,
    cost_mapping: Mapping[str, ModelCost],
) -> decimal.Decimal:
    """Get the cost of an LLM call.

    Args:
        model_id: The model ID to look up in the cost mapping
        completion: The completion object from OpenAI Chat Completions, OpenAI Responses, or Anthropic Messages API
        cost_mapping: Mapping of model IDs to cost information

    Returns:
        The cost in decimal format

    Raises:
        ValueError: If model not in cost mapping or completion has no usage

    Note:
        This doesn't take into account:
        - completion_tokens_details.reasoning_tokens
        - prompt_tokens_details.cached_tokens
        It seems for now that the Martian API doesn't include internal reasoning tokens in cost,
        and just considers them all output tokens.
    """
    if model_id not in cost_mapping:
        raise ValueError(f"Model {model_id} not found in cost mapping.")
    model_pricing = cost_mapping[model_id].pricing

    zero = decimal.Decimal("0")

    # Handle different API response types with different field names
    if isinstance(completion, openai.types.chat.chat_completion.ChatCompletion):
        # Chat Completions API uses prompt_tokens/completion_tokens
        if completion.usage is None:
            raise ValueError("Cannot compute cost of a completion with no usage.")
        return (
            (model_pricing.request or zero)
            + (model_pricing.prompt or zero) * completion.usage.prompt_tokens
            + (model_pricing.completion or zero) * completion.usage.completion_tokens
        )
    elif isinstance(completion, openai.types.responses.Response):
        # Responses API uses input_tokens/output_tokens
        if completion.usage is None:
            raise ValueError("Cannot compute cost of a response with no usage.")
        return (
            (model_pricing.request or zero)
            + (model_pricing.prompt or zero) * completion.usage.input_tokens
            + (model_pricing.completion or zero) * completion.usage.output_tokens
        )
    elif isinstance(completion, anthropic.types.Message):
        # Anthropic Messages API uses input_tokens/output_tokens
        return (
            (model_pricing.request or zero)
            + (model_pricing.prompt or zero) * completion.usage.input_tokens
            + (model_pricing.completion or zero) * completion.usage.output_tokens
        )
    else:
        assert_never(completion)
