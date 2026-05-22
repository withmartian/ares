"""An LLM client for OpenAI Responses-compatible models."""

import dataclasses
import logging
import threading
from typing import Any

import httpx
import openai
import tenacity

from ares import config
from ares.llms import accounting
from ares.llms import llm_clients
from ares.llms import openai_responses_converter
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)

_thread_local = threading.local()


def _get_llm_client(base_url: str, api_key: str) -> openai.AsyncClient:
    key = (base_url, api_key)
    clients: dict[tuple[str, str], openai.AsyncClient] = getattr(_thread_local, "responses_clients", {})
    if key not in clients:
        clients[key] = openai.AsyncClient(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=60.0, http2=True),
        )
        _thread_local.responses_clients = clients
    return clients[key]


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60) + tenacity.wait_random(min=0, max=1),
    before_sleep=tenacity.before_sleep_log(_LOGGER, logging.INFO),
)
async def _query_llm_with_retry(
    llm_client: openai.AsyncClient,
    model: str,
    req: request.LLMRequest,
) -> Any:
    kwargs = openai_responses_converter.to_external(req)
    kwargs["reasoning"] = {"effort": "xhigh"}
    return await llm_client.responses.create(model=model, **kwargs)


def _get_output_text(resp: Any) -> str:
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str):
        return output_text

    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _get_usage(resp: Any) -> response.Usage:
    usage = getattr(resp, "usage", None)
    input_tokens_details = getattr(usage, "input_tokens_details", None)
    return response.Usage(
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        generated_tokens=getattr(usage, "output_tokens", 0) or 0,
        cached_prompt_tokens=getattr(input_tokens_details, "cached_tokens", 0) or 0,
    )


def _get_cost(model: str, usage: response.Usage) -> float:
    completion_usage = _CompletionUsage(usage)
    try:
        return float(accounting.get_usage_cost(model, completion_usage, cost_mapping=accounting.martian_cost_list()))
    except Exception:
        _LOGGER.exception("Failed to fetch model cost list; reporting zero cost.")
        return 0.0


@dataclasses.dataclass(frozen=True)
class _CompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int

    def __init__(self, usage: response.Usage) -> None:
        object.__setattr__(self, "prompt_tokens", usage.prompt_tokens)
        object.__setattr__(self, "completion_tokens", usage.generated_tokens)
        object.__setattr__(self, "cached_prompt_tokens", usage.cached_prompt_tokens)


@dataclasses.dataclass(frozen=True)
class OpenAIResponsesCompatibleLLMClient(llm_clients.LLMClient):
    model: str
    base_url: str = config.CONFIG.chat_completion_api_base_url
    api_key: str = config.CONFIG.chat_completion_api_key

    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse:
        _LOGGER.debug("[%d] Requesting LLM with Responses API.", id(self))

        # GPT-5 models don't support temperature.
        if self.model.startswith("openai/gpt-5"):
            request = dataclasses.replace(request, temperature=None)

        resp = await _query_llm_with_retry(_get_llm_client(self.base_url, self.api_key), self.model, request)
        _LOGGER.debug("[%d] LLM response received.", id(self))

        usage = _get_usage(resp)
        return response.LLMResponse(
            data=[response.TextData(content=_get_output_text(resp))],
            cost=_get_cost(self.model, usage),
            usage=usage,
        )
