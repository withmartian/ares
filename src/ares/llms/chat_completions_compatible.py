"""An LLM Client for Chat Completions-compatible models."""

import dataclasses
import functools
import logging

import httpx
import openai
import openai.types.chat.chat_completion
import tenacity

from ares import config
from ares.llms import accounting
from ares.llms import llm_clients

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache
def _get_llm_client(base_url: str, api_key: str) -> openai.AsyncClient:
    return openai.AsyncClient(
        base_url=base_url,
        api_key=api_key,
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=60.0, http2=True),
    )


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60) + tenacity.wait_random(min=0, max=1),
    before_sleep=tenacity.before_sleep_log(_LOGGER, logging.INFO),
)
async def _query_llm_with_retry(
    llm_client: openai.AsyncClient, model: str, request: llm_clients.LLMRequest
) -> openai.types.chat.chat_completion.ChatCompletion:
    response = await llm_client.chat.completions.create(model=model, **request.as_kwargs())
    return response


@dataclasses.dataclass(frozen=True)
class ChatCompletionCompatibleLLMClient(llm_clients.LLMClient):
    model: str
    base_url: str = config.CONFIG.chat_completion_api_base_url
    api_key: str = config.CONFIG.chat_completion_api_key

    async def __call__(self, request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
        _LOGGER.debug("[%d] Requesting LLM.", id(self))

        # GPT-5 models don't support temperature.
        if self.model.startswith("openai/gpt-5"):
            request = dataclasses.replace(request, temperature=None)

        response = await _query_llm_with_retry(_get_llm_client(self.base_url, self.api_key), self.model, request)
        _LOGGER.debug("[%d] LLM response received.", id(self))

        cost = accounting.get_llm_cost(self.model, response, cost_mapping=accounting.martian_cost_list())
        cost = float(cost)

        return llm_clients.LLMResponse(chat_completion_response=response, cost=cost)
