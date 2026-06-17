"""An LLM Client for Chat Completions-compatible models."""

import dataclasses
import logging
import threading

import httpx
import openai
import openai.types.chat.chat_completion
import tenacity

from ares import config
from ares.llms import accounting
from ares.llms import llm_clients
from ares.llms import openai_chat_converter
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


_thread_local = threading.local()


def _get_llm_client(base_url: str, api_key: str) -> openai.AsyncClient:
    """Return a per-thread cached AsyncClient.

    httpx.AsyncClient is bound to the event loop of the thread that created it,
    so a global lru_cache deadlocks when workers run in separate threads.
    """
    key = (base_url, api_key)
    clients: dict[tuple[str, str], openai.AsyncClient] = getattr(_thread_local, "clients", {})
    if key not in clients:
        clients[key] = openai.AsyncClient(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            http_client=httpx.AsyncClient(timeout=60.0, http2=True),
        )
        _thread_local.clients = clients
    return clients[key]


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(min=1, max=60) + tenacity.wait_random(min=0, max=1),
    before_sleep=tenacity.before_sleep_log(_LOGGER, logging.INFO),
)
async def _query_llm_with_retry(
    llm_client: openai.AsyncClient, model: str, req: request.LLMRequest
) -> openai.types.chat.chat_completion.ChatCompletion:
    response = await llm_client.chat.completions.create(
        model=model, **openai_chat_converter.ares_request_to_external(req)
    )
    return response


@dataclasses.dataclass(frozen=True)
class ChatCompletionCompatibleLLMClient(llm_clients.LLMClient):
    model: str
    base_url: str = config.CONFIG.chat_completion_api_base_url
    api_key: str = config.CONFIG.chat_completion_api_key

    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse:
        _LOGGER.debug("[%d] Requesting LLM.", id(self))

        # GPT-5 models don't support temperature.
        if self.model.startswith("openai/gpt-5"):
            request = dataclasses.replace(request, temperature=None)

        resp = await _query_llm_with_retry(_get_llm_client(self.base_url, self.api_key), self.model, request)
        _LOGGER.debug("[%d] LLM response received.", id(self))

        # Use the converter to handle both text and tool calls
        return openai_chat_converter.ares_response_from_external(
            resp, model=self.model, cost_mapping=accounting.martian_cost_list()
        )
