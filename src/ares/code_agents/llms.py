"""Classes for making LLM requests."""

import asyncio
from collections.abc import Iterable
import dataclasses
import functools
import logging
from typing import Protocol

import httpx
import openai
from openai.types.chat import chat_completion as chat_completion_type
from openai.types.chat import chat_completion_message_param
import openai.types.chat.chat_completion
import tenacity

from ares import async_utils
from ares import config
from ares.code_agents import accounting

_LOGGER = logging.getLogger(__name__)

# TODO: Move these to their own module.


@dataclasses.dataclass(frozen=True)
class LLMRequest:
    messages: Iterable[chat_completion_message_param.ChatCompletionMessageParam]
    temperature: float = 1.0


@dataclasses.dataclass(frozen=True)
class LLMResponse:
    chat_completion_response: chat_completion_type.ChatCompletion
    cost: float


# TODO: Move this to its own module.
class LLMClient(Protocol):
    # TODO: expand the request/response model for LLM reqs.
    async def __call__(self, request: LLMRequest) -> LLMResponse: ...


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
    llm_client: openai.AsyncClient, model: str, request: LLMRequest
) -> openai.types.chat.chat_completion.ChatCompletion:
    response = await llm_client.chat.completions.create(
        model=model,
        messages=request.messages,
        temperature=request.temperature,
    )
    return response


@dataclasses.dataclass(frozen=True)
class ChatCompletionCompatibleLLMClient(LLMClient):
    model: str
    base_url: str = config.CONFIG.chat_completion_api_base_url
    api_key: str = config.CONFIG.chat_completion_api_key

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        _LOGGER.debug("[%d] Requesting LLM.", id(self))
        response = await _query_llm_with_retry(_get_llm_client(self.base_url, self.api_key), self.model, request)
        _LOGGER.debug("[%d] LLM response received.", id(self))

        cost = accounting.get_llm_cost(self.model, response, cost_mapping=accounting.martian_cost_list())
        cost = float(cost)

        return LLMResponse(chat_completion_response=response, cost=cost)


@dataclasses.dataclass(frozen=True)
class QueueMediatedLLMClient(LLMClient):
    """An LLM Client that uses a queue to mediate requests and responses.

    This allows us to "intercept" requests from a code agent and send it
    to another task or coroutine to answer the request. This is vital for
    ARES environments, so that agents can be written in a linear form
    but still allow us to expose LLM requests as observations and
    LLM responses as actions.

    Typical usage:

    ```python
        llm_client = llms.QueueMediatedLLMClient(q=asyncio.Queue())
        code_agent = code_agents.MiniSWECodeAgent(container=container, llm_client=llm_client)
        code_agent_task = asyncio.create_task(code_agent.run("print('Hello, world!')"))

        # The code agent will block until the LLM request is answered.
        # The LLM request will be sent to the queue and the code agent will
        # wait for the LLM response.
        llm_request, future = llm_client.q.get()
        llm_resposne = ...
        future.set_result(llm_response)
    ```

    Attributes:
        q: A queue which receives LLM requests. A task _must_ be watching
            this queue for LLM requests to answer them, otherwise
            awaiting __call__ will block forever.
    """

    q: asyncio.Queue[async_utils.ValueAndFuture[LLMRequest, LLMResponse]] = dataclasses.field(
        default_factory=asyncio.Queue
    )

    async def __call__(self, request: LLMRequest) -> LLMResponse:
        future = asyncio.Future[LLMResponse]()
        await self.q.put(async_utils.ValueAndFuture(value=request, future=future))
        return await future
