"""A queue-mediated LLM client."""

import asyncio
import dataclasses

from ares import async_utils
from ares.llms import llm_clients


@dataclasses.dataclass(frozen=True)
class QueueMediatedLLMClient(llm_clients.LLMClient):
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
        llm_request, future = await llm_client.q.get()
        llm_response = ...
        future.set_result(llm_response)
    ```

    Attributes:
        q: A queue which receives LLM requests. A task _must_ be watching
            this queue for LLM requests to answer them, otherwise
            awaiting __call__ will block forever.
    """

    q: asyncio.Queue[async_utils.ValueAndFuture[llm_clients.LLMRequest, llm_clients.LLMResponse]] = dataclasses.field(
        default_factory=asyncio.Queue
    )

    async def __call__(self, request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
        future = asyncio.Future[llm_clients.LLMResponse]()
        await self.q.put(async_utils.ValueAndFuture(value=request, future=future))
        return await future
