How It Works
============

This page explains the key implementation patterns that enable ARES's RL abstraction. Understanding these details isn't required to use ARES, but can help when debugging, extending the framework, or implementing custom components.

Queue-Mediated Communication
-----------------------------

The most critical implementation pattern in ARES is the **queue-mediated LLM client**. This pattern enables the RL abstraction by intercepting LLM calls from code agents transparently.

The Problem
~~~~~~~~~~~

How do you create an RL environment where:

* Code agents are written in natural, linear code (reason → execute → repeat)
* The environment exposes LLM interactions as observations and actions
* Agents remain unaware of the RL loop

How It Works
~~~~~~~~~~~~

The ``QueueMediatedLLMClient`` implements the ``LLMClient`` protocol, but instead of making API calls, it:

1. **Puts requests into an async queue**: When code agent calls ``await llm_client(request)``
2. **Waits on a Future**: The call blocks until someone provides a response
3. **Returns the response**: Code agent continues with the response

Meanwhile, the environment:

1. **Watches the queue**: Extracts ``LLMRequest`` objects as they arrive
2. **Exposes them as observations**: Returns them from ``reset()`` and ``step()``
3. **Provides responses**: When you call ``step(action)``, sets the Future's result

Implementation
~~~~~~~~~~~~~~

The core implementation is simple:

.. code-block:: python

    @dataclass(frozen=True)
    class QueueMediatedLLMClient(LLMClient):
        q: asyncio.Queue[ValueAndFuture[LLMRequest, LLMResponse]]

        async def __call__(self, request: LLMRequest) -> LLMResponse:
            future = asyncio.Future[LLMResponse]()
            await self.q.put(ValueAndFuture(value=request, future=future))
            return await future  # Blocks until env provides response

The environment side:

.. code-block:: python

    async def _get_time_step(self) -> TimeStep:
        # Wait for code agent to finish OR make an LLM request
        get_request = asyncio.create_task(self._llm_client.q.get())
        done, _ = await asyncio.wait(
            [self._code_agent_task, get_request],
            return_when=asyncio.FIRST_COMPLETED
        )

        if get_request in done:
            request, future = get_request.result()
            self._llm_req_future = future
            return TimeStep(step_type="MID", observation=request, ...)

    async def step(self, action: LLMResponse) -> TimeStep:
        # Unblock the code agent by providing response
        self._llm_req_future.set_result(action)
        return await self._get_time_step()

To dive into the code, see :py:class:`ares.llms.queue_mediated_client.QueueMediatedLLMClient` and :py:meth:`ares.environments.base.CodeBaseEnv._get_time_step`.

Multiple Environments
--------------------------------

ARES environments are async and independent, enabling parallel data collection:

.. code-block:: python

    async def collect_episodes(num_parallel: int):
        async def run_episode(env):
            async with env:
                ts = await env.reset()
                while not ts.last():
                    action = await policy(ts.observation)
                    ts = await env.step(action)
                return ts.reward

        envs = [create_env() for _ in range(num_parallel)]
        rewards = await asyncio.gather(*[run_episode(env) for env in envs])
        return rewards

Each environment runs independently with its own container, code agent, and queue. This scales naturally to hundreds of parallel episodes for distributed training.

Limitations and Trade-offs
---------------------------

**Not Suitable For**
    * Sub-millisecond latency requirements (queue adds overhead)
    * Synchronous (non-async) agent code (requires async/await)

**Design Trade-offs**
    * **Async Requirement**: Both agent and environment must use async/await
    * **Hidden Control Flow**: Agent appears to block but yields control to environment (can surprise when debugging)

Further Reading
---------------

* :doc:`core-concepts` - Overview of ARES architecture
* `dm_env specification <https://github.com/google-deepmind/dm_env>`_ - The RL interface ARES implements
