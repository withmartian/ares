Queue-Mediated LLM Client
=========================

The ``QueueMediatedLLMClient`` is the key implementation detail that enables ARES's RL abstraction. It allows code agents to be written in a natural, linear style while the environment exposes their LLM interactions as observations and actions.

The Problem
-----------

Consider how you'd build an RL environment for code agents without this pattern:

**Naive Approach**
    Make the agent explicitly RL-aware. The agent would need methods like ``get_next_action()`` that return commands to execute, rather than just executing them. This breaks the abstraction and makes agent code complex.

**Why This Fails**
    1. Agent code becomes tangled with RL logic
    2. Can't reuse existing agent implementations
    3. Loses the natural flow of "reason → act → observe → repeat"

The queue-mediated pattern solves this by **intercepting** LLM calls transparently.

How It Works
------------

The ``QueueMediatedLLMClient`` implements the ``LLMClient`` protocol, but instead of making real API calls, it:

1. **Puts requests into a queue**: When the agent calls ``await llm_client(request)``, the request goes into an ``asyncio.Queue``
2. **Waits for a response via a Future**: The call blocks until someone sets the result on an ``asyncio.Future``
3. **Returns the response**: Once the future is set, the agent continues executing with the response

Meanwhile, the environment:

1. **Watches the queue**: Extracts requests as they arrive
2. **Exposes them as observations**: Returns ``LLMRequest`` objects from ``reset()`` and ``step()``
3. **Provides responses as actions**: When you call ``step(action)``, it sets the future's result

This creates a bidirectional communication channel that lets the environment control the RL loop while the agent remains unaware.

Implementation
--------------

The core implementation is remarkably simple:

.. code-block:: python

    @dataclass(frozen=True)
    class QueueMediatedLLMClient(LLMClient):
        q: asyncio.Queue[ValueAndFuture[LLMRequest, LLMResponse]] = field(
            default_factory=asyncio.Queue
        )

        async def __call__(self, request: LLMRequest) -> LLMResponse:
            # Create a future that will hold the response
            future = asyncio.Future[LLMResponse]()

            # Put the request and future into the queue
            await self.q.put(ValueAndFuture(value=request, future=future))

            # Block until someone sets the future's result
            return await future

The environment side looks like this:

.. code-block:: python

    async def _get_time_step(self) -> TimeStep:
        # Wait for either the agent to finish or make an LLM request
        get_request_task = asyncio.create_task(self._llm_client.q.get())
        done, _ = await asyncio.wait(
            [self._code_agent_task, get_request_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        if get_request_task in done:
            # Got an LLM request - expose it as an observation
            request, future = get_request_task.result()
            self._llm_req_future = future  # Store for later
            return TimeStep(
                step_type="MID",
                observation=request,
                reward=None,
                discount=None,
            )

    async def step(self, action: LLMResponse) -> TimeStep:
        # Provide the action as the response to the agent's request
        self._llm_req_future.set_result(action)

        # Wait for the next request or completion
        return await self._get_time_step()

Example: Training Loop
----------------------

Here's how it all fits together in a training loop:

.. code-block:: python

    # The agent code is simple and linear
    class MyAgent:
        def __init__(self, container, llm_client):
            self._container = container
            self._llm_client = llm_client

        async def run(self, task: str):
            # The agent doesn't know this blocks until the environment
            # provides a response!
            response = await self._llm_client(
                LLMRequest(messages=[
                    {"role": "user", "content": task}
                ])
            )

            command = parse_command(response)
            await self._container.exec_run(command)
            # ... repeat ...

    # The environment orchestrates the RL loop
    async with env:
        ts = await env.reset()  # Gets first LLM request from agent

        while not ts.last():
            # Your RL policy observes the request and generates a response
            action = await your_policy(ts.observation)

            # This unblocks the agent and gets the next request
            ts = await env.step(action)

Why This Matters
----------------

This pattern enables several key capabilities:

**Agent Reusability**
    Any code agent that uses an ``LLMClient`` can work in ARES environments without modification. You can take existing agents and immediately use them for RL training.

**Clean Separation of Concerns**
    Agent logic (reasoning, tool use, planning) is separate from RL logic (policy learning, reward computation, exploration). Each can evolve independently.

**Mechanistic Interpretability**
    Since you have raw access to every LLM request and can provide any response, you can instrument agents for interpretability research - swap in probes, analyze activations, test counterfactuals, etc.

**Flexible Training Regimes**
    The same agent can be used for behavioral cloning (provide expert responses), online RL (learn a policy), or evaluation (use a fixed policy). The agent code never changes.

Comparison to Alternatives
---------------------------

Other Approaches
~~~~~~~~~~~~~~~~

**Callback-Based**
    Some frameworks use callbacks: the agent calls a function, the framework executes it and returns the result. This works but requires the agent to use framework-specific APIs.

    *Limitation*: Can't reuse arbitrary existing code agents.

**Message-Passing**
    The agent explicitly sends "messages" to an RL controller and waits for replies.

    *Limitation*: Agent code becomes RL-aware and harder to understand.

**Wrapper-Based**
    Wrap the entire agent in a harness that intercepts I/O.

    *Limitation*: Coarse-grained - you get all I/O, not specifically LLM interactions. Hard to distinguish LLM calls from logging, debugging, etc.

Why Queues and Futures?
~~~~~~~~~~~~~~~~~~~~~~~~

The queue-mediated pattern is **async-native** and **precisely scoped**:

* **Async-native**: Uses ``asyncio`` primitives efficiently, no threading or multiprocessing overhead
* **Precisely scoped**: Only intercepts ``LLMClient`` calls, not all I/O or function calls
* **Type-safe**: Both sides work with well-typed ``LLMRequest`` and ``LLMResponse`` objects
* **Non-invasive**: Zero changes to agent code required

Advanced: Multiple Agents
--------------------------

The pattern scales naturally to multiple agents. Each agent gets its own ``QueueMediatedLLMClient`` with its own queue, and the environment can multiplex between them:

.. code-block:: python

    # Environment with multiple agents
    class MultiAgentEnv:
        def __init__(self, num_agents: int):
            self._agents = []
            for i in range(num_agents):
                llm_client = QueueMediatedLLMClient()
                agent = MyAgent(container=..., llm_client=llm_client)
                self._agents.append((agent, llm_client))

        async def step(self, actions: list[LLMResponse]):
            # Set results for all agents
            for (agent, client), action in zip(self._agents, actions):
                client._pending_future.set_result(action)

            # Wait for next requests from any agent
            tasks = [client.q.get() for _, client in self._agents]
            results = await asyncio.gather(*tasks)
            return [TimeStep(..., observation=req, ...) for req, _ in results]

This enables multi-agent RL scenarios without changing individual agent implementations.

Limitations and Trade-offs
---------------------------

**Not Suitable For**
    * Real-time systems requiring sub-millisecond latency (the queue adds overhead)
    * Agents that need to make parallel LLM calls (requests are serialized through the queue)
    * Synchronous (non-async) agent code (requires async/await support)

**Design Trade-offs**
    * **Serialization**: Requests must be serializable for the queue. Complex stateful callbacks won't work.
    * **Async Requirement**: Both agent and environment must use ``async/await``. This is usually fine but rules out some existing sync codebases.
    * **Hidden Control Flow**: The agent appears to block on LLM calls, but actually yields control to the environment. This can be surprising when debugging.

Despite these trade-offs, the pattern is a net win for most RL training scenarios where the benefits (agent reusability, clean separation, interpretability) outweigh the costs.

Further Reading
---------------

* :doc:`core-concepts` - Overview of how this fits into ARES's architecture
* `asyncio documentation <https://docs.python.org/3/library/asyncio.html>`_ - Python's async primitives
* `dm_env specification <https://github.com/google-deepmind/dm_env>`_ - The RL environment interface ARES implements
