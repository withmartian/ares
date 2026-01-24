Core Concepts
=============

ARES provides a reinforcement learning framework that enables training policies (agents) to produce better LLM responses for code agents. Unlike traditional frameworks that treat the entire code agent as the optimization target, ARES trains the **LLM within the agent** by treating LLM interactions as observations and actions within a standard RL loop.

**Key Distinction**

It's important to understand two different concepts in ARES:

* Code Agent (Static)
    The orchestration logic that uses a Container and LLM to solve tasks (e.g., MiniSWECodeAgent). This is **part of the environment** and remains fixed during training. Think of it as the scaffold that defines how an LLM interacts with code.

* Agent/Policy (Trained)
    The component you're actually training - a function that maps ``LLMRequest → LLMResponse``. This could be a fine-tuned LLM, a prompt optimizer, or any policy that produces better responses. This is what improves through reinforcement learning.

System Architecture
-------------------

Here's how the components fit together:

.. code-block:: text

    Your Training Loop                    ARES Environment
    ═════════════════                     ════════════════

    ┌────────────────────────┐
    │  Your RL Policy/Agent  │            ┌──────────────────────────────────────┐
    │  (e.g. Fine-tuned LLM) │            │         CodeBaseEnv                  │
    │  receives request,     |            |                                      |
    |  generates response    |            │                                      │
    └──────────┬─────────────┘            │  ┌────────────────────────────────┐  │
        ^      │                          │  │   QueueMediatedLLMClient       │  │
        |      │ LLMResponse (action)     │  │                                │  │
        |      └──────────────────────────┼─>│   Intercepts LLM calls         │  │
        |                                 │  │   from code agent via          │  │
        └─────────────────────────────────┼──│   QueueMediatedLLMClient       │  │
                 LLMRequest (observation) │  └──────────────────┬─────────────┘  │
                                          │                 ^   │                │
                                          │      LLMRequest │   │ LLMResponse    │
                                          │                 │   v                │
                                          │  ┌──────────────└─────────────────┐  │
                                          │  │       CodeAgent                │  │
                                          │  │  (e.g. MiniSWECodeAgent)       │  │
                                          │  │                                │  │
                                          │  │  - Reasons about task          │  │
                                          │  │  - Calls LLM (blocks)          │  │
    ┌────────────────────────────┐        │  │  - Runs commands in Container  │  │
    │  Multiple Environments     │        │  │  - Iterates until done         │  │
    │  can run in parallel       │        │  └────────────────────┬───────────┘  │
    │                            │        │                 ^     |              │
    │  async with env1, env2:    │        │      cmd output |     │ exec_run()   │
    │      # Parallel episodes   │        │                 |     v              │
    └────────────────────────────┘        │  ┌──────────────└─────────────────┐  │
                                          │  │       Container                │  │
                                          │  │  (Docker or Daytona)           │  │
                                          │  │                                │  │
                                          │  │  - Isolated environment        │  │
                                          │  │  - Runs bash commands          │  │
                                          │  │  - File upload/download        │  │
                                          │  └────────────────────────────────┘  │
                                          └──────────────────────────────────────┘

Key Properties
~~~~~~~~~~~~~~

**Composability**
    Each component has a narrow interface and can be swapped independently. Want cloud containers? Switch the factory. Want a different agent? Swap the agent factory.

**Scalability**
    Environments are async and independent. Run hundreds in parallel with ``asyncio.gather()`` for distributed data collection.

**RL Native**
    The architecture naturally maps to RL: observations, actions, rewards, episodes. Use any RL algorithm - policy gradient, Q-learning, behavioral cloning, etc.

**LLM-Focused Optimization**
    Unlike frameworks that treat the entire agent as a black box, ARES gives you fine-grained control over the LLM's behavior at every step.

Environment
-----------

An **Environment** encapsulates the task, container, and code agent as a single RL environment. ARES implements an async version of `DeepMind's dm_env specification <https://github.com/google-deepmind/dm_env>`_.

The key abstraction is ``CodeBaseEnv``, which:

* **Manages a Container** - Provides an isolated execution environment
* **Manages a CodeAgent** - Runs the orchestration logic for solving the task
* **Exposes LLM requests as observations** - Intercepts calls from the code agent
* **Treats LLM responses as actions** - Your trainable agent/policy provides responses

Crucially, the **CodeAgent is part of the environment**, not what you're training. Your training loop optimizes an agent/policy that produces better ``LLMResponse`` outputs given ``LLMRequest`` observations.

Standard RL Loop
~~~~~~~~~~~~~~~~

Every environment follows the standard RL pattern:

.. code-block:: python

    async with env:
        # Start a new episode
        timestep = await env.reset()

        while not timestep.last():
            # timestep.observation is an LLMRequest from the code agent
            action = await your_policy(timestep.observation)

            # action is an LLMResponse that continues the agent's execution
            timestep = await env.step(action)

        # timestep.reward contains the reward for the final step
        print(f"Final reward: {timestep.reward}")

TimeStep Structure
~~~~~~~~~~~~~~~~~~

Each call to ``reset()`` or ``step()`` returns a ``TimeStep`` with:

* ``step_type``: One of ``"FIRST"``, ``"MID"``, or ``"LAST"``
* ``observation``: An ``LLMRequest`` object (or ``None`` on termination)
* ``reward``: A float reward for each step
* ``discount``: A float discount factor for RL algorithms

CodeAgent
---------

A **CodeAgent** implements the orchestration logic for attempting to solve a task. It has access to a ``Container`` (to execute shell commands) and an ``LLMClient`` (to interact with the language model).

The minimal interface is simple:

.. code-block:: python

    class CodeAgent(Protocol):
        async def run(task: str) -> None:
            """Runs the agent for the specific task."""

    class CodeAgentFactory[T: CodeAgent](Protocol):
        def __call__(self, *, container: Container, llm_client: QueueMediatedLLMClient) -> T: ...
            """Instantiates a new CodeAgent."""

Agent Implementation Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A typical code agent:

1. Receives a task description (e.g., "Fix the authentication bug")
2. Makes LLM calls to reason about what to do
3. Executes bash commands in the container to inspect code, run tests, make edits
4. Iterates between LLM reasoning and command execution
5. Signals completion when done (implementation-specific)

Example structure:

.. code-block:: python

    class MyCodeAgent:
        def __init__(self, container: Container, llm_client: QueueMediatedLLMClient):
            self._container = container
            self._llm_client = llm_client

        async def run(self, task: str) -> None:
            while not self.is_done():
                # Ask LLM what to do next
                request = LLMRequest(messages=[...])
                response = await self._llm_client(request)

                # Parse and execute commands from LLM response
                commands = self.parse_commands(response)
                for cmd in commands:
                    result = await self._container.exec_run(cmd)
                    # Use result in next LLM call...

Connection to the RL Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's the key insight: **The agent doesn't know it's part of an RL loop.**

When the agent calls ``await self._llm_client(request)``, it blocks and waits for a response. But the ``LLMClient`` is actually a ``QueueMediatedLLMClient`` (see :doc:`how-it-works`), which:

1. Puts the request into a queue
2. Waits for someone to provide a response
3. Returns that response to the agent

The environment watches this queue and exposes requests as **observations**. Your RL policy provides responses as **actions**. This lets you train the LLM while the agent code remains simple and linear.

Available Agents
~~~~~~~~~~~~~~~~

**MiniSWECodeAgent** (``ares.code_agents.mini_swe_agent``)
    Wraps the `mini-swe-agent <https://github.com/SWE-agent/mini-swe-agent>`_ library. Uses Jinja2 templates for prompts, parses bash commands from markdown, handles timeouts and retries.

Implementing your own CodeAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To bring in your own CodeAgent implementation, the main blocker is typically rewriting around any LLM calls and comand execution that your agent makes.
This can look like:

.. code-block:: python

    class MyCurrentCodeAgent:
        def __init__(self, ..., llm_client: openai.AsyncClient):
            ...
            self.llm_client = llm_client

        async def run(self, task: str) -> None:
            # Do some setup for tools and what not
            ...
            while not self.is_done():
                # Decide what to ask LLM
                ...
                llm_response = await self.llm_client.chat.completions.create(
                    ...
                    messages=[...],
                )
                # Parse the LLM response and execute commands
                ...
                cmd_output = await self.run_command(command)
                ...

Which you will need to rewrite into something like:

.. code-block:: python

    class MyARESCodeAgent:
        def __init__(self, container: Container, llm_client, QueueMediatedLLMClient):
            self.llm_client = llm_client
            self.container = container
            # Replace other init setup
            ...

        async def run(self, task: str) -> None:
            # Do some setup for tools and what not
            ...
            while not self.is_done():
                # Decide what to ask LLM next
                ...
                llm_response = await self.llm_client(
                    LLMRequest(
                        messages=[...],
                        ...  # Other request params
                    )
                )
                # Parse the LLM response and execute commands
                ...
                cmd_output = await self.container.exec_run(command)
                ...

We are working on making the integration for adding CodeAgents as easy as possible, and hopefully more to come on this soon! We unfortunately don't support arbitrary MCP tool calls yet, but that is one of multiple things that are top of mind. For the time being, depending on the specific tools you may be able to fit them into the existing CodeAgent API - and if not, please let us know on [GitHub](https://github.com/withmartian/ares/issues) or join our [Discord server](https://discord.gg/5Y93Zhg3eS)!

Container
---------

A **Container** provides an isolated execution environment where code agents can safely run commands, modify files, and execute code.

Containers abstract over different backend implementations (local Docker, cloud providers) with a consistent interface:

.. code-block:: python

    class Container(Protocol):
        async def start(env: dict[str, str] | None) -> None
        async def stop() -> None
        async def exec_run(command, workdir, env, timeout_s) -> ExecResult
        async def upload_files(local_paths, remote_paths) -> None
        async def download_files(remote_paths, local_paths) -> None

Available Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

**DockerContainer** (``ares.containers.docker``)
    Uses local Docker for container management. Builds images from Dockerfiles on-demand. Best for development and single-machine experiments.

**DaytonaContainer** (``ares.containers.daytona``)
    Uses `Daytona <https://www.daytona.io>`_ for cloud-based containers. Supports distributed workloads, resource limits (CPU/memory/disk/GPU), and auto-cleanup. Best for production training runs.

Container Lifecycle
~~~~~~~~~~~~~~~~~~~

Containers are managed by the environment:

1. **Creation**: Environment calls the container factory with an image or Dockerfile
2. **Start**: Container is started with task-specific environment variables
3. **Execution**: Code agent runs commands via ``exec_run()``
4. **Cleanup**: Container is stopped and removed when the environment closes

You typically don't interact with containers directly - the ``CodeBaseEnv`` handles their lifecycle.

LLMClient
---------

An **LLMClient** provides a simple, uniform interface for making LLM API calls. It's a quality-of-life abstraction that makes it easy to treat LLM interactions as observations and actions.

Core Interface
~~~~~~~~~~~~~~

.. code-block:: python

    class LLMClient(Protocol):
        async def __call__(request: LLMRequest) -> LLMResponse

    @dataclass(frozen=True)
    class LLMRequest:
        messages: Iterable[ChatCompletionMessageParam]
        temperature: float | None = None

    @dataclass(frozen=True)
    class LLMResponse:
        chat_completion_response: ChatCompletion
        cost: float

This simple interface wraps OpenAI-style chat completion APIs. The ``messages`` field follows the OpenAI format with ``role`` (system/user/assistant) and ``content``.

Why LLMClient?
~~~~~~~~~~~~~~

The ``LLMClient`` abstraction serves two purposes:

1. **Observations = LLM Requests**: In the RL loop, ``timestep.observation`` is an ``LLMRequest`` containing the messages the code agent wants to send to the LLM. This is the "state" your policy observes.

2. **Actions = LLM Responses**: In the RL loop, the ``action`` you pass to ``env.step()`` is an ``LLMResponse`` containing the LLM's reply. This is how your policy controls the agent's behavior.

This framing makes it natural to think about code agent training as an RL problem: you're learning a policy that maps agent requests to helpful responses.

Available Implementations
~~~~~~~~~~~~~~~~~~~~~~~~~

**ChatCompletionCompatibleLLMClient** (``ares.llms.chat_completions_compatible``)
    Makes real API calls to OpenAI-compatible endpoints (OpenAI, Martian, etc.). Includes retry logic, cost tracking, and configurable base URLs.

**QueueMediatedLLMClient** (``ares.llms.queue_mediated_client``)
    The critical piece that enables the RL abstraction. See :doc:`how-it-works` for details.

**MockLLMClient** (``ares.llms.mock_llm_client``)
    Returns pre-defined responses for testing and debugging.

Next Steps
----------

* Learn about the QueueMediatedLLMClient pattern that makes the RL abstraction possible - :doc:`how-it-works`
* See the `README <https://github.com/withmartian/ares>`_ for usage examples
