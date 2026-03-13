"""An interface for code agents."""

import dataclasses
from typing import Protocol

from ares.containers import containers
from ares.llms import llm_clients
from ares.llms import request


class CodeAgent(Protocol):
    async def run(self, task: str) -> None:
        """Runs the agent for the specific task."""
        ...


class CodeAgentFactory[T: CodeAgent](Protocol):
    def __call__(self, *, container: containers.Container, llm_client: llm_clients.LLMClient) -> T: ...


@dataclasses.dataclass(frozen=True)
class CodeAgentState:
    """Serializable snapshot of a code agent's conversational state.

    Captures everything needed to resume an agent from where it left off,
    given that its container is in the correct filesystem state.
    """

    messages: list[request.Message]
    n_calls: int
    total_cost: float


class CheckpointableCodeAgent(CodeAgent, Protocol):
    """A CodeAgent that supports state capture and restoration.

    Used by Go-Explore to checkpoint and resume agents at arbitrary
    points in their execution.
    """

    def get_state(self) -> CodeAgentState:
        """Capture the agent's current conversational state.

        Returns:
            A CodeAgentState that can be passed to restore_and_resume().
        """
        ...

    async def restore_and_resume(self, state: CodeAgentState, task: str) -> None:
        """Restore agent state and resume execution from the next LLM call.

        Sets the agent's message history and counters to the saved state,
        then enters the normal query/execute loop. The first query() call
        will produce an LLM request with the restored message history,
        which gets intercepted by the QueueMediatedLLMClient.

        Args:
            state: A previously captured CodeAgentState.
            task: The task instruction (used for context, not re-executed).
        """
        ...


class TrivialCodeAgent(CodeAgent):
    """An example code agent that does nothing for debugging.

    Note: May be removed in the near future.
    """

    def __init__(self, container: containers.Container, llm_client: llm_clients.LLMClient):
        self._container = container
        self._llm_client = llm_client

    async def run(self, problem_statement: str) -> None:
        del problem_statement  # Unused.

        for _ in range(50):
            await self._llm_client(request.LLMRequest(messages=[{"role": "user", "content": "Print 'Yes' only."}]))
            await self._container.exec_run("sleep 1")

        return
