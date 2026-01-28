"""An interface for code agents."""

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
