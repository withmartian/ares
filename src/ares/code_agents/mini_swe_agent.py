"""A Code Agent wrapping the mini-swe-agent.

Last checked: Nov 15, 2025
GH link: https://github.com/SWE-agent/mini-swe-agent
Commit hash: 6ff7d26ac371e5bb9611ec37074fc1bedf400895
"""

import dataclasses
import logging
import pathlib
import re
from typing import Literal

import jinja2
import minisweagent
from minisweagent.agents import default as default_agent
import minisweagent.config
from openai.types.chat import chat_completion_message_param
import yaml

from ares.code_agents import code_agent_base
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients

_LOGGER = logging.getLogger(__name__)


# Copied from minisweagent's default config.
_TIMEOUT_TEMPLATE = """
The last command <command>{action}</command> timed out and has been killed.
The output of the command was:
<output>
{output}
</output>
Please try another command and make sure to avoid those requiring interactive input.
""".strip()


class _NonTerminatingError(Exception):
    """Raised for conditions that can be handled by the agent."""


class _FormatError(_NonTerminatingError):
    """Raised when the LM's output is not in the expected format."""


class _ExecutionTimeoutError(_NonTerminatingError):
    """Raised when the action execution timed out."""


class _TerminatingError(Exception):
    """Raised for conditions that terminate the agent."""


class _SubmittedError(_TerminatingError):
    """Raised when the LM declares that the agent has finished its task."""


class _LimitsExceededError(_TerminatingError):
    """Raised when the agent has reached its cost or step limit."""


@dataclasses.dataclass(frozen=True)
class _MiniSWEAgentOutput:
    returncode: int
    output: str


def _render_system_template(system_template: str) -> str:
    return jinja2.Template(system_template, undefined=jinja2.StrictUndefined).render()


def _render_instance_template(
    instance_template: str, task: str, system: str, release: str, version: str, machine: str
) -> str:
    return jinja2.Template(instance_template, undefined=jinja2.StrictUndefined).render(
        task=task,
        system=system,
        release=release,
        version=version,
        machine=machine,
    )


def _render_action_observation_template(action_observation_template: str, output: _MiniSWEAgentOutput) -> str:
    return jinja2.Template(action_observation_template, undefined=jinja2.StrictUndefined).render(
        output=output,
    )


def _render_format_error_template(format_error_template: str, actions: list[str]) -> str:
    return jinja2.Template(format_error_template, undefined=jinja2.StrictUndefined).render(
        actions=actions,
    )


def _render_timeout_template(action: str, output: str) -> str:
    # TODO: Use jinja2, and allow updating of configuration.
    return _TIMEOUT_TEMPLATE.format(action=action, output=output)


@dataclasses.dataclass(kw_only=True)
class MiniSWECodeAgent(code_agent_base.CodeAgent):
    container: containers.Container
    llm_client: llm_clients.LLMClient
    tracker: stat_tracker.StatTracker = dataclasses.field(default_factory=stat_tracker.NullStatTracker)

    def __post_init__(self):
        config_path = pathlib.Path(minisweagent.config.builtin_config_dir) / "extra" / "swebench.yaml"
        self._config = yaml.safe_load(config_path.read_text())
        self._agent_config = self._config.get("agent", {})

        environment_config = self._config.get("environment", {})
        self._env_timeout = environment_config.get("timeout", None)
        self._environment_env_vars = environment_config.get("env", None)

        # Somewhat frustratingly, minisweagent uses kwargs.
        # We handle this by inspecting whether an argument will be accepted by the agent config.
        agent_config_dict = self._config.get("agent", {})
        agent_config = default_agent.AgentConfig()
        for k, v in agent_config_dict.items():
            if hasattr(default_agent.AgentConfig, k):
                setattr(agent_config, k, v)
            else:
                _LOGGER.info("Ignoring argument %s in agent configuration.", k)

        self._messages: list[chat_completion_message_param.ChatCompletionMessageParam] = []
        _LOGGER.debug("[%d] Initialized MiniSWECodeAgent.", id(self))

    def _add_message(self, role: Literal["system", "user", "assistant"], content: str) -> None:
        if not content.strip():
            _LOGGER.warning("[%d] Ignoring empty message with role %s.", id(self), role)
            content = "[Empty content]"

        self._messages.append(
            {"role": role, "content": content},  # type: ignore
        )

    async def run(self, task: str) -> None:
        """Run step() until agent is finished. Return exit status & message"""
        # Get system information from the container.
        with self.tracker.timeit("mswea/setup"):
            _LOGGER.debug("[%d] Starting miniswe-agent run.", id(self))

            system = (await self.container.exec_run("uname -a", env=self._environment_env_vars)).output.strip()
            release = (await self.container.exec_run("uname -r", env=self._environment_env_vars)).output.strip()
            version = (await self.container.exec_run("uname -v", env=self._environment_env_vars)).output.strip()
            machine = (await self.container.exec_run("uname -m", env=self._environment_env_vars)).output.strip()

            _LOGGER.debug("[%d] System information: %s %s %s %s", id(self), system, release, version, machine)

            self._add_message("system", _render_system_template(self._agent_config["system_template"]))
            self._add_message(
                "user",
                _render_instance_template(
                    self._agent_config["instance_template"],
                    task=task,
                    system=system,
                    release=release,
                    version=version,
                    machine=machine,
                ),
            )

        while True:
            try:
                with self.tracker.timeit("mswea/step"):
                    await self.step()
            except _NonTerminatingError as e:
                _LOGGER.debug("[%d] Non-terminating error: %s", id(self), e)
                self._add_message("user", str(e))
            except _TerminatingError as e:
                _LOGGER.debug("[%d] Terminating error: %s", id(self), e)
                self._add_message("user", str(e))
                return

    async def step(self) -> None:
        """Query the LM, execute the action, return the observation."""
        llm_response = await self.query()
        await self.execute_action(llm_response)

    async def query(self) -> llm_clients.LLMResponse:
        """Query the model and return the response."""
        # if 0 < self._config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
        # raise LimitsExceeded()
        _LOGGER.debug("[%d] Querying LLM.", id(self))

        with self.tracker.timeit("mswea/llm_request"):
            response = await self.llm_client(
                llm_clients.LLMRequest(
                    messages=self._messages,
                    temperature=0.0,
                )
            )
        _LOGGER.debug("[%d] LLM response received.", id(self))

        message_content = response.chat_completion_response.choices[0].message.content
        assert message_content is not None

        self._add_message("assistant", message_content)

        return response

    async def execute_action(self, response: llm_clients.LLMResponse) -> None:
        """Execute the action and return the observation."""
        _LOGGER.debug("[%d] Executing action.", id(self))
        response_text = response.chat_completion_response.choices[0].message.content
        assert response_text is not None

        action = self.parse_action(response_text)

        with self.tracker.timeit("mswea/exec_run"):
            try:
                output = await self.container.exec_run(
                    action,
                    timeout_s=self._env_timeout,
                    env=self._environment_env_vars,
                )
            except TimeoutError as e:
                raise _ExecutionTimeoutError(_render_timeout_template(action, "")) from e

        _LOGGER.debug("[%d] Action executed.", id(self))
        self._raise_if_finished(output)

        observation = _render_action_observation_template(
            self._agent_config["action_observation_template"],
            output=_MiniSWEAgentOutput(returncode=output.exit_code, output=output.output),
        )
        self._add_message("user", observation)

    def parse_action(self, response_text: str) -> str:
        """Parse the action from the message. Returns the action."""
        actions = re.findall(r"```bash\s*\n(.*?)\n```", response_text, re.DOTALL)
        if len(actions) == 1:
            return actions[0].strip()

        format_error_str = _render_format_error_template(self._agent_config["format_error_template"], actions=actions)
        raise _FormatError(format_error_str)

    def _raise_if_finished(self, output: containers.ExecResult):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise _SubmittedError("".join(lines[1:]))
