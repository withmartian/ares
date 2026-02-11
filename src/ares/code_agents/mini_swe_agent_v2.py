# Original code: https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License.
#
# Modifications Copyright (c) 2026 Martian

"""MiniSWECodeAgentV2 - Text-based interaction model with v2 style submission.

This is a v2 implementation that:
- Uses text-based interaction (parsing markdown code blocks)
- Uses COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT as the submission sentinel
- Based on mini_textbased.yaml configuration
- Inherits from code_agent_base.CodeAgent protocol

Last checked: Feb 11, 2026
GH link: https://github.com/SWE-agent/mini-swe-agent
"""

import dataclasses
import logging
import re
from typing import Literal, assert_never

import jinja2

from ares.code_agents import code_agent_base
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


# V2 text-based templates (from mini_textbased.yaml)
_V2_SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
""".strip()


_V2_INSTANCE_TEMPLATE = """
Please solve this issue: {{task}}

You can execute bash commands and edit files to implement the necessary changes.

## Recommended Workflow

This workflows should be done step-by-step so that you can iterate on your changes and any possible problems.

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust
6. Submit your changes and finish your work by issuing the following command:
   `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
   Do not combine it with any other command.
   <important>After this command, you cannot continue working on this task.</important>

## Important Rules

1. Every response must contain exactly one action
2. The action must be enclosed in triple backticks
3. Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
   However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...`
   or write/load environment variables from files

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Formatting your response

Here is an example of a correct response:

<example_response>
THOUGHT: I need to understand the structure of the repository first.
Let me check what files are in the current directory to get a better understanding of the codebase.

```mswea_bash_command
ls -la
```
</example_response>

## Useful command examples

### Create a new file:

```mswea_bash_command
cat <<'EOF' > newfile.py
import numpy as np
hello = "world"
print(hello)
EOF
```

### Edit files with sed:

{%- if system == "Darwin" -%}
<important>
You are on MacOS. For all the below examples, you need to use `sed -i ''` instead of `sed -i`.
</important>
{%- endif -%}

```mswea_bash_command
# Replace all occurrences
sed -i 's/old_string/new_string/g' filename.py

# Replace only first occurrence
sed -i 's/old_string/new_string/' filename.py

# Replace first occurrence on line 1
sed -i '1s/old_string/new_string/' filename.py

# Replace all occurrences in lines 1-10
sed -i '1,10s/old_string/new_string/g' filename.py
```

### View file content:

```mswea_bash_command
# View specific lines with numbers
nl -ba filename.py | sed -n '10,20p'
```

### Any other command you want to run

```mswea_bash_command
anything
```
""".strip()


_V2_OBSERVATION_TEMPLATE = """
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file
and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
""".strip()


_V2_FORMAT_ERROR_TEMPLATE = """
Format error:

<error>
{{error}}
</error>

Here is general guidance on how to format your response:

Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
Here are some thoughts about why you want to perform the action.

```mswea_bash_command
<action>
```
</response_example>

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
""".strip()


_V2_TIMEOUT_TEMPLATE = """
The last command <command>{action}</command> timed out and has been killed.
The output of the command was:
<output>
{output}
</output>
Please try another command and make sure to avoid those requiring interactive input.
""".strip()


# Default environment variables for text-based mode
_V2_DEFAULT_ENV_VARS = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


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
class _AgentOutput:
    """Output from executing a bash command in the container."""

    returncode: int
    output: str
    exception_info: str | None = None


def _render_system_template() -> str:
    """Render the v2 system template."""
    return jinja2.Template(_V2_SYSTEM_TEMPLATE, undefined=jinja2.StrictUndefined).render()


def _render_instance_template(task: str, system: str, release: str, version: str, machine: str) -> str:
    """Render the v2 instance template with task and system information."""
    return jinja2.Template(_V2_INSTANCE_TEMPLATE, undefined=jinja2.StrictUndefined).render(
        task=task,
        system=system,
        release=release,
        version=version,
        machine=machine,
    )


def _render_observation_template(output: _AgentOutput) -> str:
    """Render the v2 observation template with command output."""
    return jinja2.Template(_V2_OBSERVATION_TEMPLATE, undefined=jinja2.StrictUndefined).render(
        output=output,
    )


def _render_format_error_template(error: str, actions: list[str]) -> str:
    """Render the v2 format error template."""
    return jinja2.Template(_V2_FORMAT_ERROR_TEMPLATE, undefined=jinja2.StrictUndefined).render(
        error=error,
        actions=actions,
    )


def _render_timeout_template(action: str, output: str) -> str:
    """Render the v2 timeout template."""
    return _V2_TIMEOUT_TEMPLATE.format(action=action, output=output)


@dataclasses.dataclass(kw_only=True)
class MiniSWECodeAgentV2(code_agent_base.CodeAgent):
    """V2 text-based mini-swe-agent implementation.

    This agent uses the text-based interaction model where:
    - Commands are parsed from markdown code blocks (```mswea_bash_command or ```bash)
    - The agent submits by echoing COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
    - ARES does not yet support structured tool calls in LLMResponse
    """

    container: containers.Container
    llm_client: llm_clients.LLMClient
    tracker: stat_tracker.StatTracker = dataclasses.field(default_factory=stat_tracker.NullStatTracker)

    # Configuration
    step_limit: int = 0  # 0 means no limit
    cost_limit: float = 3.0  # Default $3 limit
    timeout_s: int | None = None  # Command execution timeout

    def __post_init__(self):
        """Initialize the agent with v2 configuration."""
        # Initialize step and cost tracking
        self._n_calls = 0
        self._total_cost = 0.0

        # Environment variables for text-based mode
        self._environment_env_vars = _V2_DEFAULT_ENV_VARS

        # Render system prompt once
        self._system_prompt = _render_system_template()
        self._messages: list[request.Message] = []

        _LOGGER.debug("[%d] Initialized MiniSWECodeAgentV2.", id(self))

    def _add_message(self, role: Literal["user", "assistant"], content: str) -> None:
        """Add a message to the conversation history."""
        if role == "user":
            self._messages.append(request.UserMessage(role="user", content=content))
        elif role == "assistant":
            self._messages.append(request.AssistantMessage(role="assistant", content=content))
        else:
            assert_never(role)

    async def run(self, task: str) -> None:
        """Run the agent loop until completion or termination.

        Args:
            task: The task description/problem statement to solve.
        """
        # Get system information from the container
        with self.tracker.timeit("mswea_v2/setup"):
            _LOGGER.debug("[%d] Starting MiniSWECodeAgentV2 run.", id(self))

            system = (await self.container.exec_run("uname -a", env=self._environment_env_vars)).output.strip()
            release = (await self.container.exec_run("uname -r", env=self._environment_env_vars)).output.strip()
            version = (await self.container.exec_run("uname -v", env=self._environment_env_vars)).output.strip()
            machine = (await self.container.exec_run("uname -m", env=self._environment_env_vars)).output.strip()

            _LOGGER.debug("[%d] System information: %s %s %s %s", id(self), system, release, version, machine)

            # Add initial task message
            self._add_message(
                "user",
                _render_instance_template(
                    task=task,
                    system=system,
                    release=release,
                    version=version,
                    machine=machine,
                ),
            )

        # Main agent loop
        while True:
            try:
                with self.tracker.timeit("mswea_v2/step"):
                    await self.step()
            except _NonTerminatingError as e:
                _LOGGER.debug("[%d] Non-terminating error: %s", id(self), e)
                self._add_message("user", str(e))
            except _TerminatingError as e:
                _LOGGER.debug("[%d] Terminating error: %s", id(self), e)
                return

    async def step(self) -> None:
        """Execute one step of the agent loop: query LLM and execute action."""
        llm_response = await self.query()
        await self.execute_action(llm_response)

    async def query(self) -> response.LLMResponse:
        """Query the LLM and return the response."""
        # Check step limit before making LLM call
        if 0 < self.step_limit <= self._n_calls:
            raise _LimitsExceededError(f"Step limit of {self.step_limit} exceeded")

        # Check cost limit before making LLM call
        if 0 < self.cost_limit <= self._total_cost:
            raise _LimitsExceededError(f"Cost limit of ${self.cost_limit:.2f} exceeded")

        _LOGGER.debug("[%d] Querying LLM (call #%d).", id(self), self._n_calls + 1)

        with self.tracker.timeit("mswea_v2/llm_request"):
            llm_response = await self.llm_client(
                request.LLMRequest(
                    messages=self._messages,
                    system_prompt=self._system_prompt,
                    temperature=0.0,
                )
            )

        _LOGGER.debug("[%d] LLM response received.", id(self))

        # Update tracking
        self._n_calls += 1
        self._total_cost += llm_response.cost

        # Extract and add assistant message
        message_content = llm_response.data[0].content
        assert message_content is not None

        self._add_message("assistant", message_content)

        return llm_response

    async def execute_action(self, llm_response: response.LLMResponse) -> None:
        """Parse and execute the action from the LLM response."""
        _LOGGER.debug("[%d] Executing action.", id(self))

        response_text = llm_response.data[0].content
        assert response_text is not None

        # Parse the bash command from the response
        action = self.parse_action(response_text)

        # Execute the command in the container
        with self.tracker.timeit("mswea_v2/exec_run"):
            try:
                exec_result = await self.container.exec_run(
                    action,
                    timeout_s=self.timeout_s,
                    env=self._environment_env_vars,
                )
                output = _AgentOutput(
                    returncode=exec_result.exit_code,
                    output=exec_result.output,
                    exception_info=None,
                )
            except TimeoutError as e:
                # NOTE: Unlike the official implementation, we cannot retrieve partial output on timeout
                # because asyncio.wait_for cancels the task without preserving partial execution state.
                raise _ExecutionTimeoutError(_render_timeout_template(action, "")) from e
            except Exception as e:
                # Catch other execution errors
                output = _AgentOutput(
                    returncode=-1,
                    output="",
                    exception_info=str(e),
                )

        _LOGGER.debug("[%d] Action executed with returncode %d.", id(self), output.returncode)

        # Check if the agent has submitted its final output
        self._raise_if_finished(exec_result)

        # Render observation and add to messages
        observation = _render_observation_template(output)
        self._add_message("user", observation)

    def parse_action(self, response_text: str) -> str:
        """Parse bash command from markdown code blocks.

        Supports both ```mswea_bash_command and ```bash code block formats.

        Args:
            response_text: The LLM's response text.

        Returns:
            The parsed bash command.

        Raises:
            _FormatError: If the response doesn't contain exactly one code block.
        """
        # Try both mswea_bash_command and bash patterns
        patterns = [
            r"```mswea_bash_command\s*\n(.*?)\n```",
            r"```bash\s*\n(.*?)\n```",
        ]

        all_actions = []
        for pattern in patterns:
            actions = re.findall(pattern, response_text, re.DOTALL)
            all_actions.extend(actions)

        if len(all_actions) == 1:
            return all_actions[0].strip()

        # Format error - wrong number of code blocks
        error_msg = f"Expected exactly 1 code block, found {len(all_actions)}"
        format_error_str = _render_format_error_template(error_msg, all_actions)
        raise _FormatError(format_error_str)

    def _raise_if_finished(self, output: containers.ExecResult) -> None:
        """Check if the agent has submitted and raise _SubmittedError if so.

        The v2 submission sentinel is COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.

        Args:
            output: The execution result from the container.

        Raises:
            _SubmittedError: If the output contains the submission sentinel.
        """
        lines = output.output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
            # Everything after the first line is the final output
            final_output = "".join(lines[1:])
            _LOGGER.info("[%d] Agent submitted final output.", id(self))
            raise _SubmittedError(final_output)

    # TODO: Implement compact logic
    # The compact feature allows the agent to output a special token/command to signal
    # that it wants to receive condensed observations (e.g., omitting verbose output).
    # This is a placeholder for future implementation when ARES supports this feature.
    # For now, all observations are rendered in full according to _V2_OBSERVATION_TEMPLATE.
