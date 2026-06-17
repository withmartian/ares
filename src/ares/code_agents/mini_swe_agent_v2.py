# Original code: https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py
# Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez
# Licensed under the MIT License.
#
# Modifications Copyright (c) 2026 Martian

"""MiniSWECodeAgentV2 - Tool-call based interaction model matching mini-swe-agent v2.

This is a v2 implementation that:
- Uses OpenAI-style tool calls (BASH_TOOL) instead of regex parsing
- Supports parallel tool calls for executing multiple commands per turn
- Uses role="tool" with tool_call_id for observations
- Fully configurable templates, limits, and behavior
- Based on swebench.yaml configuration from mini-swe-agent

All parameters can be customized via constructor. For convenience, use preset
configurations from mini_swe_agent_v2_configs module:

Example usage:
    from ares.code_agents import mini_swe_agent_v2_configs

    # Tool-call based agent (default, matches reference implementation)
    config = mini_swe_agent_v2_configs.MiniSWEAgentV2Config.tool_call_based()
    agent = MiniSWECodeAgentV2(
        container=container,
        llm_client=client,
        **dataclasses.asdict(config)
    )

    # SWE-bench Verified configuration
    config = mini_swe_agent_v2_configs.MiniSWEAgentV2Config.swe_bench_verified()
    agent = MiniSWECodeAgentV2(
        container=container,
        llm_client=client,
        **dataclasses.asdict(config)
    )

Last checked: Feb 17, 2026
GH link: https://github.com/SWE-agent/mini-swe-agent
"""

import dataclasses
import logging
from typing import Literal, assert_never

import jinja2

from ares.code_agents import code_agent_base
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


# Tool definition matching mini-swe-agent v2
# https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/models/utils/actions_toolcall.py
BASH_TOOL: request.Tool = {
    "name": "bash",
    "description": "Execute a bash command",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            }
        },
        "required": ["command"],
    },
}


# V2 tool-call based templates (from swebench.yaml)
_V2_SYSTEM_TEMPLATE = """
You are a helpful assistant that can interact with a computer shell to solve programming tasks.
""".strip()


_V2_INSTANCE_TEMPLATE = """
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue AT LEAST ONE command, see the result, then think and issue your next command(s).</important>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide one or more bash tool calls to execute

## Important Boundaries

- MODIFY: Regular source code files in {{working_directory}} (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. **Reasoning text** where you explain your analysis and plan
2. At least one tool call with your command

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE bash tool call. You can make MULTIPLE tool calls in a single response when the commands are independent (e.g., searching multiple files, reading different parts of the codebase).
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

Example of a CORRECT response:
<example_response>
I need to understand the Builder-related code. Let me find relevant files and check the project structure.

[Makes multiple bash tool calls: {"command": "ls -la"}, {"command": "find src -name '*.java' | grep -i builder"}, {"command": "cat README.md | head -50"}]
</example_response>

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can use bash commands or invoke any tool that is available in the environment
- You can also create new tools or scripts to help you with the task
- If a tool isn't available, you can also install it

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:

- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless they are directly part of the issue you were fixing (you can assume that the environment is already set up for your client)
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
- If you modify patch.txt after verifying, you SHOULD verify again before submitting.
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>
""".strip()  # noqa: E501


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
Tool call error:

<error>
{{error}}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
""".strip()


# Default environment variables
_V2_DEFAULT_ENV_VARS = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}


class _InterruptAgentFlowError(Exception):
    """Base exception for interrupting the agent flow with messages to add."""

    def __init__(self, messages: list[request.Message]):
        self.messages = messages
        super().__init__()


class _FormatError(_InterruptAgentFlowError):
    """Raised when the LM's output is not in the expected format."""


class _SubmittedError(_InterruptAgentFlowError):
    """Raised when the LM declares that the agent has finished its task."""


class _LimitsExceededError(_InterruptAgentFlowError):
    """Raised when the agent has reached its cost or step limit."""


@dataclasses.dataclass(frozen=True)
class _AgentOutput:
    """Output from executing a bash command in the container."""

    returncode: int
    output: str
    exception_info: str | None = None


@dataclasses.dataclass(frozen=True)
class _Action:
    """Parsed action from a tool call."""

    command: str
    tool_call_id: str


@dataclasses.dataclass(kw_only=True)
class MiniSWECodeAgentV2(code_agent_base.CodeAgent):
    """V2 tool-call based mini-swe-agent implementation.

    This agent uses OpenAI-style tool calls where:
    - Commands are specified via the bash tool (BASH_TOOL)
    - Multiple commands can be executed in parallel per turn
    - Observations use role="tool" with tool_call_id
    - The agent submits by echoing a submission sentinel

    All templates and behavior can be customized via constructor parameters.
    """

    # Required dependencies
    container: containers.Container
    llm_client: llm_clients.LLMClient
    tracker: stat_tracker.StatTracker = dataclasses.field(default_factory=stat_tracker.NullStatTracker)

    # Limits and behavior
    step_limit: int = 0  # 0 means no limit
    cost_limit: float = 3.0  # Default $3 limit
    timeout_s: int | None = None  # Command execution timeout
    temperature: float = 0.0  # LLM temperature
    parallel_tool_calls: bool = True  # Enable parallel tool calls

    # Templates (configurable for different benchmarks)
    system_template: str = _V2_SYSTEM_TEMPLATE
    instance_template: str = _V2_INSTANCE_TEMPLATE
    observation_template: str = _V2_OBSERVATION_TEMPLATE
    format_error_template: str = _V2_FORMAT_ERROR_TEMPLATE

    # Environment and execution
    env_vars: dict[str, str] = dataclasses.field(default_factory=lambda: _V2_DEFAULT_ENV_VARS.copy())
    working_directory: str | None = None  # None = use container default

    # Action parsing and submission
    submission_sentinel: str = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

    def __post_init__(self) -> None:
        """Initialize the agent with configuration."""
        # Initialize step and cost tracking
        self._n_calls = 0
        self._total_cost = 0.0

        # Render system prompt once using configurable template
        self._system_prompt = jinja2.Template(self.system_template, undefined=jinja2.StrictUndefined).render()
        self._messages: list[request.Message] = []

        _LOGGER.debug("[%d] Initialized MiniSWECodeAgentV2.", id(self))

    def _add_message(
        self, role: Literal["user", "assistant", "tool"], content: str, *, tool_call_id: str | None = None
    ) -> None:
        """Add a message to the conversation history."""
        if role == "user":
            self._messages.append(request.UserMessage(role="user", content=content))
        elif role == "assistant":
            self._messages.append(request.AssistantMessage(role="assistant", content=content))
        elif role == "tool":
            if tool_call_id is None:
                raise ValueError("tool_call_id is required for tool messages")
            self._messages.append(
                request.ToolCallResponseMessage(role="tool", content=content, tool_call_id=tool_call_id)
            )
        else:
            assert_never(role)

    def _add_assistant_message_with_tool_calls(
        self, content: str | None, tool_calls: list[response.ToolUseData]
    ) -> None:
        """Add an assistant message with tool calls to the conversation history.

        In the Chat Completions format, tool calls are embedded in the assistant message.
        """
        # Create the assistant message
        assistant_msg: request.AssistantMessage = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content

        # Add tool_calls if present
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _dict_to_json(tc.input),
                    },
                }
                for tc in tool_calls
            ]

        self._messages.append(assistant_msg)

    async def run(self, task: str) -> None:
        """Run the agent loop until completion or termination.

        Args:
            task: The task description/problem statement to solve.
        """
        with self.tracker.timeit("mswea_v2/setup"):
            _LOGGER.debug("[%d] Starting MiniSWECodeAgentV2 run.", id(self))

            # Determine working directory
            workdir = self.working_directory or "/"

            # Add initial task message
            instance_message = jinja2.Template(self.instance_template, undefined=jinja2.StrictUndefined).render(
                task=task,
                working_directory=workdir,
            )
            self._add_message("user", instance_message)

        # Main agent loop
        while True:
            try:
                with self.tracker.timeit("mswea_v2/step"):
                    await self.step()
            except _InterruptAgentFlowError as e:
                _LOGGER.debug("[%d] Agent flow interrupted: %s", id(self), type(e).__name__)
                # Add the interrupt messages to the conversation
                self._messages.extend(e.messages)
                # Check if we should terminate
                if isinstance(e, (_SubmittedError, _LimitsExceededError)):
                    return

    async def step(self) -> None:
        """Execute one step of the agent loop: query LLM and execute actions."""
        llm_response = await self.query()
        await self.execute_actions(llm_response)

    async def query(self) -> response.LLMResponse:
        """Query the LLM and return the response."""
        # Check step limit before making LLM call
        if 0 < self.step_limit <= self._n_calls:
            raise _LimitsExceededError(
                [
                    {
                        "role": "user",
                        "content": f"Step limit of {self.step_limit} exceeded",
                    }
                ]
            )

        # Check cost limit before making LLM call
        if 0 < self.cost_limit <= self._total_cost:
            raise _LimitsExceededError(
                [
                    {
                        "role": "user",
                        "content": f"Cost limit of ${self.cost_limit:.2f} exceeded",
                    }
                ]
            )

        _LOGGER.debug("[%d] Querying LLM (call #%d).", id(self), self._n_calls + 1)

        with self.tracker.timeit("mswea_v2/llm_request"):
            llm_response = await self.llm_client(
                request.LLMRequest(
                    messages=self._messages,
                    system_prompt=self._system_prompt,
                    temperature=self.temperature,
                    tools=[BASH_TOOL],
                    tool_choice="any",  # Must use at least one tool
                )
            )

        _LOGGER.debug("[%d] LLM response received.", id(self))

        # Update tracking
        self._n_calls += 1
        self._total_cost += llm_response.cost

        return llm_response

    async def execute_actions(self, llm_response: response.LLMResponse) -> None:
        """Parse and execute actions from the LLM response.

        This method:
        1. Extracts tool calls from the response
        2. Validates and parses them into actions
        3. Executes each action in the container
        4. Formats observation messages with tool_call_id
        """
        _LOGGER.debug("[%d] Executing actions.", id(self))

        # Parse actions from tool calls
        actions = self._parse_actions(llm_response)

        # Extract text content for the assistant message
        text_content = self._extract_text_content(llm_response)

        # Extract tool use data for the assistant message
        tool_calls = [block for block in llm_response.data if isinstance(block, response.ToolUseData)]

        # Add the assistant message with tool calls
        self._add_assistant_message_with_tool_calls(text_content, tool_calls)

        # Execute each action and collect outputs
        # Note: _check_submission raises _SubmittedError on submission, stopping execution
        outputs: list[tuple[_Action, _AgentOutput]] = []
        for action in actions:
            with self.tracker.timeit("mswea_v2/exec_run"):
                output = await self._execute_single_action(action)
            outputs.append((action, output))

            # Check for submission - raises _SubmittedError if detected
            self._check_submission(output)

        # Format and add observation messages for all actions
        for action, output in outputs:
            observation = self._format_observation(output)
            self._add_message("tool", observation, tool_call_id=action.tool_call_id)

    def _parse_actions(self, llm_response: response.LLMResponse) -> list[_Action]:
        """Parse actions from tool calls in the LLM response.

        Args:
            llm_response: The LLM response containing tool calls.

        Returns:
            List of parsed actions.

        Raises:
            _FormatError: If no valid tool calls are found or tool calls are invalid.
        """
        tool_calls = [block for block in llm_response.data if isinstance(block, response.ToolUseData)]

        if not tool_calls:
            error_msg = "No tool calls found in the response. Every response MUST include at least one tool call."
            format_error_str = jinja2.Template(self.format_error_template, undefined=jinja2.StrictUndefined).render(
                error=error_msg
            )
            raise _FormatError([{"role": "user", "content": format_error_str}])

        actions: list[_Action] = []
        for tool_call in tool_calls:
            # Validate tool name
            if tool_call.name != "bash":
                error_msg = f"Unknown tool '{tool_call.name}'. Only 'bash' tool is supported."
                format_error_str = jinja2.Template(self.format_error_template, undefined=jinja2.StrictUndefined).render(
                    error=error_msg
                )
                raise _FormatError([{"role": "user", "content": format_error_str}])

            # Validate command argument
            command = tool_call.input.get("command")
            if not command:
                error_msg = "Missing 'command' argument in bash tool call."
                format_error_str = jinja2.Template(self.format_error_template, undefined=jinja2.StrictUndefined).render(
                    error=error_msg
                )
                raise _FormatError([{"role": "user", "content": format_error_str}])

            actions.append(_Action(command=str(command), tool_call_id=tool_call.id))

        return actions

    def _extract_text_content(self, llm_response: response.LLMResponse) -> str | None:
        """Extract text content from the LLM response."""
        for block in llm_response.data:
            if isinstance(block, response.TextData):
                return block.content
        return None

    async def _execute_single_action(self, action: _Action) -> _AgentOutput:
        """Execute a single action in the container."""
        try:
            exec_result = await self.container.exec_run(
                action.command,
                workdir=self.working_directory,
                timeout_s=self.timeout_s,
                env=self.env_vars,
            )
            return _AgentOutput(
                returncode=exec_result.exit_code,
                output=exec_result.output,
                exception_info=None,
            )
        except Exception as e:
            # Catch execution errors including timeouts
            return _AgentOutput(
                returncode=-1,
                output="",
                exception_info=f"An error occurred while executing the command: {e}",
            )

    def _format_observation(self, output: _AgentOutput) -> str:
        """Format the observation message from execution output."""
        return jinja2.Template(self.observation_template, undefined=jinja2.StrictUndefined).render(output=output)

    def _check_submission(self, output: _AgentOutput) -> None:
        """Check if the output indicates submission and raise _SubmittedError if so.

        Args:
            output: The execution output to check.

        Raises:
            _SubmittedError: If the output contains the submission sentinel with exit code 0.
        """
        lines = output.output.lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == self.submission_sentinel and output.returncode == 0:
            # Everything after the first line is the final output (the patch)
            submission = "".join(lines[1:])
            _LOGGER.info("[%d] Agent submitted final output.", id(self))
            # Raise with an exit message
            raise _SubmittedError(
                [
                    {
                        "role": "user",
                        "content": submission,
                    }
                ]
            )


def _dict_to_json(d: dict) -> str:
    """Convert a dictionary to a JSON string."""
    import json

    return json.dumps(d)
