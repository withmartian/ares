"""Terminus 2 Code Agent implementation using tmux.

Adapted from Harbor's Terminus 2 agent:
https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/terminus_2/terminus_2.py

Uses tmux for persistent terminal sessions to maintain shell state across commands.
"""

import asyncio
import dataclasses
import functools
import logging
import pathlib
import re
import shlex
from typing import Literal, cast

from ares.code_agents import code_agent_base
from ares.code_agents.terminus2 import json_parser
from ares.code_agents.terminus2 import xml_parser
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.llms import request

_LOGGER = logging.getLogger(__name__)

# Default timeout for command execution
_DEFAULT_TIMEOUT_S = 180.0

# Maximum output length to prevent overwhelming the context
_MAX_OUTPUT_BYTES = 200_000 * 3

_SUMMARY_PROMPT_TEMPLATE = """
You are about to hand off your work to another AI agent.
Please provide a comprehensive summary of what you have accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.

Be comprehensive and detailed. The next agent needs to understand everything that has happened so far in order to continue.
""".strip()  # noqa: E501

_QUESTIONS_PROMPT_TEMPLATE = """
You are picking up work from a previous AI agent on this task:

**Original Task:** {original_instruction}

**Summary from Previous Agent:**
{summary_content}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary) about the current state of the solution that are not answered in the summary from the prior agent. After you ask these questions you will be on your own, so ask everything you need to know.
""".strip()  # noqa: E501


@functools.cache
def _load_template(template_path: pathlib.Path) -> str:
    """Load a template file with caching to avoid repeated blocking I/O.

    Args:
        template_path: Path to the template file.

    Returns:
        The template content as a string.
    """
    return template_path.read_text()


def _sanitize_content(content: str) -> str:
    """Remove problematic characters from content.

    Args:
        content: The content to sanitize.

    Returns:
        Sanitized content with null bytes, control characters, and ANSI codes removed.
    """
    # Remove null bytes
    content = content.replace("\x00", "")

    # Remove ANSI escape sequences
    ansi_escape = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
    content = ansi_escape.sub("", content)

    # Remove other control characters except newline, tab, carriage return
    content = "".join(char for char in content if char in "\n\t\r" or ord(char) >= 32)

    return content


@dataclasses.dataclass
class SubagentMetrics:
    """Metrics for subagent operations (summarization, etc.)."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0


@dataclasses.dataclass(kw_only=True)
class Terminus2Agent(code_agent_base.CodeAgent):
    """Terminus 2 agent using tmux for persistent terminal sessions.

    This agent executes commands via a persistent tmux session to maintain
    shell state (working directory, environment variables, etc.) across commands.

    Attributes:
        container: The container to execute commands in.
        llm_client: The LLM client for making requests.
        parser_format: The response format to use ("json" or "xml").
        max_turns: Maximum number of LLM interactions before stopping.
        timeout_s: Default timeout for command execution in seconds.
        enable_summarization: Enable context summarization when context limit is exceeded.
        tmux_pane_width: Width of tmux pane in characters (default: 160).
        tmux_pane_height: Height of tmux pane in characters (default: 40).
    """

    container: containers.Container
    llm_client: llm_clients.LLMClient
    # TODO: Actually use the stat tracker in the agent.
    tracker: stat_tracker.StatTracker = dataclasses.field(default_factory=stat_tracker.NullStatTracker)
    parser_format: Literal["json", "xml"] = "json"
    max_turns: int = 50
    timeout_s: float = _DEFAULT_TIMEOUT_S
    enable_summarization: bool = True
    tmux_pane_width: int = 160
    tmux_pane_height: int = 40

    def __post_init__(self):
        """Initialize the agent."""
        # Load the appropriate parser
        if self.parser_format == "json":
            self._parser = json_parser.Terminus2JSONParser()
        elif self.parser_format == "xml":
            self._parser = xml_parser.Terminus2XMLParser()
        else:
            raise ValueError(f"Unknown parser format: {self.parser_format}. Use 'json' or 'xml'.")

        # Load the prompt template (cached to avoid repeated blocking I/O)
        template_dir = pathlib.Path(__file__).parent / "templates"
        if self.parser_format == "json":
            template_path = template_dir / "terminus-json-plain.txt"
        else:
            template_path = template_dir / "terminus-xml-plain.txt"

        self._prompt_template = _load_template(template_path)
        self._timeout_template = _load_template(template_dir / "timeout.txt")
        self._summarize_template = _load_template(template_dir / "summarize.txt")

        # Conversation history
        self._messages: list[request.Message] = []

        # State tracking
        self._turn_count = 0
        self._pending_completion = False
        self._original_instruction: str | None = None  # Store for summarization
        self._summarization_count: int = 0  # Track number of summarizations
        self._subagent_metrics = SubagentMetrics()  # Track subagent metrics separately
        self._last_output: str = ""  # Track last command output for summarization

        # Tmux session state
        self._tmux_session_name = f"terminus2_{id(self)}"
        self._tmux_initialized = False

        _LOGGER.debug("[%d] Initialized Terminus2Agent with %s format.", id(self), self.parser_format)

    async def _ensure_tmux_session(self) -> None:
        """Initialize tmux session on first use.

        Creates a detached tmux session with configured dimensions and sets up
        the working directory.
        """
        if self._tmux_initialized:
            _LOGGER.debug("[%d] Tmux session already initialized: %s", id(self), self._tmux_session_name)
            return

        # Check if tmux is installed, install if needed
        _LOGGER.debug("[%d] Checking if tmux is installed", id(self))
        check_result = await self.container.exec_run("which tmux", timeout_s=10.0)

        if check_result.exit_code != 0:
            _LOGGER.info("[%d] tmux not found, installing...", id(self))
            install_result = await self.container.exec_run(
                "apt-get update -qq && apt-get install -y -qq tmux",
                timeout_s=60.0,
            )
            if install_result.exit_code != 0:
                _LOGGER.error("[%d] Failed to install tmux: %s", id(self), install_result.output)
                raise RuntimeError(f"Could not install tmux: {install_result.output}")
            _LOGGER.info("[%d] tmux installed successfully", id(self))
        else:
            _LOGGER.debug("[%d] tmux is already installed", id(self))

        _LOGGER.info("[%d] Creating tmux session: %s", id(self), self._tmux_session_name)
        _LOGGER.debug("[%d] Tmux dimensions: %dx%d", id(self), self.tmux_pane_width, self.tmux_pane_height)

        # Create detached tmux session with specific dimensions
        try:
            result = await self.container.exec_run(
                f"tmux new-session -d -s {self._tmux_session_name} "
                f"-x {self.tmux_pane_width} -y {self.tmux_pane_height}",
                workdir="/testbed",
                timeout_s=10.0,
            )
            _LOGGER.debug("[%d] Tmux new-session result: exit_code=%s", id(self), result.exit_code)
            if result.exit_code != 0:
                _LOGGER.error("[%d] Failed to create tmux session: %s", id(self), result.output)
        except Exception as e:
            _LOGGER.error("[%d] Error creating tmux session: %s", id(self), e)
            raise

        # Change to testbed directory in the session
        try:
            result = await self.container.exec_run(
                f"tmux send-keys -t {self._tmux_session_name} 'cd /testbed' Enter",
                timeout_s=5.0,
            )
            _LOGGER.debug("[%d] Tmux cd command result: exit_code=%s", id(self), result.exit_code)
        except Exception as e:
            _LOGGER.warning("[%d] Error sending cd to tmux: %s", id(self), e)

        # Small delay to let session initialize
        await asyncio.sleep(0.2)

        self._tmux_initialized = True
        _LOGGER.info("[%d] Tmux session ready: %s", id(self), self._tmux_session_name)

    async def _cleanup_tmux_session(self) -> None:
        """Clean up tmux session on shutdown."""
        if not self._tmux_initialized:
            return

        try:
            _LOGGER.info("[%d] Killing tmux session: %s", id(self), self._tmux_session_name)
            await self.container.exec_run(
                f"tmux kill-session -t {self._tmux_session_name}",
                timeout_s=5.0,
            )
        except Exception as e:
            _LOGGER.warning("[%d] Error killing tmux session: %s", id(self), e)

    async def run(self, task: str) -> None:
        """Run the agent on the given task.

        Args:
            task: The task description/problem statement.
        """
        _LOGGER.info("[%d] Starting Terminus2Agent run.", id(self))
        self._original_instruction = task  # Store for summarization

        # Create initial prompt
        initial_prompt = self._prompt_template.format(
            instruction=task,
            terminal_state="(Starting in /testbed directory)",
        )

        # Add system message
        self._add_message("system", initial_prompt)

        # Main agent loop
        while self._turn_count < self.max_turns:
            self._turn_count += 1
            _LOGGER.info("[%d] Turn %d/%d", id(self), self._turn_count, self.max_turns)

            try:
                # Query the LLM
                try:
                    response = await self._query_llm()
                    assistant_message = response.chat_completion_response.choices[0].message.content
                    assert assistant_message is not None

                    self._add_message("assistant", assistant_message)

                except llm_clients.OutputLengthExceededError as e:
                    # Handle output length exceeded like the reference implementation
                    _LOGGER.info("[%d] Output length exceeded: %s", id(self), e)

                    truncated_response = e.truncated_response or "[TRUNCATED RESPONSE NOT AVAILABLE]"

                    # Try to salvage a valid response from the truncated output (XML only)
                    salvaged_response = None
                    if isinstance(self._parser, xml_parser.Terminus2XMLParser):
                        salvaged_response, _ = self._parser.salvage_truncated_response(truncated_response)

                    if salvaged_response:
                        # Valid response salvaged! Use it
                        _LOGGER.debug(
                            "[%d] Output exceeded length but found valid response. Using truncated version.", id(self)
                        )
                        assistant_message = salvaged_response
                        self._add_message("assistant", assistant_message)
                    else:
                        # Couldn't salvage - provide error feedback
                        error_msg = (
                            "ERROR!! NONE of the actions you just requested were performed "
                            "because you exceeded the maximum output length of 4096 tokens. "
                            "Your outputs must be less than 4096 tokens. Re-issue this request, "
                            "breaking it into chunks each of which is less than 4096 tokens."
                        )

                        # Add truncated response and error to history
                        self._add_message("assistant", truncated_response)
                        self._add_message("user", error_msg)
                        continue

                # Parse the response
                _LOGGER.debug(
                    "[%d] Parsing LLM response (format: %s, length: %d)",
                    id(self),
                    self.parser_format,
                    len(assistant_message),
                )
                parsed, feedback = self._parser.parse(assistant_message)

                # Handle parsing errors
                if feedback:
                    _LOGGER.warning("[%d] Parsing error: %s", id(self), feedback)
                    # Log the actual response that failed to parse (truncated for readability)
                    _LOGGER.warning("[%d] Failed response (first 2000 chars): %s", id(self), assistant_message[:2000])
                    self._add_message("user", feedback)
                    continue

                _LOGGER.debug(
                    "[%d] Parsed successfully: %d commands, task_complete=%s",
                    id(self),
                    len(parsed.commands),
                    parsed.task_complete,
                )

                # Execute commands
                if parsed.commands:
                    # Show all commands (truncated to 50 chars each for readability)
                    preview = [cmd.keystrokes[:50] for cmd in parsed.commands]
                    _LOGGER.info("[%d] Executing %d command(s): %s", id(self), len(parsed.commands), preview)
                    observation = await self._execute_commands(cast(list[json_parser.Command], parsed.commands))
                    # Ensure observation is not empty
                    if not observation.strip():
                        observation = "(no output)"
                else:
                    observation = "No commands provided."

                # Handle task completion
                if parsed.task_complete:
                    if self._pending_completion:
                        # Second time marking complete - actually finish
                        _LOGGER.info("[%d] Task marked complete (confirmed). Finishing.", id(self))
                        self._add_message("user", "Task marked as complete. Finishing execution.")
                        # Cleanup tmux session
                        await self._cleanup_tmux_session()
                        return
                    else:
                        # First time - ask for confirmation
                        _LOGGER.info("[%d] Task marked complete (first time). Asking for confirmation.", id(self))
                        self._pending_completion = True
                        confirmation_msg = self._get_completion_confirmation_message(observation)
                        self._add_message("user", confirmation_msg)
                        continue
                else:
                    # Not complete, reset pending flag
                    self._pending_completion = False

                # Add observation to conversation
                _LOGGER.debug(
                    "[%d] Observation length: %d chars, first 200: %s", id(self), len(observation), observation[:200]
                )
                self._add_message("user", observation)

            except Exception as e:
                _LOGGER.exception("[%d] Error during agent execution: %s", id(self), e)
                error_msg = f"Internal error occurred: {type(e).__name__}: {e}"
                self._add_message("user", error_msg)
                # Continue to give the agent a chance to recover

        _LOGGER.warning("[%d] Reached maximum turns (%d). Stopping.", id(self), self.max_turns)

        # Cleanup tmux session
        await self._cleanup_tmux_session()

    async def _query_llm(self) -> llm_clients.LLMResponse:
        """Query the LLM with the current conversation history.

        Returns:
            The LLM response.
        """
        _LOGGER.debug("[%d] Querying LLM with %d messages", id(self), len(self._messages))

        # Proactive summarization: Estimate token count and summarize if approaching limit
        # Rough estimate: 1 token ≈ 2 characters
        if self.enable_summarization and len(self._messages) > 3:
            total_chars = sum(len(str(msg.get("content", ""))) for msg in self._messages)
            estimated_tokens = total_chars // 2

            # Trigger summarization if we're approaching 200k tokens (leave buffer for response)
            if estimated_tokens > 200000:
                _LOGGER.warning(
                    "[%d] Proactively summarizing: ~%d estimated tokens exceeds safe threshold",
                    id(self),
                    estimated_tokens,
                )
                handoff_prompt = await self._summarize()
                self._add_message("user", handoff_prompt)

                # Recalculate after summarization
                total_chars_after = sum(len(str(msg.get("content", ""))) for msg in self._messages)
                estimated_tokens_after = total_chars_after // 4
                _LOGGER.info(
                    "[%d] After proactive summarization: ~%d estimated tokens", id(self), estimated_tokens_after
                )

        try:
            response = await self.llm_client(request.LLMRequest(messages=self._messages))
            _LOGGER.debug("[%d] Received LLM response", id(self))
            return response

        except Exception as e:
            # Check if it's a context length error
            error_str = str(e).lower()
            is_context_error = (
                "context_length_exceeded" in error_str
                or "context length" in error_str
                or "tokens exceed" in error_str
                or "input tokens exceed" in error_str
                or ("invalid request parameters" in error_str and len(self._messages) > 10)
            )

            if is_context_error and self.enable_summarization and len(self._messages) > 3:
                _LOGGER.warning(
                    "[%d] Context length exceeded (%d messages), attempting Harbor-style 3-step summarization",
                    id(self),
                    len(self._messages),
                )

                # Use Harbor's 3-step summarization process
                handoff_prompt = await self._summarize()

                # Add the handoff prompt as a user message
                self._add_message("user", handoff_prompt)

                _LOGGER.info("[%d] Retrying query with summarized context (%d messages)", id(self), len(self._messages))

                # Retry the query with summarized context
                response = await self.llm_client(request.LLMRequest(messages=self._messages))
                _LOGGER.debug("[%d] Received LLM response after summarization", id(self))
                return response

            # If not a context error or summarization failed/disabled, re-raise
            raise

    async def _execute_commands(self, commands: list[json_parser.Command]) -> str:
        """Execute a list of commands via tmux and return the terminal output.

        Args:
            commands: List of commands to execute.

        Returns:
            The terminal output after executing all commands.
        """
        if not commands:
            return "(no commands executed)"

        # Ensure tmux session exists
        await self._ensure_tmux_session()

        for i, cmd in enumerate(commands):
            # Skip empty commands
            if not cmd.keystrokes or not cmd.keystrokes.strip():
                _LOGGER.warning("[%d] Skipping empty command %d/%d", id(self), i + 1, len(commands))
                continue

            _LOGGER.debug("[%d] Sending to tmux %d/%d: %s", id(self), i + 1, len(commands), cmd.keystrokes[:100])

            try:
                # Send keystrokes to tmux session
                # Split on newlines: send each line literally (-l flag), then press Enter
                # This ensures commands execute (newline → Enter key press)
                lines = cmd.keystrokes.split("\n")
                _LOGGER.debug("[%d] Split keystroke into %d lines", id(self), len(lines))

                for line_idx, line in enumerate(lines):
                    # Skip the final empty string after trailing newline
                    if line_idx == len(lines) - 1 and not line:
                        _LOGGER.debug("[%d] Skipping empty final line", id(self))
                        continue

                    # Send the line literally (no key interpretation)
                    if line:
                        send_cmd = f"tmux send-keys -t {self._tmux_session_name} -l {shlex.quote(line)}"
                        _LOGGER.debug("[%d] Sending line %d: %s", id(self), line_idx, send_cmd[:100])
                        result = await self.container.exec_run(send_cmd, timeout_s=5.0)
                        if result.exit_code != 0:
                            _LOGGER.warning(
                                "[%d] send-keys returned exit_code=%s: %s", id(self), result.exit_code, result.output
                            )

                    # Press Enter after each line (except the last empty one)
                    if line_idx < len(lines) - 1:
                        enter_cmd = f"tmux send-keys -t {self._tmux_session_name} Enter"
                        _LOGGER.debug("[%d] Sending Enter after line %d", id(self), line_idx)
                        result = await self.container.exec_run(enter_cmd, timeout_s=5.0)
                        if result.exit_code != 0:
                            _LOGGER.warning(
                                "[%d] Enter returned exit_code=%s: %s", id(self), result.exit_code, result.output
                            )

                # Wait for command to complete (cap at max timeout)
                wait_time = min(cmd.duration, self.timeout_s)
                _LOGGER.debug("[%d] Waiting %.1fs for command to complete", id(self), wait_time)
                await asyncio.sleep(wait_time)

            except TimeoutError:
                _LOGGER.warning("[%d] Timeout sending command %d: %s", id(self), i + 1, cmd.keystrokes[:100])
                # Continue to next command - tmux session still alive

            except Exception as e:
                _LOGGER.error("[%d] Error sending command %d: %s", id(self), i + 1, e)
                # Continue to next command - tmux session may still be alive

        # Capture terminal output after all commands
        try:
            _LOGGER.debug("[%d] Capturing terminal output from tmux", id(self))
            result = await self.container.exec_run(
                f"tmux capture-pane -t {self._tmux_session_name} -p",
                timeout_s=5.0,
            )
            _LOGGER.debug(
                "[%d] Capture result: exit_code=%s, output_length=%d", id(self), result.exit_code, len(result.output)
            )
            terminal_output = result.output
            self._last_output = terminal_output

            # Log first 500 chars of output for debugging
            _LOGGER.debug("[%d] Terminal output (first 500 chars): %s", id(self), terminal_output[:500])

            # Limit output length (async to avoid blocking on large outputs)
            limited = await self._limit_output_length_async(terminal_output)
            _LOGGER.debug("[%d] Returning output (length: %d)", id(self), len(limited))
            return limited

        except Exception as e:
            _LOGGER.error("[%d] Error capturing tmux output: %s", id(self), e)
            # Return last known output or error message
            if self._last_output:
                _LOGGER.warning("[%d] Returning last known output", id(self))
                return self._last_output
            return f"(error capturing terminal output: {e})"

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: The role of the message sender ("system", "user", or "assistant").
            content: The message content.
        """
        # Sanitize content before adding
        sanitized = _sanitize_content(content)
        assert role in ("system", "user", "assistant")
        self._messages.append(cast(request.Message, {"role": role, "content": sanitized}))

    async def _summarize(self) -> str:
        """Create a summary of the agent's work using Harbor's 3-step process.

        Returns:
            The handoff prompt to continue with.
        """
        if not self._original_instruction or len(self._messages) == 0:
            return self._original_instruction or "Continue working on the task."

        # Increment summarization count
        self._summarization_count += 1

        _LOGGER.info("[%d] Starting 3-step summarization (count: %d)", id(self), self._summarization_count)

        # ===== STEP 1: Summary Generation =====
        summary_prompt = _SUMMARY_PROMPT_TEMPLATE.format(original_instruction=self._original_instruction)

        try:
            # Use the conversation history for context (same as Harbor's implementation)
            summary_messages = list(self._messages)
            summary_messages.append({"role": "user", "content": summary_prompt})

            summary_response = await self.llm_client(request.LLMRequest(messages=summary_messages))

            summary_content = summary_response.chat_completion_response.choices[0].message.content
            assert summary_content is not None

            # Track subagent metrics
            usage = summary_response.chat_completion_response.usage
            if usage:
                self._subagent_metrics.total_prompt_tokens += usage.prompt_tokens or 0
                self._subagent_metrics.total_completion_tokens += usage.completion_tokens or 0

            _LOGGER.info("[%d] Step 1/3: Summary generated", id(self))

        except Exception as e:
            error_str = str(e).lower()
            # If summarization itself hits context limit, we're in a bad state
            # Fall back to keeping as much recent context as possible
            if "context_length_exceeded" in error_str or "context length" in error_str or "tokens exceed" in error_str:
                _LOGGER.warning("[%d] Summarization hit context limit, using last 20 messages only", id(self))
                try:
                    summary_messages = [self._messages[0], *self._messages[-20:]]
                    summary_messages.append({"role": "user", "content": summary_prompt})
                    summary_response = await self.llm_client(request.LLMRequest(messages=summary_messages))
                    summary_content = summary_response.chat_completion_response.choices[0].message.content
                    assert summary_content is not None
                    usage = summary_response.chat_completion_response.usage
                    if usage:
                        self._subagent_metrics.total_prompt_tokens += usage.prompt_tokens or 0
                        self._subagent_metrics.total_completion_tokens += usage.completion_tokens or 0
                    _LOGGER.info("[%d] Step 1/3: Summary generated (with fallback)", id(self))
                except Exception as e2:
                    _LOGGER.error("[%d] Error generating summary even with fallback: %s", id(self), e2)
                    return self._original_instruction or "Continue working on the task."
            else:
                _LOGGER.error("[%d] Error generating summary: %s", id(self), e)
                return self._original_instruction or "Continue working on the task."

        # ===== STEP 2: Question Asking =====
        # Use last output instead of tmux screen
        current_screen = self._last_output or "(no recent output)"

        question_prompt = _QUESTIONS_PROMPT_TEMPLATE.format(
            original_instruction=self._original_instruction,
            summary_content=summary_content,
            current_screen=await self._limit_output_length_async(current_screen),
        )

        try:
            questions_response = await self.llm_client(
                request.LLMRequest(messages=[{"role": "user", "content": question_prompt}])
            )

            model_questions = questions_response.chat_completion_response.choices[0].message.content
            assert model_questions is not None

            # Track subagent metrics
            usage = questions_response.chat_completion_response.usage
            if usage:
                self._subagent_metrics.total_prompt_tokens += usage.prompt_tokens or 0
                self._subagent_metrics.total_completion_tokens += usage.completion_tokens or 0

            _LOGGER.info("[%d] Step 2/3: Questions generated", id(self))

        except Exception as e:
            _LOGGER.error("[%d] Error generating questions: %s", id(self), e)
            return self._original_instruction or "Continue working on the task."

        # ===== STEP 3: Answer Providing =====
        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each of them one by one in detail:\n\n"
            + model_questions
        )

        try:
            # Use the conversation history for context (same as Harbor's implementation)
            answer_messages = list(self._messages)
            answer_messages.append({"role": "user", "content": answer_request_prompt})

            answers_response = await self.llm_client(request.LLMRequest(messages=answer_messages))

            answers_content = answers_response.chat_completion_response.choices[0].message.content
            assert answers_content is not None

            # Track subagent metrics
            usage = answers_response.chat_completion_response.usage
            if usage:
                self._subagent_metrics.total_prompt_tokens += usage.prompt_tokens or 0
                self._subagent_metrics.total_completion_tokens += usage.completion_tokens or 0

            _LOGGER.info("[%d] Step 3/3: Answers provided", id(self))

        except Exception as e:
            error_str = str(e).lower()
            # If answer generation hits context limit, fall back to truncated history
            if "context_length_exceeded" in error_str or "context length" in error_str or "tokens exceed" in error_str:
                _LOGGER.warning("[%d] Answer generation hit context limit, using last 20 messages only", id(self))
                try:
                    answer_messages = [self._messages[0], *self._messages[-20:]]
                    answer_messages.append({"role": "user", "content": answer_request_prompt})
                    answers_response = await self.llm_client(request.LLMRequest(messages=answer_messages))
                    answers_content = answers_response.chat_completion_response.choices[0].message.content
                    assert answers_content is not None
                    usage = answers_response.chat_completion_response.usage
                    if usage:
                        self._subagent_metrics.total_prompt_tokens += usage.prompt_tokens or 0
                        self._subagent_metrics.total_completion_tokens += usage.completion_tokens or 0
                    _LOGGER.info("[%d] Step 3/3: Answers provided (with fallback)", id(self))
                except Exception as e2:
                    _LOGGER.error("[%d] Error providing answers even with fallback: %s", id(self), e2)
                    return self._original_instruction or "Continue working on the task."
            else:
                _LOGGER.error("[%d] Error providing answers: %s", id(self), e)
                return self._original_instruction or "Continue working on the task."

        # Create handoff prompt that includes the full context
        handoff_prompt = (
            f"**Summary from Previous Agent:**\n{summary_content}\n\n"
            f"**Questions:**\n{model_questions}\n\n"
            f"**Answers:**\n{answers_content}\n\n"
            "Continue working on this task from where the previous agent left off. "
            "You can no longer ask questions. Please follow the spec to interact with the terminal."
        )

        # Update chat history: keep only system message
        self._messages = [
            self._messages[0],  # Keep system message with terminal instructions
        ]

        _LOGGER.info(
            "[%d] Summarization complete. Subagent tokens: prompt=%d, completion=%d",
            id(self),
            self._subagent_metrics.total_prompt_tokens,
            self._subagent_metrics.total_completion_tokens,
        )

        return handoff_prompt

    def _limit_output_length(self, output: str) -> str:
        """Limit output length to prevent overwhelming the context.

        Implementation matches terminal-bench reference: keep first and last portions,
        omitting the middle.

        Args:
            output: The output to limit.

        Returns:
            The potentially truncated output.
        """
        output_bytes = output.encode("utf-8")

        if len(output_bytes) <= _MAX_OUTPUT_BYTES:
            return output

        # Take first and last portions (half each)
        portion_size = _MAX_OUTPUT_BYTES // 8

        # Get first portion
        first_portion = output_bytes[:portion_size].decode("utf-8", errors="ignore")

        # Get last portion
        last_portion = output_bytes[-portion_size:].decode("utf-8", errors="ignore")

        # Calculate omitted bytes
        omitted_bytes = len(output_bytes) - len(first_portion.encode("utf-8")) - len(last_portion.encode("utf-8"))

        return (
            f"{first_portion}\n[... output limited to {_MAX_OUTPUT_BYTES} bytes; "
            f"{omitted_bytes} interior bytes omitted ...]\n{last_portion}"
        )

    async def _limit_output_length_async(self, output: str) -> str:
        """Async wrapper for _limit_output_length that offloads to thread pool for large outputs.

        For small outputs, runs synchronously. For large outputs (>50KB), offloads the
        encoding/decoding work to a thread pool to avoid blocking the event loop.

        Args:
            output: The output to limit.

        Returns:
            The potentially truncated output.
        """
        # For small outputs, run synchronously to avoid thread pool overhead
        if len(output) < 50_000:
            return self._limit_output_length(output)

        # For large outputs, offload to thread pool to avoid blocking event loop
        return await asyncio.to_thread(self._limit_output_length, output)

    def _get_completion_confirmation_message(self, terminal_output: str) -> str:
        """Get the message to confirm task completion.

        Args:
            terminal_output: The current terminal output.

        Returns:
            The confirmation message.
        """
        limited_output = self._limit_output_length(terminal_output)

        if self.parser_format == "json":
            return (
                f"Current terminal state:\n{limited_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                'make any further corrections. If so, include "task_complete": true '
                "in your JSON response again."
            )
        else:  # xml
            return (
                f"Current terminal state:\n{limited_output}\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded and you won't be able to "
                "make any further corrections. If so, include "
                "<task_complete>true</task_complete> again."
            )
