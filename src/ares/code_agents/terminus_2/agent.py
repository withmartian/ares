"""Terminus 2 Code Agent implementation WITHOUT tmux.

This is identical to Terminus2Agent except it uses direct command execution
instead of tmux.

Adapted from Harbor's Terminus 2 agent:
https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/terminus_2/terminus_2.py
"""

import dataclasses
import logging
import pathlib
import re
from typing import Literal, cast

from ares.code_agents import code_agent_base
from ares.code_agents.terminus_2 import json_parser
from ares.code_agents.terminus_2 import xml_parser
from ares.containers import containers
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
    """Terminus 2 agent using direct command execution.

    This agent executes commands directly via container.exec_run().

    Attributes:
        container: The container to execute commands in.
        llm_client: The LLM client for making requests.
        parser_format: The response format to use ("json" or "xml").
        max_turns: Maximum number of LLM interactions before stopping.
        timeout_s: Default timeout for command execution in seconds.
        enable_summarization: Enable context summarization when context limit is exceeded.
    """

    container: containers.Container
    llm_client: llm_clients.LLMClient
    parser_format: Literal["json", "xml"] = "json"
    max_turns: int = 50
    timeout_s: float = _DEFAULT_TIMEOUT_S
    enable_summarization: bool = True

    def __post_init__(self):
        """Initialize the agent."""
        # Load the appropriate parser
        if self.parser_format == "json":
            self._parser = json_parser.Terminus2JSONParser()
        elif self.parser_format == "xml":
            self._parser = xml_parser.Terminus2XMLParser()
        else:
            raise ValueError(f"Unknown parser format: {self.parser_format}. Use 'json' or 'xml'.")

        # Load the prompt template
        template_dir = pathlib.Path(__file__).parent / "templates"
        if self.parser_format == "json":
            template_path = template_dir / "terminus-json-plain.txt"
        else:
            template_path = template_dir / "terminus-xml-plain.txt"

        self._prompt_template = template_path.read_text()
        self._timeout_template = (template_dir / "timeout.txt").read_text()
        self._summarize_template = (template_dir / "summarize.txt").read_text()

        # Conversation history
        self._messages: list[request.Message] = []

        # State tracking
        self._turn_count = 0
        self._pending_completion = False
        self._original_instruction: str | None = None  # Store for summarization
        self._summarization_count: int = 0  # Track number of summarizations
        self._subagent_metrics = SubagentMetrics()  # Track subagent metrics separately
        self._last_output: str = ""  # Track last command output for summarization

        _LOGGER.debug("[%d] Initialized Terminus2Agent with %s format.", id(self), self.parser_format)

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
                parsed, feedback = self._parser.parse(assistant_message)

                # Handle parsing errors
                if feedback:
                    _LOGGER.warning("[%d] Parsing error: %s", id(self), feedback)
                    # Log the actual response that failed to parse (truncated for readability)
                    _LOGGER.warning("[%d] Failed response (first 2000 chars): %s", id(self), assistant_message[:2000])
                    self._add_message("user", feedback)
                    continue

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
        """Execute a list of commands and return the combined output.

        Args:
            commands: List of commands to execute.

        Returns:
            The combined output from all commands.
        """
        if not commands:
            return "(no commands executed)"

        outputs = []

        for i, cmd in enumerate(commands):
            # Skip empty commands
            if not cmd.keystrokes or not cmd.keystrokes.strip():
                _LOGGER.warning("[%d] Skipping empty command %d/%d", id(self), i + 1, len(commands))
                continue

            _LOGGER.debug("[%d] Executing command %d/%d: %s", id(self), i + 1, len(commands), cmd.keystrokes[:100])

            try:
                # Execute command directly (NO TMUX)
                # Use the duration as timeout, with a minimum and maximum
                timeout = max(5.0, min(cmd.duration, self.timeout_s))

                result = await self.container.exec_run(
                    cmd.keystrokes,
                    timeout_s=timeout,
                    workdir="/testbed",
                )

                # Format output like a terminal would show it
                output_text = f"$ {cmd.keystrokes}\n{result.output}"
                if result.exit_code != 0:
                    output_text += f"\n(exit code: {result.exit_code})"

                outputs.append(output_text)
                self._last_output = result.output

            except TimeoutError:
                _LOGGER.warning("[%d] Timeout executing command %d: %s", id(self), i + 1, cmd.keystrokes[:100])
                timeout_msg = self._timeout_template.format(
                    timeout=timeout,
                    command=cmd.keystrokes,
                    output=self._last_output or "(no output)",
                )
                outputs.append(timeout_msg)

            except Exception as e:
                _LOGGER.error("[%d] Error executing command %d: %s", id(self), i + 1, e)
                error_msg = f"$ {cmd.keystrokes}\n(error: {e})"
                outputs.append(error_msg)

        # Combine all outputs
        combined_output = "\n\n".join(outputs)

        # Limit output length
        return self._limit_output_length(combined_output)

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
            current_screen=self._limit_output_length(current_screen),
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


@dataclasses.dataclass(kw_only=True)
class Terminus2AgentFactory(code_agent_base.CodeAgentFactory):
    """Factory for creating Terminus2Agent instances."""

    parser_format: Literal["json", "xml"] = "json"
    max_turns: int = 50
    timeout_s: float = _DEFAULT_TIMEOUT_S
    enable_summarization: bool = True

    def __call__(self, container: containers.Container, llm_client: llm_clients.LLMClient) -> Terminus2Agent:
        """Create a new Terminus2Agent instance.

        Args:
            container: The container to use.
            llm_client: The LLM client to use.

        Returns:
            A new Terminus2Agent instance.
        """
        return Terminus2Agent(
            container=container,
            llm_client=llm_client,
            parser_format=self.parser_format,
            max_turns=self.max_turns,
            timeout_s=self.timeout_s,
            enable_summarization=self.enable_summarization,
        )
