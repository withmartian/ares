"""Unified LLM request abstraction supporting multiple API formats."""

from collections.abc import Iterable
import dataclasses
import logging
from typing import Any, Literal

import anthropic.types
import openai.types.chat.chat_completion_message_param
import openai.types.chat.completion_create_params
import openai.types.responses.response_create_params

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LLMRequest:
    """Unified request format for OpenAI Chat Completions, OpenAI Responses, and Claude Messages APIs.

    This class provides a common abstraction over three major LLM API formats, making it easy
    to convert between them. It includes the 9 core parameters that exist across all APIs.

    Core Parameters (all APIs):
        messages: List of messages in OpenAI Chat Completions format (will be converted as needed)
        max_output_tokens: Maximum tokens to generate (field name varies by API)
        temperature: Sampling temperature (range 0-2 for OpenAI, auto-converted to 0-1 for Claude)
        top_p: Nucleus sampling parameter
        stream: Enable streaming responses
        tools: Tool definitions (schema varies by API, will be converted)
        tool_choice: Tool selection strategy (options vary by API, will be converted)
        metadata: Custom key-value pairs (location varies by API)

    Extended Parameters (partial support):
        service_tier: Processing tier (options differ by API)
        stop_sequences: Stop sequences (not supported in OpenAI Responses)
        system_prompt: System instructions (location varies by API)
        top_k: Top-K sampling (Claude only)

    Note:
        - Model is NOT stored here - it should be managed by the LLMClient
        - Messages are stored in OpenAI Chat Completions format internally
        - Temperature is stored in OpenAI range (0-2), converted to Claude range (0-1) on export
        - Tool schemas are converted as needed for each API
        - Some parameters may be lost or unsupported when converting between APIs
    """

    messages: Iterable[openai.types.chat.chat_completion_message_param.ChatCompletionMessageParam]
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority", "standard_only"] | None = None
    stop_sequences: list[str] | None = None
    system_prompt: str | None = None
    top_k: int | None = None

    def to_chat_completion_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to OpenAI Chat Completions API format.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for openai.ChatCompletion.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - top_k is not supported (Claude-specific)
            - service_tier="standard_only" is not supported
            - stop_sequences truncated to 4 if more provided
        """
        # Check for information loss
        lost_info = []
        if self.top_k is not None:
            lost_info.append(f"top_k={self.top_k} (Claude-specific, not supported)")
        if self.service_tier == "standard_only":
            lost_info.append("service_tier='standard_only' (not supported by Chat API)")
        if self.stop_sequences and len(self.stop_sequences) > 4:
            lost_info.append(
                f"stop_sequences truncated from {len(self.stop_sequences)} to 4 "
                f"(Chat API limit: {self.stop_sequences[4:]} will be dropped)"
            )

        if lost_info:
            msg = f"Converting to Chat Completions will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        kwargs: dict[str, Any] = {
            "messages": list(self.messages),
        }

        # Add system prompt as first message if present
        if self.system_prompt:
            kwargs["messages"] = [
                {"role": "system", "content": self.system_prompt},
                *kwargs["messages"],
            ]

        # Add optional parameters (filter None values)
        if self.max_output_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.service_tier and self.service_tier != "standard_only":
            kwargs["service_tier"] = self.service_tier
        if self.stop_sequences:
            # OpenAI Chat supports up to 4 stop sequences
            kwargs["stop"] = self.stop_sequences[:4]

        return kwargs

    def to_responses_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to OpenAI Responses API format.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for openai.Responses.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - messages are converted to input items
            - system_prompt is mapped to instructions parameter
            - stop_sequences are not supported in Responses API
            - top_k is not supported (Claude-specific)
            - service_tier="standard_only" is not supported
        """
        # Check for information loss
        lost_info = []
        if self.stop_sequences:
            lost_info.append(f"stop_sequences={self.stop_sequences} (not supported by Responses API)")
        if self.top_k is not None:
            lost_info.append(f"top_k={self.top_k} (Claude-specific, not supported)")
        if self.service_tier == "standard_only":
            lost_info.append("service_tier='standard_only' (not supported by Responses API)")

        if lost_info:
            msg = f"Converting to Responses will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        kwargs: dict[str, Any] = {
            "input": self._messages_to_responses_input(),
        }

        if self.system_prompt:
            kwargs["instructions"] = self.system_prompt

        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            kwargs["tool_choice"] = self.tool_choice
        if self.metadata:
            kwargs["metadata"] = self.metadata
        if self.service_tier and self.service_tier != "standard_only":
            kwargs["service_tier"] = self.service_tier

        return kwargs

    def to_messages_kwargs(self, *, strict: bool = True) -> dict[str, Any]:
        """Convert to Claude Messages API format.

        Args:
            strict: If True, raise ValueError on information loss. If False, log warnings.

        Returns:
            Dictionary of kwargs for anthropic.messages.create() (without model)

        Raises:
            ValueError: If strict=True and information would be lost in conversion

        Note:
            - Model parameter is NOT included - it should be added by the LLMClient
            - temperature is converted from OpenAI range (0-2) to Claude range (0-1)
            - messages must alternate user/assistant (enforced by Claude API)
            - system_prompt is mapped to system parameter
            - service_tier options are limited to "auto" and "standard_only"
            - tool schemas may need conversion (not implemented yet)
        """
        # Check for information loss
        lost_info = []
        if self.service_tier not in (None, "auto", "standard_only"):
            lost_info.append(f"service_tier='{self.service_tier}' (Claude only supports 'auto' and 'standard_only')")

        # Check for filtered messages
        filtered_messages = []
        for msg in self.messages:
            msg_dict = dict(msg)
            role = msg_dict["role"]
            if role in ("system", "developer"):
                filtered_messages.append(f"{role} message: {msg_dict.get('content', '')[:50]}...")

        if filtered_messages:
            lost_info.append(f"Messages filtered out (use system_prompt instead): {'; '.join(filtered_messages)}")

        if lost_info:
            msg = f"Converting to Claude Messages will lose information: {'; '.join(lost_info)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        kwargs: dict[str, Any] = {
            "messages": self._messages_to_claude_format(),
            "max_tokens": self.max_output_tokens or 1024,  # max_tokens is required by Claude
        }

        if self.system_prompt:
            kwargs["system"] = self.system_prompt

        if self.temperature is not None:
            # Convert from OpenAI range (0-2) to Claude range (0-1)
            kwargs["temperature"] = min(self.temperature / 2.0, 1.0)
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.stream:
            kwargs["stream"] = True
        if self.tools:
            # TODO: Convert tool schemas from OpenAI to Claude format
            kwargs["tools"] = self.tools
        if self.tool_choice is not None:
            # TODO: Convert tool_choice from OpenAI to Claude format
            kwargs["tool_choice"] = self.tool_choice
        if self.metadata:
            # Claude uses metadata.user_id specifically
            kwargs["metadata"] = self.metadata
        if self.service_tier in ("auto", "standard_only"):
            kwargs["service_tier"] = self.service_tier
        if self.stop_sequences:
            kwargs["stop_sequences"] = self.stop_sequences

        return kwargs

    def _messages_to_responses_input(self) -> list[dict[str, Any]]:
        """Convert messages from Chat format to Responses input items.

        Returns:
            List of input items for Responses API
        """
        input_items = []
        for msg in self.messages:
            # Convert to EasyInputMessageParam format
            msg_dict = dict(msg)  # Convert to regular dict for type safety
            input_items.append(
                {
                    "type": "message",
                    "role": msg_dict["role"],
                    "content": msg_dict.get("content", ""),
                }
            )
        return input_items

    def _messages_to_claude_format(self) -> list[dict[str, Any]]:
        """Convert messages from Chat format to Claude alternating format.

        Returns:
            List of messages in Claude format (user/assistant alternating)

        Note:
            Claude requires strict alternation. This method filters out system/developer
            messages (should be in system_prompt) and ensures alternation.
        """
        claude_messages = []
        for msg in self.messages:
            msg_dict = dict(msg)  # Convert to regular dict for type safety
            role = msg_dict["role"]
            # Skip system/developer messages (should be in system_prompt)
            if role in ("system", "developer"):
                continue
            # Map tool/function to user role (tool results)
            if role in ("tool", "function"):
                role = "user"
            # Keep user/assistant as-is
            claude_messages.append(
                {
                    "role": role,
                    "content": msg_dict.get("content", ""),
                }
            )
        return claude_messages

    @classmethod
    def from_chat_completion(
        cls,
        kwargs: openai.types.chat.completion_create_params.CompletionCreateParamsBase,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Chat Completions API kwargs.

        Args:
            kwargs: OpenAI Chat Completions API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters

        Note:
            Model parameter is ignored - it should be managed by the LLMClient
        """
        # Define parameters we actually store/handle
        handled_params = {
            "model",  # Handled by being ignored (LLMClient manages this)
            "messages",
            "max_completion_tokens",
            "max_tokens",  # Fallback for max_output_tokens
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "stop",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Chat Completions parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        messages = list(kwargs["messages"])
        system_prompt = None

        # Extract system prompt from messages
        if messages and messages[0].get("role") in ("system", "developer"):
            system_prompt = messages[0]["content"]
            messages = messages[1:]

        return cls(
            messages=messages,
            max_output_tokens=kwargs.get("max_completion_tokens") or kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stream=kwargs.get("stream", False),
            tools=kwargs.get("tools"),  # type: ignore[arg-type]
            tool_choice=kwargs.get("tool_choice"),  # type: ignore[arg-type]
            metadata=kwargs.get("metadata"),
            service_tier=kwargs.get("service_tier"),  # type: ignore[arg-type]
            stop_sequences=kwargs.get("stop") if isinstance(kwargs.get("stop"), list) else None,
            system_prompt=system_prompt,  # type: ignore[arg-type]
        )

    @classmethod
    def from_responses(
        cls,
        kwargs: openai.types.responses.response_create_params.ResponseCreateParamsBase,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from OpenAI Responses API kwargs.

        Args:
            kwargs: OpenAI Responses API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters

        Note:
            Model parameter is ignored - it should be managed by the LLMClient
        """
        # Define parameters we actually store/handle
        handled_params = {
            "model",  # Handled by being ignored (LLMClient manages this)
            "input",
            "max_output_tokens",
            "temperature",
            "top_p",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "instructions",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Responses parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Convert input items to messages
        input_param = kwargs.get("input", [])
        messages: list[dict[str, Any]] = []
        if isinstance(input_param, str):
            messages = [{"role": "user", "content": input_param}]
        elif isinstance(input_param, list):
            for item in input_param:
                if item.get("type") == "message":
                    messages.append(
                        {
                            "role": item["role"],
                            "content": item["content"],
                        }
                    )

        return cls(
            messages=messages,  # type: ignore[arg-type]
            max_output_tokens=kwargs.get("max_output_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stream=kwargs.get("stream", False),
            tools=kwargs.get("tools"),  # type: ignore[arg-type]
            tool_choice=kwargs.get("tool_choice"),  # type: ignore[arg-type]
            metadata=kwargs.get("metadata"),
            service_tier=kwargs.get("service_tier"),  # type: ignore[arg-type]
            system_prompt=kwargs.get("instructions"),
        )

    @classmethod
    def from_messages(
        cls,
        kwargs: anthropic.types.MessageCreateParams,
        *,
        strict: bool = True,
    ) -> "LLMRequest":
        """Create LLMRequest from Claude Messages API kwargs.

        Args:
            kwargs: Claude Messages API parameters
            strict: If True, raise ValueError for unhandled parameters. If False, log warnings.

        Returns:
            LLMRequest instance

        Raises:
            ValueError: If strict=True and there are unhandled parameters
        """
        # Define parameters we handle (model is accepted but not stored)
        handled_params = {
            "model",  # Accepted but not stored - managed by LLMClient
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stream",
            "tools",
            "tool_choice",
            "metadata",
            "service_tier",
            "stop_sequences",
            "system",
        }

        # Check for unhandled parameters
        unhandled = set(kwargs.keys()) - handled_params
        if unhandled:
            msg = f"Unhandled Claude Messages parameters (will be ignored): {sorted(unhandled)}"
            if strict:
                raise ValueError(msg)
            _LOGGER.warning(msg)

        # Convert temperature from Claude range (0-1) to OpenAI range (0-2)
        temperature = kwargs.get("temperature")
        if temperature is not None:
            temperature = temperature * 2.0

        # Extract system prompt (can be str or list of text blocks)
        system_param = kwargs.get("system")
        system_prompt = None
        if isinstance(system_param, str):
            system_prompt = system_param
        elif isinstance(system_param, list) and system_param:
            # Extract text from first text block
            system_prompt = system_param[0].get("text", "") if isinstance(system_param[0], dict) else ""

        return cls(
            messages=kwargs["messages"],  # type: ignore[arg-type]
            max_output_tokens=kwargs["max_tokens"],
            temperature=temperature,
            top_p=kwargs.get("top_p"),
            top_k=kwargs.get("top_k"),
            stream=kwargs.get("stream", False),
            tools=kwargs.get("tools"),  # type: ignore[arg-type]
            tool_choice=kwargs.get("tool_choice"),  # type: ignore[arg-type]
            metadata=kwargs.get("metadata"),  # type: ignore[arg-type]
            service_tier=kwargs.get("service_tier"),  # type: ignore[arg-type]
            stop_sequences=kwargs.get("stop_sequences"),  # type: ignore[arg-type]
            system_prompt=system_prompt,
        )
