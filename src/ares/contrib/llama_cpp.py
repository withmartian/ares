"""Starter LLM client using llama-cpp-python for local CPU inference.

This is a simple reference implementation to help you get started with local models.
ARES focuses on providing consistent task interfaces - this is just a convenience utility.

WARNING: This is not a core ARES component. The interface may change without notice.
For production use, consider implementing your own LLM client following the LLMClient protocol.

Required dependency group:
    uv sync --group contrib-llamacpp

Example usage:
    from ares.contrib import llama_cpp
    from ares.llms import llm_clients

    # Initialize with a local GGUF model file
    client = llama_cpp.LlamaCppLLMClient(
        model_path="/path/to/model.gguf",
        n_ctx=2048,  # Context window size
        n_threads=4,  # CPU threads to use
    )

    # Use like any other LLM client
    request = llm_clients.LLMRequest(messages=[{"role": "user", "content": "Hello!"}])
    response = await client(request)

Note: Download GGUF models from HuggingFace. For example:
    huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf
"""

import dataclasses
import logging
import time
import uuid

import llama_cpp
import openai.types.chat.chat_completion
import openai.types.chat.chat_completion_message
import openai.types.completion_usage

from ares.llms import llm_clients

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class LlamaCppLLMClient(llm_clients.LLMClient):
    """Simple LLM client for local CPU inference using llama.cpp.

    This is a starter implementation - not part of core ARES. Use as a reference
    for building your own local inference clients.

    Attributes:
        model_path: Path to the GGUF model file
        n_ctx: Context window size (default: 2048)
        n_threads: Number of CPU threads to use (default: None, uses all available)
        temperature: Default temperature for generation (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 512)
        verbose: Enable llama.cpp verbose logging (default: False)
    """

    model_path: str
    n_ctx: int = 2048
    n_threads: int | None = None
    temperature: float = 0.7
    max_tokens: int = 512
    verbose: bool = False

    def __post_init__(self):
        """Initialize lazy loading placeholder."""
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "_llm", None)

    def _get_llm(self) -> llama_cpp.Llama:
        """Get or create the llama.cpp model instance (lazy loading)."""
        if self._llm is None:
            _LOGGER.info("Loading llama.cpp model from %s", self.model_path)
            llm = llama_cpp.Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=self.verbose,
            )
            object.__setattr__(self, "_llm", llm)
            _LOGGER.info("Model loaded successfully")
        return self._llm

    async def __call__(self, request: llm_clients.LLMRequest) -> llm_clients.LLMResponse:
        """Generate a response using llama.cpp.

        Args:
            request: The LLM request containing messages and optional temperature

        Returns:
            LLMResponse with the generated completion
        """
        _LOGGER.debug("[%d] Requesting LLM.", id(self))

        llm = self._get_llm()

        # Use request temperature if provided, otherwise use instance default
        temperature = request.temperature if request.temperature is not None else self.temperature

        # Convert messages to llama.cpp format
        messages = list(request.messages)

        # Generate completion using llama.cpp's chat completion API
        # llama-cpp-python returns OpenAI-compatible dict
        response_dict = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
        )

        _LOGGER.debug("[%d] LLM response received.", id(self))

        # Convert to OpenAI ChatCompletion type
        chat_completion = openai.types.chat.chat_completion.ChatCompletion(
            id=response_dict.get("id", str(uuid.uuid4())),
            choices=[
                openai.types.chat.chat_completion.Choice(
                    message=openai.types.chat.chat_completion_message.ChatCompletionMessage(
                        content=response_dict["choices"][0]["message"]["content"],
                        role=response_dict["choices"][0]["message"]["role"],
                    ),
                    finish_reason=response_dict["choices"][0].get("finish_reason", "stop"),
                    index=0,
                )
            ],
            created=response_dict.get("created", int(time.time())),
            model=response_dict.get("model", self.model_path),
            object="chat.completion",
            usage=openai.types.completion_usage.CompletionUsage(
                prompt_tokens=response_dict["usage"]["prompt_tokens"],
                completion_tokens=response_dict["usage"]["completion_tokens"],
                total_tokens=response_dict["usage"]["total_tokens"],
            ),
        )

        # Local inference has zero cost
        return llm_clients.LLMResponse(chat_completion_response=chat_completion, cost=0.0)
