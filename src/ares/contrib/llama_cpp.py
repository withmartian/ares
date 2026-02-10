"""Starter LLM client using llama-cpp-python for local CPU inference.

This is a simple reference implementation to help you get started with local models.
ARES focuses on providing consistent task interfaces - this is just a convenience utility.

WARNING: This is not a core ARES component. The interface may change without notice.
For production use, consider implementing your own LLM client following the LLMClient protocol.

Required dependency group:
    uv add withmartian-ares[llamacpp]
    OR
    uv sync --group llamacpp

Example usage:
    from ares.contrib import llama_cpp
    from ares.llms import llm_clients
from ares.llms import request

    # Initialize with a local GGUF model file
    client = llama_cpp.LlamaCppLLMClient(
        model_name="Qwen/Qwen2-0.5B-Instruct-GGUF",
        filename="*q8_0.gguf",
    )

    # Use like any other LLM client
    request = request.LLMRequest(messages=[{"role": "user", "content": "Hello!"}])
    response = await client(request)

Note: Download GGUF models from HuggingFace. For example:
    huggingface-cli download TheBloke/Llama-2-7B-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf
"""

import asyncio
import dataclasses
import functools
import logging

import llama_cpp
import openai.types.chat.chat_completion

from ares.llms import llm_clients
from ares.llms import openai_chat_converter
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class LlamaCppLLMClient(llm_clients.LLMClient):
    """Simple LLM client for local CPU inference using llama.cpp.

    This is a starter implementation - not part of core ARES. Use as a reference
    for building your own local inference clients.

    Attributes:
        model_name: huggingface ID. E.g. "Qwen/Qwen2.5-3B-Instruct"
        n_ctx: Context window size (default: 2048)
        n_threads: Number of CPU threads to use (default: None, uses all available)
        temperature: Default temperature for generation (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 512)
        verbose: Enable llama.cpp verbose logging (default: False)
    """

    model_name: str
    filename: str
    n_ctx: int = 2_048

    @functools.cached_property
    def _llm(self) -> llama_cpp.Llama:
        return llama_cpp.Llama.from_pretrained(
            self.model_name,
            filename=self.filename,
            verbose=False,
            n_ctx=self.n_ctx,
        )

    async def __call__(self, req: request.LLMRequest) -> response.LLMResponse:
        """Generate a response using llama.cpp.

        Args:
            request: The LLM request containing messages and optional temperature

        Returns:
            LLMResponse with the generated completion
        """
        _LOGGER.debug("[%d] Requesting LLM.", id(self))

        completion_kwargs = openai_chat_converter.to_external(req)
        # Since llama-cpp-python sets default temperature to 0.2, we explicitly
        # override it to 1.0 if it's not provided by the request.
        completion_kwargs.setdefault("temperature", 1.0)

        # Generate completion using llama.cpp's chat completion API
        chat_completion = await asyncio.to_thread(self._llm.create_chat_completion, **completion_kwargs)
        chat_completion = openai.types.chat.chat_completion.ChatCompletion.model_validate(chat_completion)

        _LOGGER.debug("[%d] LLM response received.", id(self))

        content = chat_completion.choices[0].message.content or ""
        usage = response.Usage(
            prompt_tokens=chat_completion.usage.prompt_tokens if chat_completion.usage else 0,
            generated_tokens=chat_completion.usage.completion_tokens if chat_completion.usage else 0,
        )
        return response.LLMResponse(data=[response.TextData(content=content)], cost=0.0, usage=usage)


create_qwen2_0_5b_instruct_llama_cpp_client = functools.partial(
    LlamaCppLLMClient, model_name="Qwen/Qwen2-0.5B-Instruct-GGUF", filename="*q8_0.gguf"
)
