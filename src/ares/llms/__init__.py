"""LLM client interfaces and data types.

Canonical request builders and request types live in :mod:`ares.llms.open_responses`.

Prefer ``from ares.llms import open_responses`` to access request types and builders
rather than importing individual type aliases from this package.
"""

from ares.llms.chat_completions_compatible import ChatCompletionCompatibleLLMClient
from ares.llms.llm_clients import LLMClient
from ares.llms.response import InferenceResult
from ares.llms.response import extract_text_content
from ares.llms.response import make_response

__all__ = [
    "ChatCompletionCompatibleLLMClient",
    "InferenceResult",
    "LLMClient",
    "extract_text_content",
    "make_response",
]
