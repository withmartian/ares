"""LLM client interfaces and data types.

Canonical request builders and request types live in :mod:`ares.llms.open_responses`.
Legacy request compatibility helpers remain in :mod:`ares.llms.request`.
"""

from ares.llms.chat_completions_compatible import ChatCompletionCompatibleLLMClient
from ares.llms.llm_clients import LLMClient
from ares.llms.open_responses import MODEL_STUB
from ares.llms.open_responses import InputItemFunctionCall
from ares.llms.open_responses import InputItemFunctionCallOutput
from ares.llms.open_responses import InputItemMessage
from ares.llms.open_responses import OpenResponsesRequest
from ares.llms.open_responses import OpenResponsesResponse
from ares.llms.response import LLMResponse
from ares.llms.response import TextData
from ares.llms.response import Usage

__all__ = [
    "MODEL_STUB",
    "ChatCompletionCompatibleLLMClient",
    "InputItemFunctionCall",
    "InputItemFunctionCallOutput",
    "InputItemMessage",
    "LLMClient",
    "LLMResponse",
    "OpenResponsesRequest",
    "OpenResponsesResponse",
    "TextData",
    "Usage",
]
