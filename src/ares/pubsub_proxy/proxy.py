"""FastAPI-based proxy that bridges agent HTTP requests to local RL training loop.

This proxy runs in a container and provides an OpenAI-compatible API endpoint.
Agent containers make HTTP requests to this proxy, which queues them for the
local machine to consume. The local machine provides responses, which are
returned to the agents.

Architecture:
- Agents → HTTP POST /v1/chat/completions → Proxy
- Proxy → Queues request → Local machine polls GET /requests
- Local machine → HTTP POST /responses/{id} → Proxy
- Proxy → Returns response → Agent

All state is in-memory using asyncio primitives.
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_LOGGER = logging.getLogger(__name__)

# In-memory state
_request_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
_pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}

# Default timeout for agent requests waiting for responses (5 minutes)
DEFAULT_TIMEOUT_S = 300.0

app = FastAPI(title="ARES PubSub Proxy", version="0.1.0")


class ChatCompletionRequest(BaseModel):
    """Simplified OpenAI chat completion request."""

    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None


class LLMRequestMessage(BaseModel):
    """Message format sent to local machine via GET /requests."""

    request_id: str
    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
    """OpenAI-compatible chat completions endpoint.

    This is called by agents running in containers. The request is queued
    and the handler blocks until the local machine provides a response.

    Args:
        request: OpenAI-compatible chat completion request

    Returns:
        OpenAI-compatible chat completion response

    Raises:
        HTTPException: If timeout waiting for response or other error
    """
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    request_id = str(uuid.uuid4())
    _LOGGER.info("Received chat completion request %s", request_id)

    # Create future for this request
    future: asyncio.Future[dict[str, Any]] = asyncio.Future()
    _pending_responses[request_id] = future

    # Queue request for local machine to consume
    await _request_queue.put(
        {
            "request_id": request_id,
            "messages": request.messages,
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
    )

    _LOGGER.debug("Request %s queued, waiting for response...", request_id)

    # Wait for response with timeout
    try:
        response = await asyncio.wait_for(future, timeout=DEFAULT_TIMEOUT_S)
        _LOGGER.info("Request %s received response", request_id)
        return JSONResponse(content=response)
    except asyncio.TimeoutError:
        _LOGGER.error("Request %s timed out waiting for response", request_id)
        raise HTTPException(status_code=504, detail="Timeout waiting for LLM response")
    finally:
        # Clean up
        _pending_responses.pop(request_id, None)


@app.get("/requests")
async def get_requests(timeout: float = 30.0) -> LLMRequestMessage | None:
    """Poll for pending LLM requests.

    This is called by the local machine to retrieve queued requests from agents.
    Uses long-polling - blocks until a request is available or timeout.

    Args:
        timeout: Maximum time to wait for a request (seconds)

    Returns:
        LLM request message if available, None if timeout
    """
    try:
        request_data = await asyncio.wait_for(_request_queue.get(), timeout=timeout)
        _LOGGER.debug("Returning request %s to local machine", request_data["request_id"])
        return LLMRequestMessage(**request_data)
    except asyncio.TimeoutError:
        # This is normal - long polling timeout
        return None


@app.post("/responses/{request_id}")
async def post_response(request_id: str, request: Request) -> dict[str, str]:
    """Submit a response for a pending request.

    This is called by the local machine to provide an LLM response for a
    queued request. The response is used to unblock the waiting agent.

    Args:
        request_id: ID of the request to respond to
        request: OpenAI-compatible chat completion response

    Returns:
        Status message

    Raises:
        HTTPException: If request_id not found
    """
    response_data = await request.json()

    if request_id not in _pending_responses:
        _LOGGER.warning("Received response for unknown request %s", request_id)
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    _LOGGER.debug("Received response for request %s", request_id)

    # Resolve the future, which unblocks the agent's HTTP request
    future = _pending_responses[request_id]
    future.set_result(response_data)

    return {"status": "ok", "request_id": request_id}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pending_requests": str(_request_queue.qsize()),
        "pending_responses": str(len(_pending_responses)),
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with basic info."""
    return {
        "service": "ARES PubSub Proxy",
        "version": "0.1.0",
        "endpoints": {
            "agents": "POST /v1/chat/completions",
            "local_poll": "GET /requests?timeout=30",
            "local_respond": "POST /responses/{request_id}",
            "health": "GET /health",
        },
    }
