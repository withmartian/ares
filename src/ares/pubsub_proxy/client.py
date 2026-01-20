"""Client for communicating with the PubSub proxy from the local machine.

This client is used by the local RL training loop to consume LLM requests
from agents and provide responses. It replaces QueueMediatedLLMClient for
distributed multi-container scenarios.
"""

import asyncio
import dataclasses
import logging
import time
from typing import Any

import httpx

from ares import async_utils
from ares.llms import llm_clients

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PubSubMediatedLLMClient:
    """LLM client that communicates with agents via HTTP PubSub proxy.

    This client polls the PubSub proxy for LLM requests from agents and
    submits responses. It's designed to work similarly to QueueMediatedLLMClient
    but for distributed scenarios where agents run in separate containers.

    The client maintains an async queue that the Environment can consume from,
    just like QueueMediatedLLMClient, providing a drop-in replacement.

    Attributes:
        proxy_url: Base URL of the PubSub proxy (e.g., http://localhost:8000)
        poll_timeout: Timeout for long-polling requests endpoint (seconds)
        q: Queue for exposing requests to the Environment
    """

    proxy_url: str
    poll_timeout: float = 30.0
    q: asyncio.Queue[async_utils.ValueAndFuture[llm_clients.LLMRequest, llm_clients.LLMResponse]] = dataclasses.field(
        default_factory=asyncio.Queue
    )

    def __post_init__(self) -> None:
        # Use object.__setattr__ since this is a frozen dataclass
        object.__setattr__(self, "_polling_task", None)
        object.__setattr__(self, "_pending_requests", {})
        object.__setattr__(self, "_http_client", httpx.AsyncClient(timeout=httpx.Timeout(60.0)))

    async def start_polling(self) -> None:
        """Start background task that polls proxy for requests.

        This should be called once the proxy container is ready. The polling
        task runs in the background and automatically puts requests into the
        queue for the Environment to consume.
        """
        if self._polling_task is not None:
            _LOGGER.warning("Polling already started")
            return

        _LOGGER.info("Starting polling loop for proxy at %s", self.proxy_url)
        polling_task = asyncio.create_task(self._polling_loop())
        object.__setattr__(self, "_polling_task", polling_task)

    async def stop_polling(self) -> None:
        """Stop the background polling task."""
        if self._polling_task is None:
            return

        _LOGGER.info("Stopping polling loop")
        self._polling_task.cancel()
        try:
            await self._polling_task
        except asyncio.CancelledError:
            pass

        object.__setattr__(self, "_polling_task", None)

    async def _polling_loop(self) -> None:
        """Background task that continuously polls for requests.

        This runs indefinitely, polling the proxy's /requests endpoint
        and putting received requests into the queue.
        """
        while True:
            try:
                # Poll for a request with timeout
                response = await self._http_client.get(
                    f"{self.proxy_url}/requests",
                    params={"timeout": self.poll_timeout},
                )

                if response.status_code == 200:
                    request_data = response.json()

                    # None response means timeout (no requests available)
                    if request_data is None:
                        continue

                    await self._handle_request(request_data)
                else:
                    _LOGGER.warning("Polling failed with status %d: %s", response.status_code, response.text)
                    await asyncio.sleep(1.0)  # Back off on error

            except httpx.ConnectError:
                _LOGGER.error("Failed to connect to proxy at %s", self.proxy_url)
                await asyncio.sleep(5.0)  # Back off on connection error
            except asyncio.CancelledError:
                _LOGGER.info("Polling loop cancelled")
                raise
            except Exception as e:
                _LOGGER.exception("Error in polling loop: %s", e)
                await asyncio.sleep(1.0)

    async def _handle_request(self, request_data: dict[str, Any]) -> None:
        """Process a received request and put it in the queue.

        Args:
            request_data: Request data from proxy containing request_id, messages, etc.
        """
        request_id = request_data["request_id"]
        _LOGGER.debug("Received request %s from proxy", request_id)

        # Create LLM request from the data
        llm_request = llm_clients.LLMRequest(
            messages=request_data["messages"],
            temperature=request_data.get("temperature"),
        )

        # Create future for the response
        future: asyncio.Future[llm_clients.LLMResponse] = asyncio.Future()
        self._pending_requests[request_id] = future

        # Put in queue for Environment to consume
        await self.q.put(async_utils.ValueAndFuture(value=llm_request, future=future))

        # Wait for response and send to proxy
        try:
            response = await future
            await self._send_response(request_id, response)
        except Exception as e:
            _LOGGER.exception("Error handling response for request %s: %s", request_id, e)
        finally:
            self._pending_requests.pop(request_id, None)

    async def _send_response(self, request_id: str, response: llm_clients.LLMResponse) -> None:
        """Send a response back to the proxy.

        Args:
            request_id: ID of the request being responded to
            response: LLM response to send
        """
        _LOGGER.debug("Sending response for request %s to proxy", request_id)

        # Convert LLMResponse to OpenAI-compatible format
        response_data = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.chat_completion_response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.chat_completion_response.choices[0].message.content,
                    },
                    "finish_reason": response.chat_completion_response.choices[0].finish_reason,
                }
            ],
            "usage": (
                {
                    "prompt_tokens": response.chat_completion_response.usage.prompt_tokens,
                    "completion_tokens": response.chat_completion_response.usage.completion_tokens,
                    "total_tokens": response.chat_completion_response.usage.total_tokens,
                }
                if response.chat_completion_response.usage
                else None
            ),
        }

        try:
            result = await self._http_client.post(
                f"{self.proxy_url}/responses/{request_id}",
                json=response_data,
            )

            if result.status_code != 200:
                _LOGGER.error("Failed to send response for %s: %s", request_id, result.text)
        except Exception as e:
            _LOGGER.exception("Error sending response for %s: %s", request_id, e)

    async def close(self) -> None:
        """Clean up resources."""
        await self.stop_polling()
        await self._http_client.aclose()
