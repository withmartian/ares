"""Expose an in-sandbox OpenCode process as an ARES code agent."""

import asyncio
import contextlib
import dataclasses
import json
import logging
import pathlib
import tempfile
import time
from typing import Any, cast

import openai.types.responses.response_create_params

from ares.code_agents import ares_proxy
from ares.code_agents import opencode_cli
from ares.containers import containers
from ares.llms import llm_clients
from ares.llms import openai_responses_converter
from ares.llms import openai_responses_stream
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)

_MEDIATED_LOG_PATH = "/logs/agent/mediated.jsonl"
_POLL_INTERVAL_SECONDS = 0.25
_TITLE_SYSTEM_FINGERPRINT = "You are a title generator. You output ONLY a thread title."
_TITLE_USER_SENTINEL = "Generate a title for this conversation:\n"


def _to_llm_request(proxy_request: ares_proxy.PendingRequest) -> request.LLMRequest:
    if proxy_request.endpoint != "/v1/responses":
        raise ValueError(f"OpenCode must use /v1/responses, got {proxy_request.endpoint!r}")
    params = dict(proxy_request.request)
    params.pop("stream", None)
    return openai_responses_converter.from_external(
        cast(openai.types.responses.response_create_params.ResponseCreateParamsBase, params),
        strict=False,
    )


def _is_title_request(llm_request: request.LLMRequest) -> bool:
    """Identify OpenCode's hidden title request without relying on request order."""
    user_texts = [message.get("content") for message in llm_request.messages if message.get("role") == "user"]
    return (
        _TITLE_SYSTEM_FINGERPRINT in (llm_request.system_prompt or "")
        and not llm_request.tools
        and len(user_texts) >= 2
        and user_texts[0] == _TITLE_USER_SENTINEL
    )


@dataclasses.dataclass(kw_only=True)
class OpenCodeAgent:
    """Install and run OpenCode in a sandbox through ares-proxy."""

    container: containers.Container
    llm_client: llm_clients.LLMClient
    opencode_version: str = opencode_cli.DEFAULT_VERSION
    proxy_timeout_minutes: int = 15
    poll_interval_seconds: float = _POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        self._proxy = ares_proxy.SandboxProxy(
            container=self.container,
            timeout_minutes=self.proxy_timeout_minutes,
        )
        self._cli = opencode_cli.OpenCodeCLI(container=self.container, version=self.opencode_version)
        self._opencode_task: asyncio.Task[None] | None = None
        self._trajectory: list[dict[str, Any]] = []

    async def run(self, task: str) -> None:
        try:
            await self._proxy.install()
            await self._cli.install()
            await self._proxy.start()
            await self._cli.configure(self._proxy.url)
            self._opencode_task = asyncio.create_task(self._cli.run(task))
            await self._bridge_requests()
        finally:
            try:
                await self._persist_trajectory()
            except Exception:
                _LOGGER.exception("[%d] Failed to persist OpenCode trajectory", id(self))
            finally:
                await self._cleanup()

    async def _bridge_requests(self) -> None:
        assert self._opencode_task is not None
        while not self._opencode_task.done():
            pending_requests = await self._proxy.poll()
            await asyncio.gather(*(self._process_request(value) for value in pending_requests))
            if not pending_requests:
                await asyncio.sleep(self.poll_interval_seconds)
        await self._opencode_task

    async def _process_request(self, proxy_request: ares_proxy.PendingRequest) -> None:
        llm_request = _to_llm_request(proxy_request)
        if _is_title_request(llm_request):
            self._record_trajectory(
                "request",
                request_id=proxy_request.id,
                endpoint=proxy_request.endpoint,
                filtered="opencode_title",
                raw_request=proxy_request.request,
                llm_request=dataclasses.asdict(llm_request),
            )
            llm_response = response.LLMResponse(
                data=[response.TextData(content="ARES evaluation")],
                cost=0.0,
                usage=response.Usage(prompt_tokens=0, generated_tokens=0),
            )
            filtered = "opencode_title"
        else:
            self._record_trajectory(
                "request",
                request_id=proxy_request.id,
                endpoint=proxy_request.endpoint,
                raw_request=proxy_request.request,
                llm_request=dataclasses.asdict(llm_request),
            )
            llm_response = await self.llm_client(llm_request)
            filtered = None

        self._record_trajectory(
            "response",
            request_id=proxy_request.id,
            filtered=filtered,
            response=dataclasses.asdict(llm_response),
        )
        model = proxy_request.request.get("model")
        stream = openai_responses_stream.to_sse(
            llm_response,
            model=model if isinstance(model, str) else "ares",
        )
        await self._proxy.respond(proxy_request.id, stream, content_type="text/event-stream")

    def _record_trajectory(self, event: str, **values: Any) -> None:
        self._trajectory.append({"event": event, "timestamp": time.time(), **values})

    async def _persist_trajectory(self) -> None:
        if not self._trajectory:
            return
        with tempfile.TemporaryDirectory(prefix="ares-opencode-trajectory-") as temp_dir:
            local_path = pathlib.Path(temp_dir) / "mediated.jsonl"
            local_path.write_text(
                "".join(json.dumps(record, default=str, separators=(",", ":")) + "\n" for record in self._trajectory)
            )
            await self.container.upload_file(local_path, _MEDIATED_LOG_PATH)

    async def _cleanup(self) -> None:
        await self._cli.stop()
        await self._proxy.stop()
        if self._opencode_task is not None:
            if not self._opencode_task.done():
                self._opencode_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._opencode_task
        self._opencode_task = None
