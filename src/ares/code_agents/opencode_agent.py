"""OpenCode agent running inside an ARES container."""

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Literal, cast
import uuid

import openai.types.responses.response_create_params

from ares.containers import containers
from ares.llms import llm_clients
from ares.llms import openai_responses_converter
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)

_PROXY_PORT = 8080
_PROXY_PATH = "/usr/local/bin/ares-proxy"
_PROXY_LOG_PATH = "/logs/agent/ares-proxy.log"
_OPENCODE_LOG_PATH = "/logs/agent/opencode.jsonl"
_MEDIATED_LOG_PATH = "/logs/agent/mediated.jsonl"
_OPENCODE_PID_PATH = "/tmp/ares-opencode.pid"
_OPENCODE_VERSION = "1.14.48"
_POLL_INTERVAL_SECONDS = 0.25
_TITLE_SYSTEM_FINGERPRINT = "You are a title generator. You output ONLY a thread title."
_TITLE_USER_SENTINEL = "Generate a title for this conversation:\n"

_PROXY_BINARY_CACHE: dict[str, pathlib.Path] = {}
_PROXY_BUILD_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class PendingProxyRequest:
    """An LLM request returned by the proxy's poll endpoint."""

    id: str
    endpoint: Literal["/v1/responses"]
    request: dict[str, Any]


def _parse_pending_request(value: Any) -> PendingProxyRequest:
    if not isinstance(value, dict):
        raise ValueError(f"Proxy request must be an object, got {type(value).__name__}")

    request_id = value.get("id")
    endpoint = value.get("endpoint")
    request_body = value.get("request")
    if not isinstance(request_id, str):
        raise ValueError("Proxy request field 'id' must be a string")
    if endpoint != "/v1/responses":
        raise ValueError(f"OpenCode must use /v1/responses, got {endpoint!r}")
    if not isinstance(request_body, dict):
        raise ValueError("Proxy request field 'request' must be an object")

    return PendingProxyRequest(id=request_id, endpoint=endpoint, request=request_body)


def _is_title_request(llm_request: request.LLMRequest) -> bool:
    """Identify OpenCode's hidden title request without relying on request order."""
    user_texts = [message.get("content") for message in llm_request.messages if message.get("role") == "user"]
    return (
        _TITLE_SYSTEM_FINGERPRINT in (llm_request.system_prompt or "")
        and not llm_request.tools
        and len(user_texts) >= 2
        and user_texts[0] == _TITLE_USER_SENTINEL
    )


def _to_llm_request(proxy_request: PendingProxyRequest) -> request.LLMRequest:
    params = dict(proxy_request.request)
    params.pop("stream", None)
    return openai_responses_converter.from_external(
        cast(openai.types.responses.response_create_params.ResponseCreateParamsBase, params),
        strict=False,
    )


def _sse_event(event: dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


def _to_responses_sse(
    llm_response: response.LLMResponse,
    *,
    model: str,
) -> str:
    """Convert one atomic ARES response into a complete Responses API stream."""
    response_id = f"resp_{uuid.uuid4().hex}"
    output: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sequence_number = 0

    def add_event(event_type: str, **values: Any) -> None:
        nonlocal sequence_number
        events.append({"type": event_type, "sequence_number": sequence_number, **values})
        sequence_number += 1

    response_base: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "in_progress",
        "model": model,
        "output": [],
    }
    add_event("response.created", response=response_base)
    add_event("response.in_progress", response=response_base)

    text = "".join(part.content for part in llm_response.data)
    if text or not llm_response.tool_calls:
        output_index = len(output)
        item_id = f"msg_{uuid.uuid4().hex}"
        empty_item = {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        add_event("response.output_item.added", output_index=output_index, item=empty_item)
        empty_part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        add_event(
            "response.content_part.added",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            part=empty_part,
        )
        add_event(
            "response.output_text.delta",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            delta=text,
            logprobs=[],
        )
        add_event(
            "response.output_text.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            text=text,
            logprobs=[],
        )
        completed_part = {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        add_event(
            "response.content_part.done",
            item_id=item_id,
            output_index=output_index,
            content_index=0,
            part=completed_part,
        )
        completed_item = {**empty_item, "status": "completed", "content": [completed_part]}
        add_event("response.output_item.done", output_index=output_index, item=completed_item)
        output.append(completed_item)

    for tool_call in llm_response.tool_calls:
        output_index = len(output)
        item_id = f"fc_{uuid.uuid4().hex}"
        empty_item = {
            "id": item_id,
            "type": "function_call",
            "status": "in_progress",
            "call_id": tool_call.call_id,
            "name": tool_call.name,
            "arguments": "",
        }
        add_event("response.output_item.added", output_index=output_index, item=empty_item)
        add_event(
            "response.function_call_arguments.delta",
            item_id=item_id,
            output_index=output_index,
            delta=tool_call.arguments,
        )
        add_event(
            "response.function_call_arguments.done",
            item_id=item_id,
            output_index=output_index,
            name=tool_call.name,
            arguments=tool_call.arguments,
        )
        completed_item = {**empty_item, "status": "completed", "arguments": tool_call.arguments}
        add_event("response.output_item.done", output_index=output_index, item=completed_item)
        output.append(completed_item)

    usage = {
        "input_tokens": llm_response.usage.prompt_tokens,
        "output_tokens": llm_response.usage.generated_tokens,
        "total_tokens": llm_response.usage.total_tokens,
    }
    add_event(
        "response.completed",
        response={**response_base, "status": "completed", "output": output, "usage": usage},
    )
    return "".join(_sse_event(event) for event in events)


def _proxy_source_dir() -> pathlib.Path:
    package_root = pathlib.Path(__file__).resolve().parents[1]
    candidates = [package_root / "ares-proxy", package_root.parents[1] / "ares-proxy"]
    for candidate in candidates:
        if (candidate / "go.mod").is_file():
            return candidate
    raise RuntimeError("Could not locate packaged ares-proxy source")


def _build_proxy_sync(source_dir: pathlib.Path, output_path: pathlib.Path, goarch: str) -> None:
    env = {**os.environ, "GOOS": "linux", "GOARCH": goarch, "CGO_ENABLED": "0"}
    if go := shutil.which("go"):
        command = [go, "build", "-o", str(output_path), "."]
        result = subprocess.run(command, cwd=source_dir, env=env, capture_output=True, text=True)
    elif docker := shutil.which("docker"):
        command = [
            docker,
            "run",
            "--rm",
            "-e",
            "GOOS=linux",
            "-e",
            f"GOARCH={goarch}",
            "-e",
            "CGO_ENABLED=0",
            "-v",
            f"{source_dir}:/src:ro",
            "-v",
            f"{output_path.parent}:/out",
            "-w",
            "/src",
            "golang:1.24-alpine",
            "go",
            "build",
            "-o",
            f"/out/{output_path.name}",
            ".",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    else:
        raise RuntimeError("Building ares-proxy requires either Go or Docker on the ARES client")

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build ares-proxy: {result.stderr}")


def _get_proxy_binary_sync(goarch: str) -> pathlib.Path:
    with _PROXY_BUILD_LOCK:
        if goarch in _PROXY_BINARY_CACHE:
            return _PROXY_BINARY_CACHE[goarch]

        output_path = pathlib.Path(tempfile.mkdtemp(prefix="ares-proxy-")) / "ares-proxy"
        _build_proxy_sync(_proxy_source_dir(), output_path, goarch)
        _PROXY_BINARY_CACHE[goarch] = output_path
        return output_path


async def _get_proxy_binary(goarch: str) -> pathlib.Path:
    return await asyncio.to_thread(_get_proxy_binary_sync, goarch)


@dataclasses.dataclass(kw_only=True)
class OpenCodeAgent:
    """Install and run OpenCode in a sandbox through ares-proxy."""

    container: containers.Container
    llm_client: llm_clients.LLMClient
    opencode_version: str = _OPENCODE_VERSION
    proxy_timeout_minutes: int = 15
    poll_interval_seconds: float = _POLL_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        self._proxy_pid: int | None = None
        self._opencode_task: asyncio.Task[None] | None = None
        self._trajectory: list[dict[str, Any]] = []

    async def run(self, task: str) -> None:
        try:
            await self._install_proxy()
            await self._install_opencode()
            await self._start_proxy()
            await self._configure_opencode()
            self._opencode_task = asyncio.create_task(self._run_opencode(task))
            await self._bridge_requests()
        finally:
            try:
                await self._persist_trajectory()
            except Exception:
                _LOGGER.exception("[%d] Failed to persist OpenCode trajectory", id(self))
            finally:
                await self._cleanup()

    async def _install_proxy(self) -> None:
        result = await self.container.exec_run("uname -m")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to determine sandbox architecture: {result.output}")
        architecture = result.output.strip()
        goarch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(architecture)
        if goarch is None:
            raise RuntimeError(f"Unsupported sandbox architecture: {architecture}")

        proxy_binary = await _get_proxy_binary(goarch)
        await self.container.upload_file(proxy_binary, _PROXY_PATH)
        result = await self.container.exec_run(f"chmod +x {_PROXY_PATH}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to install ares-proxy: {result.output}")

    async def _install_opencode(self) -> None:
        install_command = " && ".join(
            [
                "if ! command -v curl >/dev/null; then "
                "if command -v apt-get >/dev/null; then apt-get update && apt-get install -y curl ca-certificates; "
                "elif command -v apk >/dev/null; then apk add --no-cache curl ca-certificates; "
                "else echo 'curl is required' >&2; exit 1; fi; fi",
                'if [ ! -s "$HOME/.nvm/nvm.sh" ]; then '
                "curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash; fi",
                '. "$HOME/.nvm/nvm.sh"',
                "nvm install 22",
                f"npm install -g opencode-ai@{shlex.quote(self.opencode_version)}",
            ]
        )
        result = await self.container.exec_run(install_command, timeout_s=600)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to install OpenCode: {result.output}")

    async def _start_proxy(self) -> None:
        result = await self.container.exec_run("mkdir -p /logs/agent")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create agent log directory: {result.output}")
        command = (
            f"nohup env PORT={_PROXY_PORT} TIMEOUT_MINUTES={self.proxy_timeout_minutes} "
            f"{_PROXY_PATH} > {_PROXY_LOG_PATH} 2>&1 & echo $!"
        )
        result = await self.container.exec_run(command)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to start ares-proxy: {result.output}")
        try:
            self._proxy_pid = int(result.output.strip().splitlines()[-1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Failed to parse ares-proxy PID: {result.output}") from error

        result = await self.container.exec_run(
            f"for _ in $(seq 1 50); do "
            f"curl -fsS http://localhost:{_PROXY_PORT}/poll >/dev/null && exit 0; "
            "sleep 0.1; done; "
            f"cat {_PROXY_LOG_PATH}; exit 1"
        )
        if result.exit_code != 0:
            raise RuntimeError(f"ares-proxy failed to start: {result.output}")

    async def _configure_opencode(self) -> None:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "agent": {"title": {"disable": True}},
            "provider": {
                "ares": {
                    "name": "ARES",
                    "npm": "@ai-sdk/openai",
                    "env": [],
                    "options": {
                        "apiKey": "unused",
                        "baseURL": f"http://localhost:{_PROXY_PORT}/v1",
                    },
                    "models": {
                        "ares": {
                            "name": "ARES-mediated model",
                            "limit": {"context": 1_000_000, "output": 128_000},
                        }
                    },
                }
            },
        }
        config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
        result = await self.container.exec_run(
            f'mkdir -p "$HOME/.config/opencode" && printf %s {config_json} > "$HOME/.config/opencode/opencode.json"'
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to configure OpenCode: {result.output}")

    async def _run_opencode(self, task: str) -> None:
        inner_command = (
            '. "$HOME/.nvm/nvm.sh" && '
            f"echo $$ > {_OPENCODE_PID_PATH} && "
            "exec opencode --model=ares/ares run --format=json --thinking "
            f"--dangerously-skip-permissions -- {shlex.quote(task)}"
        )
        result = await self.container.exec_run(
            f"bash -lc {shlex.quote(inner_command)} > {_OPENCODE_LOG_PATH} 2>&1",
            env={"OPENCODE_FAKE_VCS": "git"},
        )
        if result.exit_code != 0:
            log_result = await self.container.exec_run(f"tail -c 10000 {_OPENCODE_LOG_PATH}")
            raise RuntimeError(f"OpenCode exited with code {result.exit_code}: {log_result.output}")

    async def _bridge_requests(self) -> None:
        assert self._opencode_task is not None
        while not self._opencode_task.done():
            pending_requests = await self._poll_proxy()
            await asyncio.gather(*(self._process_request(value) for value in pending_requests))

            if not pending_requests:
                await asyncio.sleep(self.poll_interval_seconds)

        await self._opencode_task

    async def _process_request(self, value: Any) -> None:
        proxy_request = _parse_pending_request(value)
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
            title_response = response.LLMResponse(
                data=[response.TextData(content="ARES evaluation")],
                cost=0.0,
                usage=response.Usage(prompt_tokens=0, generated_tokens=0),
            )
            self._record_trajectory(
                "response",
                request_id=proxy_request.id,
                filtered="opencode_title",
                response=dataclasses.asdict(title_response),
            )
            model = proxy_request.request.get("model")
            stream = _to_responses_sse(title_response, model=model if isinstance(model, str) else "ares")
            await self._respond_to_proxy(proxy_request.id, stream)
            return

        self._record_trajectory(
            "request",
            request_id=proxy_request.id,
            endpoint=proxy_request.endpoint,
            raw_request=proxy_request.request,
            llm_request=dataclasses.asdict(llm_request),
        )
        llm_response = await self.llm_client(llm_request)
        self._record_trajectory(
            "response",
            request_id=proxy_request.id,
            response=dataclasses.asdict(llm_response),
        )
        model = proxy_request.request.get("model")
        stream = _to_responses_sse(llm_response, model=model if isinstance(model, str) else "ares")
        await self._respond_to_proxy(proxy_request.id, stream)

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

    async def _poll_proxy(self) -> list[Any]:
        result = await self.container.exec_run(
            f"curl -fsS http://localhost:{_PROXY_PORT}/poll",
            timeout_s=10,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to poll ares-proxy: {result.output}")
        try:
            value = json.loads(result.output)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ares-proxy returned invalid poll JSON: {result.output}") from error
        if not isinstance(value, list):
            raise RuntimeError(f"ares-proxy poll response must be a list, got {type(value).__name__}")
        return value

    async def _respond_to_proxy(self, request_id: str, stream: str) -> None:
        payload = shlex.quote(
            json.dumps(
                {
                    "id": request_id,
                    "response": stream,
                    "content_type": "text/event-stream",
                },
                separators=(",", ":"),
            )
        )
        result = await self.container.exec_run(
            f"curl -fsS -X POST http://localhost:{_PROXY_PORT}/respond "
            f"-H 'Content-Type: application/json' --data-binary {payload}",
            timeout_s=10,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to respond through ares-proxy: {result.output}")

    async def _cleanup(self) -> None:
        await self.container.exec_run(
            f"if [ -f {_OPENCODE_PID_PATH} ]; then kill $(cat {_OPENCODE_PID_PATH}) 2>/dev/null || true; fi"
        )
        if self._proxy_pid is not None:
            await self.container.exec_run(f"kill {self._proxy_pid} 2>/dev/null || true")
            self._proxy_pid = None

        if self._opencode_task is not None:
            if not self._opencode_task.done():
                self._opencode_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._opencode_task
        self._opencode_task = None
