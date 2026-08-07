"""Tests for the in-sandbox OpenCode agent."""

import asyncio
import dataclasses
import json
import pathlib
import shlex

import pytest

from ares.code_agents import ares_proxy
from ares.code_agents import opencode_agent
from ares.containers import containers
from ares.llms import request
from ares.llms import response
from ares.testing import mock_container


def _title_request() -> dict:
    return {
        "id": "title-request",
        "endpoint": "/v1/responses",
        "request": {
            "model": "ares",
            "stream": True,
            "store": False,
            "input": [
                {
                    "role": "developer",
                    "content": f"{opencode_agent._TITLE_SYSTEM_FINGERPRINT}\n\n<task>Generate a brief title.</task>",
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": opencode_agent._TITLE_USER_SENTINEL}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Fix the bug"}],
                },
            ],
        },
    }


def test_title_request_detection_is_content_based() -> None:
    assert opencode_agent._is_title_request(opencode_agent._to_llm_request(ares_proxy._parse_request(_title_request())))

    first_real_request = _title_request()
    first_real_request["request"]["input"][0]["content"] = "You are a coding agent."
    assert not opencode_agent._is_title_request(
        opencode_agent._to_llm_request(ares_proxy._parse_request(first_real_request))
    )


@pytest.mark.asyncio
async def test_title_request_is_answered_without_llm_client() -> None:
    container = _SimulatedOpenCodeContainer()
    llm_client = _ToolCallingLLMClient()
    agent = opencode_agent.OpenCodeAgent(container=container, llm_client=llm_client)

    await agent._process_request(ares_proxy._parse_request(_title_request()))

    assert llm_client.requests == []
    assert container.response_payload is not None
    assert "ARES evaluation" in container.response_payload["response"]
    assert agent._trajectory[0]["filtered"] == "opencode_title"


@dataclasses.dataclass
class _SimulatedOpenCodeContainer(mock_container.MockContainer):
    opencode_started: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    opencode_finished: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    request_polled: bool = False
    response_payload: dict | None = None

    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> containers.ExecResult:
        del workdir, env, timeout_s
        self.exec_commands.append(command)

        if command == "uname -m":
            return containers.ExecResult(output="x86_64\n", exit_code=0)
        if command.startswith("nohup env PORT="):
            return containers.ExecResult(output="123\n", exit_code=0)
        if command.startswith("for _ in $(seq 1 50)"):
            return containers.ExecResult(output="", exit_code=0)
        if "exec opencode" in command:
            self.opencode_started.set()
            await self.opencode_finished.wait()
            return containers.ExecResult(output="", exit_code=0)
        if command == "curl -fsS http://localhost:8080/poll":
            if not self.opencode_started.is_set() or self.request_polled:
                return containers.ExecResult(output="[]\n", exit_code=0)
            self.request_polled = True
            pending = [
                {
                    "id": "proxy-request-1",
                    "endpoint": "/v1/responses",
                    "timestamp": "2026-08-03T00:00:00Z",
                    "request": {
                        "model": "ares",
                        "stream": True,
                        "input": [
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": "Create hello.txt"}],
                            }
                        ],
                        "tools": [
                            {
                                "type": "function",
                                "name": "write",
                                "description": "Write a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "filePath": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                },
                            }
                        ],
                    },
                }
            ]
            return containers.ExecResult(output=json.dumps(pending), exit_code=0)
        if command.startswith("curl -fsS -X POST http://localhost:8080/respond"):
            parts = shlex.split(command)
            self.response_payload = json.loads(parts[parts.index("--data-binary") + 1])
            self.opencode_finished.set()
            return containers.ExecResult(output='{"status":"ok"}', exit_code=0)

        return containers.ExecResult(output="", exit_code=0)


@dataclasses.dataclass
class _ToolCallingLLMClient:
    requests: list[request.LLMRequest] = dataclasses.field(default_factory=list)

    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse:
        self.requests.append(request)
        return response.LLMResponse(
            data=[response.TextData(content="")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
            tool_calls=[
                response.ToolCallData(
                    call_id="call_write",
                    name="write",
                    arguments='{"filePath":"/workspace/hello.txt","content":"hello"}',
                )
            ],
        )


@pytest.mark.asyncio
async def test_opencode_agent_runs_proxy_bridge(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    proxy_binary = tmp_path / "ares-proxy"
    proxy_binary.write_bytes(b"binary")

    async def get_proxy_binary(goarch: str) -> pathlib.Path:
        assert goarch == "amd64"
        return proxy_binary

    monkeypatch.setattr(ares_proxy, "_get_binary", get_proxy_binary)
    container = _SimulatedOpenCodeContainer()
    llm_client = _ToolCallingLLMClient()
    agent = opencode_agent.OpenCodeAgent(
        container=container,
        llm_client=llm_client,
        poll_interval_seconds=0.001,
    )

    await asyncio.wait_for(agent.run("Create hello.txt"), timeout=2)

    assert container.uploaded_files[0] == (proxy_binary, ares_proxy.PATH)
    assert container.uploaded_files[-1][1] == "/logs/agent/mediated.jsonl"
    assert len(llm_client.requests) == 1
    assert llm_client.requests[0].messages == [{"role": "user", "content": "Create hello.txt"}]

    assert container.response_payload is not None
    assert container.response_payload["id"] == "proxy-request-1"
    assert container.response_payload["content_type"] == "text/event-stream"
