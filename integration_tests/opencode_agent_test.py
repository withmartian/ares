"""End-to-end OpenCode agent smoke test."""

import dataclasses
import os
import pathlib

import pytest

from ares.code_agents import opencode_agent
from ares.containers import docker
from ares.llms import request
from ares.llms import response


@dataclasses.dataclass
class _SmokeTestLLMClient:
    calls: int = 0
    title_calls: int = 0

    async def __call__(self, request: request.LLMRequest) -> response.LLMResponse:
        self.calls += 1
        has_tool_result = any(message.get("role") == "tool" for message in request.messages)
        if has_tool_result:
            return response.LLMResponse(
                data=[response.TextData(content="The requested file has been created.")],
                cost=0.0,
                usage=response.Usage(prompt_tokens=10, generated_tokens=5),
            )

        tool_names = {tool["name"] for tool in request.tools or []}
        if not tool_names:
            self.title_calls += 1
            return response.LLMResponse(
                data=[response.TextData(content="Create OpenCode smoke-test file")],
                cost=0.0,
                usage=response.Usage(prompt_tokens=10, generated_tokens=5),
            )
        if "write" not in tool_names:
            raise AssertionError(f"OpenCode did not expose its write tool: {sorted(tool_names)}")
        return response.LLMResponse(
            data=[response.TextData(content="")],
            cost=0.0,
            usage=response.Usage(prompt_tokens=10, generated_tokens=5),
            tool_calls=[
                response.ToolCallData(
                    call_id="call_write_smoke_test",
                    name="write",
                    arguments=('{"filePath":"/tmp/ares-opencode-smoke.txt","content":"hello from opencode\\n"}'),
                )
            ],
        )


@pytest.mark.skipif(
    os.environ.get("RUN_OPENCODE_INTEGRATION") != "1",
    reason="Set RUN_OPENCODE_INTEGRATION=1 to install and run OpenCode in Docker.",
)
@pytest.mark.asyncio
async def test_opencode_runs_through_ares_proxy(tmp_path: pathlib.Path) -> None:
    container = docker.DockerContainer.from_image(image="ubuntu:22.04")
    await container.start()
    try:
        llm_client = _SmokeTestLLMClient()
        agent = opencode_agent.OpenCodeAgent(container=container, llm_client=llm_client)

        await agent.run("Create /tmp/ares-opencode-smoke.txt containing exactly 'hello from opencode'.")

        result = await container.exec_run("cat /tmp/ares-opencode-smoke.txt")
        assert result.exit_code == 0
        assert result.output.strip() == "hello from opencode"
        assert llm_client.calls >= 2
        assert llm_client.title_calls == 0

        mediated_log = await container.exec_run("cat /logs/agent/mediated.jsonl")
        assert mediated_log.exit_code == 0
        assert '"event":"request"' in mediated_log.output
        assert '"event":"response"' in mediated_log.output
        opencode_log = await container.exec_run("cat /logs/agent/opencode.jsonl")
        assert opencode_log.exit_code == 0
        assert opencode_log.output.strip()

        artifact_dir = tmp_path / "artifacts"
        await container.download_dir("/logs", artifact_dir)
        assert (artifact_dir / "agent" / "opencode.jsonl").is_file()
        assert (artifact_dir / "agent" / "mediated.jsonl").is_file()
    finally:
        await container.stop()
