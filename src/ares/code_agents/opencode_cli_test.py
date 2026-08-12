"""Tests for the OpenCode sandbox runner."""

import pytest

from ares.code_agents import opencode_cli
from ares.testing import mock_container


@pytest.mark.asyncio
async def test_opencode_cli_commands() -> None:
    container = mock_container.MockContainer()
    cli = opencode_cli.OpenCodeCLI(container=container)

    await cli.install()
    await cli.configure("http://localhost:8080")
    await cli.run("Fix the bug")
    await cli.stop()

    assert any("npm install -g opencode-ai@1.14.48" in command for command in container.exec_commands)
    assert any('"npm":"@ai-sdk/openai"' in command for command in container.exec_commands)
    assert any('"agent":{"title":{"disable":true}}' in command for command in container.exec_commands)
    assert any("opencode --model=ares/ares run" in command for command in container.exec_commands)
    assert any("> /logs/agent/opencode.jsonl 2>&1" in command for command in container.exec_commands)
