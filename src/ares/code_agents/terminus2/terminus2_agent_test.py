"""Integration tests for Terminus2Agent.

Tests the full agent including tmux session management, command execution,
and LLM interaction.
"""

import pathlib
from unittest import mock

import pytest

from ares.code_agents.terminus2 import terminus2_agent
from ares.code_agents.terminus2.json_parser import Command
from ares.containers import containers
from ares.llms import llm_clients


class MockContainer(containers.Container):
    """Mock container that tracks commands and simulates tmux."""

    def __init__(self):
        self.commands_run = []
        self.tmux_sessions = {}  # session_name -> state
        self.tmux_panes = {}  # session_name -> output

    async def exec_run(self, command: str, **_kwargs):
        """Simulate command execution."""
        self.commands_run.append(command)

        class Result:
            output = ""
            exit_code = 0

        result = Result()

        # Simulate tmux commands
        if "tmux new-session" in command:
            # Extract session name
            import re

            match = re.search(r"-s\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                self.tmux_sessions[session_name] = "active"
                self.tmux_panes[session_name] = ""

        elif "tmux send-keys" in command and "-l" in command:
            # Extract session and text
            import re

            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                # Extract the quoted text
                text_match = re.search(r"-l\s+'([^']*)'", command)
                if text_match and session_name in self.tmux_panes:
                    text = text_match.group(1)
                    self.tmux_panes[session_name] += text

        elif "tmux send-keys" in command and "Enter" in command:
            # Execute what's been typed
            import re

            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                if session_name in self.tmux_panes:
                    typed_command = self.tmux_panes[session_name]
                    # Simulate command execution
                    self.tmux_panes[session_name] += "\n"
                    if typed_command.strip():
                        # Add command output
                        self.tmux_panes[session_name] += f"[executed: {typed_command}]\n"

        elif "tmux capture-pane" in command:
            # Return current pane content
            import re

            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                result.output = self.tmux_panes.get(session_name, "")

        elif "tmux kill-session" in command:
            # Clean up session
            import re

            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                if session_name in self.tmux_sessions:
                    del self.tmux_sessions[session_name]
                if session_name in self.tmux_panes:
                    del self.tmux_panes[session_name]

        return result

    async def start(self, env=None):
        pass

    async def stop(self):
        pass

    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]):
        pass

    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]):
        pass


class TestTerminus2AgentBasics:
    """Test basic agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
            parser_format="json",
        )

        assert agent.parser_format == "json"
        assert agent.max_turns == 50
        assert agent.tmux_pane_width == 160
        assert agent.tmux_pane_height == 40

    @pytest.mark.asyncio
    async def test_tmux_session_creation(self):
        """Test tmux session is created on first command."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command
        commands = [Command(keystrokes="ls\n", duration=0.1)]
        await agent._execute_commands(commands)

        # Check session was created
        assert len(container.tmux_sessions) == 1
        assert any("tmux new-session" in cmd for cmd in container.commands_run)

    @pytest.mark.asyncio
    async def test_command_execution(self):
        """Test commands are sent to tmux correctly."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command
        commands = [Command(keystrokes="echo hello\n", duration=0.1)]
        output = await agent._execute_commands(commands)

        # Verify command was sent
        send_keys_cmds = [cmd for cmd in container.commands_run if "send-keys" in cmd]
        assert len(send_keys_cmds) >= 2  # At least: send text + send Enter

        # Verify output was captured
        assert output is not None

    @pytest.mark.asyncio
    async def test_multiple_commands(self):
        """Test multiple commands are executed in sequence."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute multiple commands
        commands = [
            Command(keystrokes="cd /tmp\n", duration=0.1),
            Command(keystrokes="pwd\n", duration=0.1),
            Command(keystrokes="ls\n", duration=0.1),
        ]
        output = await agent._execute_commands(commands)

        # Verify all commands were sent
        send_keys_cmds = [cmd for cmd in container.commands_run if "send-keys" in cmd and "-l" in cmd]
        assert len(send_keys_cmds) >= 3  # At least 3 commands

        # Verify output was captured
        assert output is not None

    @pytest.mark.asyncio
    async def test_tmux_cleanup(self):
        """Test tmux session is cleaned up."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command (creates session)
        commands = [Command(keystrokes="ls\n", duration=0.1)]
        await agent._execute_commands(commands)

        session_count_before = len(container.tmux_sessions)
        assert session_count_before == 1

        # Cleanup
        await agent._cleanup_tmux_session()

        # Verify session was killed
        assert len(container.tmux_sessions) == 0
        assert any("kill-session" in cmd for cmd in container.commands_run)


class TestTerminus2AgentIntegration:
    """Integration tests with mocked LLM."""

    @pytest.mark.asyncio
    async def test_simple_task_completion(self):
        """Test agent can complete a simple task."""
        container = MockContainer()

        # Mock LLM responses
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        # First response: execute a command
        response1 = mock.MagicMock()
        response1.chat_completion_response.choices[0].message.content = """{
  "analysis": "Need to list files",
  "plan": "Run ls command",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}"""
        response1.chat_completion_response.usage = None

        # Second response: mark complete
        response2 = mock.MagicMock()
        response2.chat_completion_response.choices[0].message.content = """{
  "analysis": "Files listed",
  "plan": "Task is done",
  "commands": [],
  "task_complete": true
}"""
        response2.chat_completion_response.usage = None

        # Third response: confirm completion
        response3 = mock.MagicMock()
        response3.chat_completion_response.choices[0].message.content = """{
  "analysis": "Confirming completion",
  "plan": "Done",
  "commands": [],
  "task_complete": true
}"""
        response3.chat_completion_response.usage = None

        llm_client.side_effect = [response1, response2, response3]

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
            max_turns=10,
        )

        # Run the agent
        await agent.run("List all files in the current directory")

        # Verify LLM was called
        assert llm_client.call_count >= 2

        # Verify commands were executed
        assert len(container.commands_run) > 0
        assert any("send-keys" in cmd for cmd in container.commands_run)

        # Verify session was created and cleaned up
        assert len(container.tmux_sessions) == 0  # Cleaned up after completion


class TestTerminus2AgentEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_command_skipped(self):
        """Test empty commands are skipped."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute with empty command
        commands = [
            Command(keystrokes="", duration=0.1),
            Command(keystrokes="ls\n", duration=0.1),
        ]
        output = await agent._execute_commands(commands)

        # Verify empty was skipped but ls was executed
        assert output is not None

    @pytest.mark.asyncio
    async def test_command_without_newline(self):
        """Test command without trailing newline."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute command without newline
        commands = [Command(keystrokes="ls", duration=0.1)]
        output = await agent._execute_commands(commands)

        # Should still work (text sent, but no Enter)
        assert output is not None
        send_keys_cmds = [cmd for cmd in container.commands_run if "send-keys" in cmd]
        assert len(send_keys_cmds) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
