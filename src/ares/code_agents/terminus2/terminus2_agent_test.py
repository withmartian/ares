"""Integration tests for Terminus2Agent.

Tests the full agent including tmux session management, command execution,
and LLM interaction.
"""

import re
from unittest import mock

import pytest

from ares.code_agents.terminus2 import terminus2_agent
from ares.code_agents.terminus2.json_parser import Command
from ares.containers import containers
from ares.llms import llm_clients
from ares.testing.mock_container import MockContainer


class TmuxSimulator:
    """Helper class to simulate tmux behavior for testing."""

    def __init__(self):
        self.sessions = {}  # session_name -> state
        self.panes = {}  # session_name -> output

    def handle_command(self, command: str) -> containers.ExecResult:
        """Handle a tmux command and return appropriate result."""
        # Simulate tmux commands
        if "tmux new-session" in command:
            match = re.search(r"-s\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                self.sessions[session_name] = "active"
                self.panes[session_name] = ""
            return containers.ExecResult(output="", exit_code=0)

        elif "tmux send-keys" in command and "-l" in command:
            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                # Extract the quoted text (supports both single and double quotes)
                text_match = re.search(r"-l\s+(?:'([^']*)'|\"([^\"]*)\")", command)
                if text_match and session_name in self.panes:
                    text = text_match.group(1) or text_match.group(2) or ""
                    self.panes[session_name] += text
            return containers.ExecResult(output="", exit_code=0)

        elif "tmux send-keys" in command and "Enter" in command:
            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                if session_name in self.panes:
                    typed_command = self.panes[session_name]
                    self.panes[session_name] += "\n"
                    if typed_command.strip():
                        self.panes[session_name] += f"[executed: {typed_command}]\n"
            return containers.ExecResult(output="", exit_code=0)

        elif "tmux capture-pane" in command:
            match = re.search(r"-t\s+(\S+)", command)
            output = ""
            if match:
                session_name = match.group(1)
                output = self.panes.get(session_name, "")
            return containers.ExecResult(output=output, exit_code=0)

        elif "tmux kill-session" in command:
            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                self.sessions.pop(session_name, None)
                self.panes.pop(session_name, None)
            return containers.ExecResult(output="", exit_code=0)

        elif "tmux has-session" in command:
            match = re.search(r"-t\s+(\S+)", command)
            if match:
                session_name = match.group(1)
                exit_code = 0 if session_name in self.sessions else 1
                return containers.ExecResult(output="", exit_code=exit_code)
            return containers.ExecResult(output="", exit_code=1)

        elif "which tmux" in command:
            return containers.ExecResult(output="/usr/bin/tmux", exit_code=0)

        elif "tmux set-option" in command:
            return containers.ExecResult(output="", exit_code=0)

        # Default success for other commands
        return containers.ExecResult(output="", exit_code=0)


class TestTerminus2AgentBasics:
    """Test basic agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initializes correctly."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
            parser_format="json",
        )

        assert agent.parser_format == "json"
        assert agent.max_turns == 1_000_000
        assert agent.tmux_pane_width == 160
        assert agent.tmux_pane_height == 40

    @pytest.mark.asyncio
    async def test_tmux_session_creation(self):
        """Test tmux session is created on first command."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command
        commands = [Command(keystrokes="ls\n", duration=0.1)]
        await agent._execute_commands(commands)

        # Check session was created
        assert len(tmux_sim.sessions) == 1
        assert any("tmux new-session" in cmd for cmd in container.exec_commands)

    @pytest.mark.asyncio
    async def test_command_execution(self):
        """Test commands are sent to tmux correctly."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command
        commands = [Command(keystrokes="echo hello\n", duration=0.1)]
        output = await agent._execute_commands(commands)

        # Verify command was sent
        send_keys_cmds = [cmd for cmd in container.exec_commands if "send-keys" in cmd]
        assert len(send_keys_cmds) >= 2  # At least: send text + send Enter

        # Verify output was captured
        assert output is not None

    @pytest.mark.asyncio
    async def test_multiple_commands(self):
        """Test multiple commands are executed in sequence."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
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
        send_keys_cmds = [cmd for cmd in container.exec_commands if "send-keys" in cmd and "-l" in cmd]
        assert len(send_keys_cmds) >= 3  # At least 3 commands

        # Verify output was captured
        assert output is not None

    @pytest.mark.asyncio
    async def test_tmux_cleanup(self):
        """Test tmux session is cleaned up."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = terminus2_agent.Terminus2Agent(
            container=container,
            llm_client=llm_client,
        )

        # Execute a command (creates session)
        commands = [Command(keystrokes="ls\n", duration=0.1)]
        await agent._execute_commands(commands)

        session_count_before = len(tmux_sim.sessions)
        assert session_count_before == 1

        # Cleanup
        await agent._cleanup_tmux_session()

        # Verify session was killed
        assert len(tmux_sim.sessions) == 0
        assert any("kill-session" in cmd for cmd in container.exec_commands)


class TestTerminus2AgentIntegration:
    """Integration tests with mocked LLM."""

    @pytest.mark.asyncio
    async def test_simple_task_completion(self):
        """Test agent can complete a simple task."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)

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
        assert len(container.exec_commands) > 0
        assert any("send-keys" in cmd for cmd in container.exec_commands)

        # Verify session was created and cleaned up
        assert len(tmux_sim.sessions) == 0  # Cleaned up after completion


class TestTerminus2AgentEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_command_skipped(self):
        """Test empty commands are skipped."""
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
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
        tmux_sim = TmuxSimulator()
        container = MockContainer(exec_handler=tmux_sim.handle_command)
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
        send_keys_cmds = [cmd for cmd in container.exec_commands if "send-keys" in cmd]
        assert len(send_keys_cmds) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
