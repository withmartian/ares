"""Unit tests for mini_swe_agent.py"""

import sys
from unittest import mock

import pytest

# Mock minisweagent before importing the module under test
mock_minisweagent = mock.MagicMock()
mock_minisweagent.config.builtin_config_dir = "/fake/path"
mock_default_agent = mock.MagicMock()
mock_default_agent.AgentConfig = mock.MagicMock
sys.modules['minisweagent'] = mock_minisweagent
sys.modules['minisweagent.agents'] = mock.MagicMock()
sys.modules['minisweagent.agents.default'] = mock_default_agent
sys.modules['minisweagent.config'] = mock_minisweagent.config

from ares.code_agents import mini_swe_agent
from ares.containers import containers


# Test configuration to mock yaml.safe_load
TEST_CONFIG = {
    "agent": {
        "system_template": "You are a helpful assistant. System: {{ system }}",
        "instance_template": "Task: {{ task }}\nSystem: {{ system }}\nRelease: {{ release }}\nVersion: {{ version }}\nMachine: {{ machine }}",
        "action_observation_template": "Output: {{ output.output }}\nReturn code: {{ output.returncode }}",
        "format_error_template": "Format error. Actions found: {{ actions }}",
    },
    "environment": {
        "timeout": 30,
        "env": {"TEST_VAR": "test_value"},
    },
}


@pytest.fixture
def mock_container():
    """Create a mock container."""
    container = mock.AsyncMock(spec=containers.Container)
    return container


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    return mock.AsyncMock()


@pytest.fixture
def mock_yaml_config():
    """Mock yaml.safe_load to return test configuration."""
    with mock.patch('yaml.safe_load', return_value=TEST_CONFIG):
        yield


@pytest.fixture
def agent(mock_container, mock_llm_client, mock_yaml_config):
    """Create a MiniSWECodeAgent instance with mocked dependencies."""
    with mock.patch('pathlib.Path.read_text', return_value=""):
        agent = mini_swe_agent.MiniSWECodeAgent(
            container=mock_container,
            llm_client=mock_llm_client,
        )
    return agent


class TestParseAction:
    """Tests for the parse_action method."""

    def test_single_block_success(self, agent):
        """Test parsing a single bash block successfully."""
        response_text = "Let me run this command:\n```bash\necho 'hello world'\n```\nThis should work."

        action = agent.parse_action(response_text)

        assert action == "echo 'hello world'"

    def test_single_block_with_whitespace(self, agent):
        """Test parsing a single bash block with extra whitespace."""
        response_text = "```bash\n  ls -la  \n```"

        action = agent.parse_action(response_text)

        assert action == "ls -la"

    def test_multiple_blocks_error(self, agent):
        """Test that multiple bash blocks raise a FormatError."""
        response_text = """
        First command:
        ```bash
        echo 'first'
        ```
        Second command:
        ```bash
        echo 'second'
        ```
        """

        with pytest.raises(mini_swe_agent._FormatError) as exc_info:
            agent.parse_action(response_text)

        # Verify the error message includes information about the actions
        assert "Format error" in str(exc_info.value)

    def test_no_blocks_error(self, agent):
        """Test that no bash blocks raise a FormatError."""
        response_text = "I will run a command but forgot to use code blocks."

        with pytest.raises(mini_swe_agent._FormatError) as exc_info:
            agent.parse_action(response_text)

        assert "Format error" in str(exc_info.value)

    def test_multiline_command(self, agent):
        """Test parsing a multiline bash command."""
        response_text = """```bash
for i in 1 2 3; do
    echo $i
done
```"""

        action = agent.parse_action(response_text)

        assert "for i in 1 2 3; do" in action
        assert "echo $i" in action
        assert "done" in action


class TestRaiseIfFinished:
    """Tests for the _raise_if_finished method."""

    def test_mini_swe_agent_final_output(self, agent):
        """Test that MINI_SWE_AGENT_FINAL_OUTPUT raises _SubmittedError."""
        output = containers.ExecResult(
            exit_code=0,
            output="MINI_SWE_AGENT_FINAL_OUTPUT\nThis is the final output\nwith multiple lines"
        )

        with pytest.raises(mini_swe_agent._SubmittedError) as exc_info:
            agent._raise_if_finished(output)

        # Verify the error message contains the lines after the marker
        assert "This is the final output" in str(exc_info.value)
        assert "with multiple lines" in str(exc_info.value)

    def test_complete_task_and_submit_final_output(self, agent):
        """Test that COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT raises _SubmittedError."""
        output = containers.ExecResult(
            exit_code=0,
            output="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\nTask completed successfully"
        )

        with pytest.raises(mini_swe_agent._SubmittedError) as exc_info:
            agent._raise_if_finished(output)

        assert "Task completed successfully" in str(exc_info.value)

    def test_final_output_with_leading_whitespace(self, agent):
        """Test that markers with leading whitespace are recognized."""
        output = containers.ExecResult(
            exit_code=0,
            output="  \n  MINI_SWE_AGENT_FINAL_OUTPUT\nFinal result"
        )

        with pytest.raises(mini_swe_agent._SubmittedError) as exc_info:
            agent._raise_if_finished(output)

        assert "Final result" in str(exc_info.value)

    def test_normal_output_no_exception(self, agent):
        """Test that normal output does not raise an exception."""
        output = containers.ExecResult(
            exit_code=0,
            output="This is normal output\nNothing special here"
        )

        # Should not raise any exception
        agent._raise_if_finished(output)

    def test_marker_not_at_start(self, agent):
        """Test that markers not at the start of output do not trigger."""
        output = containers.ExecResult(
            exit_code=0,
            output="Some output\nMINI_SWE_AGENT_FINAL_OUTPUT\nThis should not trigger"
        )

        # Should not raise any exception
        agent._raise_if_finished(output)

    def test_empty_output(self, agent):
        """Test that empty output does not raise an exception."""
        output = containers.ExecResult(
            exit_code=0,
            output=""
        )

        # Should not raise any exception
        agent._raise_if_finished(output)


class TestExecuteAction:
    """Tests for the execute_action method."""

    @pytest.mark.anyio
    async def test_execute_action_success(self, agent, mock_container):
        """Test successful action execution."""
        # Mock LLM response
        mock_response = mock.MagicMock()
        mock_response.chat_completion_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="```bash\necho 'test'\n```"))
        ]

        # Mock container execution
        mock_container.exec_run.return_value = containers.ExecResult(
            exit_code=0,
            output="test"
        )

        # Execute action
        await agent.execute_action(mock_response)

        # Verify container.exec_run was called with correct parameters
        mock_container.exec_run.assert_called_once_with(
            "echo 'test'",
            timeout_s=30,
            env={"TEST_VAR": "test_value"}
        )

        # Verify message was added to history
        assert len(agent._messages) > 0

    @pytest.mark.anyio
    async def test_execute_action_timeout(self, agent, mock_container):
        """Test that timeout raises _ExecutionTimeoutError."""
        # Mock LLM response
        mock_response = mock.MagicMock()
        mock_response.chat_completion_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="```bash\nsleep 100\n```"))
        ]

        # Mock container execution to raise TimeoutError
        mock_container.exec_run.side_effect = TimeoutError("Command timed out")

        # Execute action and verify exception
        with pytest.raises(mini_swe_agent._ExecutionTimeoutError) as exc_info:
            await agent.execute_action(mock_response)

        # Verify the error message mentions timeout
        assert "timed out" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_action_with_final_output(self, agent, mock_container):
        """Test that final output marker triggers _SubmittedError."""
        # Mock LLM response
        mock_response = mock.MagicMock()
        mock_response.chat_completion_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="```bash\necho 'done'\n```"))
        ]

        # Mock container execution with final output
        mock_container.exec_run.return_value = containers.ExecResult(
            exit_code=0,
            output="MINI_SWE_AGENT_FINAL_OUTPUT\nTask complete"
        )

        # Execute action and verify exception
        with pytest.raises(mini_swe_agent._SubmittedError) as exc_info:
            await agent.execute_action(mock_response)

        assert "Task complete" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_execute_action_format_error(self, agent, mock_container):
        """Test that invalid format raises _FormatError."""
        # Mock LLM response with invalid format
        mock_response = mock.MagicMock()
        mock_response.chat_completion_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="No bash blocks here"))
        ]

        # Execute action and verify exception
        with pytest.raises(mini_swe_agent._FormatError):
            await agent.execute_action(mock_response)

    @pytest.mark.anyio
    async def test_execute_action_non_zero_exit_code(self, agent, mock_container):
        """Test execution with non-zero exit code."""
        # Mock LLM response
        mock_response = mock.MagicMock()
        mock_response.chat_completion_response.choices = [
            mock.MagicMock(message=mock.MagicMock(content="```bash\nfalse\n```"))
        ]

        # Mock container execution with non-zero exit code
        mock_container.exec_run.return_value = containers.ExecResult(
            exit_code=1,
            output="Command failed"
        )

        # Execute action - should not raise exception for non-zero exit code
        await agent.execute_action(mock_response)

        # Verify execution happened
        mock_container.exec_run.assert_called_once()


class TestHelperFunctions:
    """Tests for helper/render functions."""

    def test_render_system_template(self):
        """Test rendering system template."""
        template = "System info"
        result = mini_swe_agent._render_system_template(template)
        assert result == "System info"

    def test_render_instance_template(self):
        """Test rendering instance template with all variables."""
        template = "Task: {{ task }}, System: {{ system }}"
        result = mini_swe_agent._render_instance_template(
            template,
            task="Fix bug",
            system="Linux",
            release="5.15",
            version="Ubuntu",
            machine="x86_64"
        )
        assert "Task: Fix bug" in result
        assert "System: Linux" in result

    def test_render_action_observation_template(self):
        """Test rendering action observation template."""
        template = "Exit: {{ output.returncode }}, Output: {{ output.output }}"
        output = mini_swe_agent._MiniSWEAgentOutput(returncode=0, output="success")
        result = mini_swe_agent._render_action_observation_template(template, output)
        assert "Exit: 0" in result
        assert "Output: success" in result

    def test_render_format_error_template(self):
        """Test rendering format error template."""
        template = "Found {{ actions|length }} actions"
        actions = ["action1", "action2"]
        result = mini_swe_agent._render_format_error_template(template, actions)
        assert "Found 2 actions" in result

    def test_render_timeout_template(self):
        """Test rendering timeout template."""
        result = mini_swe_agent._render_timeout_template("sleep 100", "partial output")
        assert "sleep 100" in result
        assert "timed out" in result.lower()
        assert "partial output" in result


class TestMessageManagement:
    """Tests for message management."""

    def test_add_message(self, agent):
        """Test adding messages to the message list."""
        initial_length = len(agent._messages)

        agent._add_message("user", "Test message")

        assert len(agent._messages) == initial_length + 1
        assert agent._messages[-1]["role"] == "user"
        assert agent._messages[-1]["content"] == "Test message"

    def test_add_empty_message(self, agent):
        """Test that empty messages are handled with a placeholder."""
        initial_length = len(agent._messages)

        agent._add_message("user", "   ")

        assert len(agent._messages) == initial_length + 1
        assert agent._messages[-1]["content"] == "[Empty content]"
