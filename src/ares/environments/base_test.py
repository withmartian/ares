"""Unit tests for the base environment module."""

import asyncio
import pathlib
import time
from typing import Any
from unittest import mock

import pytest

# Mock the config before importing base to avoid validation errors
with mock.patch.dict("os.environ", {"CHAT_COMPLETION_API_KEY": "test_key"}):
    from ares.code_agents import code_agent_base
    from ares.containers import containers
    from ares.environments import base
    from ares.llms import llm_clients


def create_mock_llm_response(content: str = "test") -> llm_clients.LLMResponse:
    """Helper to create a mock LLMResponse."""
    mock_completion = mock.MagicMock()
    mock_completion.choices = [mock.MagicMock()]
    mock_completion.choices[0].message.content = content
    return llm_clients.LLMResponse(chat_completion_response=mock_completion, cost=0.0)


class TestTimeStep:
    """Tests for TimeStep helper methods."""

    def test_first(self):
        """Test that first() correctly identifies FIRST step type."""
        ts = base.TimeStep(step_type="FIRST", reward=None, discount=None, observation="obs")
        assert ts.first() is True
        assert ts.mid() is False
        assert ts.last() is False

    def test_mid(self):
        """Test that mid() correctly identifies MID step type."""
        ts = base.TimeStep(step_type="MID", reward=0.0, discount=1.0, observation="obs")
        assert ts.first() is False
        assert ts.mid() is True
        assert ts.last() is False

    def test_last(self):
        """Test that last() correctly identifies LAST step type."""
        ts = base.TimeStep(step_type="LAST", reward=1.0, discount=0.0, observation="obs")
        assert ts.first() is False
        assert ts.mid() is False
        assert ts.last() is True


class TestCreateContainer:
    """Tests for create_container factory function."""

    @pytest.mark.asyncio
    async def test_create_container_from_image(self):
        """Test creating a container from an image."""
        mock_container = mock.MagicMock(spec=containers.Container)
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_factory.from_image.return_value = mock_container

        result = await base.create_container(
            container_factory=mock_factory,
            container_prefix="test",
            image_name="registry.example.com/test-image:latest",
        )

        assert result == mock_container
        mock_factory.from_image.assert_called_once()
        call_kwargs = mock_factory.from_image.call_args[1]
        assert call_kwargs["image"] == "registry.example.com/test-image:latest"
        assert call_kwargs["name"].startswith("ares.test.test-image.")
        assert call_kwargs["resources"] is None

    @pytest.mark.asyncio
    async def test_create_container_from_dockerfile(self):
        """Test creating a container from a Dockerfile."""
        mock_container = mock.MagicMock(spec=containers.Container)
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_factory.from_dockerfile.return_value = mock_container

        dockerfile_path = pathlib.Path("/path/to/project.example/Dockerfile")

        result = await base.create_container(
            container_factory=mock_factory,
            container_prefix="test",
            dockerfile_path=dockerfile_path,
        )

        assert result == mock_container
        mock_factory.from_dockerfile.assert_called_once()
        call_kwargs = mock_factory.from_dockerfile.call_args[1]
        assert call_kwargs["dockerfile_path"] == dockerfile_path
        assert call_kwargs["name"].startswith("ares.test.example.")
        assert call_kwargs["resources"] is None

    @pytest.mark.asyncio
    async def test_create_container_with_resources(self):
        """Test creating a container with custom resources."""
        mock_container = mock.MagicMock(spec=containers.Container)
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_factory.from_image.return_value = mock_container

        resources = containers.Resources(cpu=4, memory=8192)

        result = await base.create_container(
            container_factory=mock_factory,
            container_prefix="test",
            image_name="test-image",
            resources=resources,
        )

        assert result == mock_container
        call_kwargs = mock_factory.from_image.call_args[1]
        assert call_kwargs["resources"] == resources

    @pytest.mark.asyncio
    async def test_create_container_name_format(self):
        """Test that container names have the correct format."""
        mock_container = mock.MagicMock(spec=containers.Container)
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_factory.from_image.return_value = mock_container

        before_time = int(time.time())
        await base.create_container(
            container_factory=mock_factory,
            container_prefix="myprefix",
            image_name="myimage:latest",
        )
        after_time = int(time.time())

        call_kwargs = mock_factory.from_image.call_args[1]
        container_name = call_kwargs["name"]

        # Check format: ares.<prefix>.<image_short>.<timestamp>.<uuid>
        parts = container_name.split(".")
        assert len(parts) == 5
        assert parts[0] == "ares"
        assert parts[1] == "myprefix"
        assert parts[2] == "myimage"

        # Check timestamp is reasonable
        timestamp = int(parts[3])
        assert before_time <= timestamp <= after_time

        # Check UUID is 8 characters
        assert len(parts[4]) == 8

    @pytest.mark.asyncio
    async def test_create_container_no_image_or_dockerfile_raises_error(self):
        """Test that an error is raised if neither image nor dockerfile is provided."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)

        with pytest.raises(ValueError, match="Must specify one of image_name or dockerfile_path"):
            await base.create_container(
                container_factory=mock_factory,
                container_prefix="test",
            )


class TestJanitor:
    """Tests for the Janitor cleanup mechanism."""

    def test_register_and_unregister(self):
        """Test registering and unregistering an environment."""
        janitor = base.Janitor()
        mock_env = mock.MagicMock(spec=base.CodeBaseEnv)

        # Register
        janitor.register_for_cleanup(mock_env)
        assert id(mock_env) in janitor._environment_by_id
        assert janitor._environment_by_id[id(mock_env)] == mock_env

        # Unregister
        janitor.unregister_for_cleanup(mock_env)
        assert id(mock_env) not in janitor._environment_by_id

    def test_cleanup_environment(self):
        """Test that cleanup stops and removes the container."""
        janitor = base.Janitor()
        mock_env = mock.MagicMock(spec=base.CodeBaseEnv)
        mock_container = mock.MagicMock(spec=containers.Container)
        mock_env._container = mock_container

        janitor._cleanup_environment(mock_env)

        mock_container.stop_and_remove.assert_called_once()

    def test_cleanup_environment_no_container(self):
        """Test that cleanup handles environments with no container."""
        janitor = base.Janitor()
        mock_env = mock.MagicMock(spec=base.CodeBaseEnv)
        mock_env._container = None

        # Should not raise
        janitor._cleanup_environment(mock_env)

    def test_sync_cleanup(self):
        """Test that sync cleanup processes all registered environments."""
        janitor = base.Janitor()

        mock_env1 = mock.MagicMock(spec=base.CodeBaseEnv)
        mock_container1 = mock.MagicMock(spec=containers.Container)
        mock_env1._container = mock_container1

        mock_env2 = mock.MagicMock(spec=base.CodeBaseEnv)
        mock_container2 = mock.MagicMock(spec=containers.Container)
        mock_env2._container = mock_container2

        janitor.register_for_cleanup(mock_env1)
        janitor.register_for_cleanup(mock_env2)

        assert len(janitor._environment_by_id) == 2

        janitor._sync_cleanup()

        # Both containers should be stopped
        mock_container1.stop_and_remove.assert_called_once()
        mock_container2.stop_and_remove.assert_called_once()

        # All environments should be unregistered
        assert len(janitor._environment_by_id) == 0


class TestCodeBaseEnv:
    """Tests for CodeBaseEnv abstract base class."""

    class TestEnv(base.CodeBaseEnv[str]):
        """Concrete test environment for testing CodeBaseEnv."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._reset_called = False
            self._container_started = False
            self._agent_started = False
            self._reward_value = 1.0
            self._requires_reset = False
            self.mock_container = mock.MagicMock(spec=containers.Container)
            self.mock_agent_coro = None

        async def _reset_task(self) -> None:
            self._reset_called = True
            self._current_task = "test_task"

        async def _start_container(self) -> None:
            self._container_started = True
            self._container = self.mock_container

        async def _start_code_agent(self) -> None:
            self._agent_started = True
            # Create a coroutine that will run the mock agent
            if self.mock_agent_coro is not None:
                self._code_agent_task = asyncio.create_task(self.mock_agent_coro())

        async def _compute_reward(self) -> float:
            return self._reward_value

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test that the environment works as an async context manager."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        assert env._is_active is False

        async with env:
            assert env._is_active is True

        assert env._is_active is False

    @pytest.mark.asyncio
    async def test_reset_requires_active_context(self):
        """Test that reset requires the environment to be used as a context manager."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        with pytest.raises(RuntimeError, match="Environment is not active"):
            await env.reset()

    @pytest.mark.asyncio
    async def test_step_requires_active_context(self):
        """Test that step requires the environment to be used as a context manager."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        action = create_mock_llm_response("test")

        with pytest.raises(RuntimeError, match="Environment is not active"):
            await env.step(action)

    @pytest.mark.asyncio
    async def test_reset_sequence_initialization(self):
        """Test that reset properly initializes the sequence."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent():
            # Simulate agent making an LLM request
            req = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])
            await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            ts = await env.reset()

            assert ts.step_type == "FIRST"
            assert ts.reward is None or ts.reward == 0.0
            assert ts.discount is None or ts.discount == 1.0
            assert ts.observation is not None
            assert isinstance(ts.observation, llm_clients.LLMRequest)

            assert env._reset_called is True
            assert env._container_started is True
            assert env._agent_started is True
            assert env._step_count == 0
            assert env._current_task == "test_task"

    @pytest.mark.asyncio
    async def test_reset_stops_existing_container(self):
        """Test that reset stops an existing container."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent():
            req = llm_clients.LLMRequest(messages=[{"role": "user", "content": "test"}])
            await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        old_container = mock.MagicMock(spec=containers.Container)
        old_container.stop = mock.AsyncMock()

        async with env:
            env._container = old_container

            await env.reset()

            old_container.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_error_on_immediate_last(self):
        """Test that reset raises an error if the agent completes immediately."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent_completes_immediately():
            # Agent completes without making any LLM requests
            return

        env.mock_agent_coro = mock_agent_completes_immediately

        async with env:
            with pytest.raises(RuntimeError, match="The code agent didn't make any LLM requests"):
                await env.reset()

    @pytest.mark.asyncio
    async def test_step_sets_future_result(self):
        """Test that step sets the future result with the action."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        request_count = 0

        async def mock_agent():
            nonlocal request_count
            # Make multiple requests
            for i in range(3):
                req = llm_clients.LLMRequest(messages=[{"role": "user", "content": f"request_{i}"}])
                await env._llm_client(req)
                request_count += 1

        env.mock_agent_coro = mock_agent

        async with env:
            ts = await env.reset()
            assert ts.step_type == "FIRST"

            # First step
            action = create_mock_llm_response("response_0")
            ts = await env.step(action)

            assert ts.step_type == "MID"
            assert ts.observation is not None
            assert env._step_count == 1

    @pytest.mark.asyncio
    async def test_step_increments_count(self):
        """Test that step increments the step count."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent():
            for i in range(5):
                req = llm_clients.LLMRequest(messages=[{"role": "user", "content": f"req_{i}"}])
                await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            await env.reset()
            assert env._step_count == 0

            for i in range(3):
                action = create_mock_llm_response(f"resp_{i}")
                await env.step(action)
                assert env._step_count == i + 1

    @pytest.mark.asyncio
    async def test_step_limit_truncation(self):
        """Test that step enforces the step limit and truncates."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        step_limit = 3
        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
            step_limit=step_limit,
        )

        async def mock_agent():
            # Agent tries to make many requests
            for i in range(10):
                req = llm_clients.LLMRequest(messages=[{"role": "user", "content": f"req_{i}"}])
                await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            await env.reset()

            # Take steps until limit
            for i in range(step_limit - 1):
                action = create_mock_llm_response(f"resp_{i}")
                ts = await env.step(action)
                assert ts.step_type == "MID"

            # The step at the limit should be LAST
            action = create_mock_llm_response("resp_final")
            ts = await env.step(action)
            assert ts.step_type == "LAST"
            assert env._step_count == step_limit

    @pytest.mark.asyncio
    async def test_get_time_step_agent_completion(self):
        """Test _get_time_step when agent completes (returns LAST)."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )
        env._reward_value = 0.75

        async def mock_agent():
            # Agent makes one request then completes
            req = llm_clients.LLMRequest(messages=[{"role": "user", "content": "single_request"}])
            await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            await env.reset()

            # Step once - agent completes after this
            action = create_mock_llm_response("response")
            ts = await env.step(action)

            assert ts.step_type == "LAST"
            assert ts.reward == 0.75
            assert ts.discount == 0.0
            assert ts.observation is None

    @pytest.mark.asyncio
    async def test_get_time_step_llm_request(self):
        """Test _get_time_step when agent makes LLM request (returns MID)."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent():
            # Agent makes multiple requests
            for i in range(3):
                req = llm_clients.LLMRequest(messages=[{"role": "user", "content": f"req_{i}"}])
                await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            ts = await env.reset()

            # First step should return MID with second request
            action = create_mock_llm_response("resp_0")
            ts = await env.step(action)

            assert ts.step_type == "MID"
            assert ts.reward == 0.0
            assert ts.discount == 1.0
            assert ts.observation is not None
            assert isinstance(ts.observation, llm_clients.LLMRequest)

    @pytest.mark.asyncio
    async def test_close_stops_container(self):
        """Test that close stops the container."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        mock_container = mock.MagicMock(spec=containers.Container)
        mock_container.stop = mock.AsyncMock()

        async with env:
            env._container = mock_container
            await env.close()

            mock_container.stop.assert_called_once()
            assert env._container is None

    @pytest.mark.asyncio
    async def test_requires_reset_after_last(self):
        """Test that environment requires reset after LAST timestep."""
        mock_factory = mock.MagicMock(spec=containers.ContainerFactory)
        mock_agent_factory = mock.MagicMock(spec=code_agent_base.CodeAgentFactory)

        env = self.TestEnv(
            container_factory=mock_factory,
            code_agent_factory=mock_agent_factory,
        )

        async def mock_agent():
            req = llm_clients.LLMRequest(messages=[{"role": "user", "content": "only_request"}])
            await env._llm_client(req)

        env.mock_agent_coro = mock_agent

        async with env:
            await env.reset()

            # Step until LAST
            action = create_mock_llm_response("response")
            ts = await env.step(action)
            assert ts.step_type == "LAST"

            # Trying to step again should raise error
            with pytest.raises(RuntimeError, match="Environment must be reset"):
                await env.step(action)
