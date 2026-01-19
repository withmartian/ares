"""Tests for ares.environments.base."""

import asyncio
from unittest import mock

import numpy as np
import pytest

from ares.code_agents import code_agent_base
from ares.containers import containers
from ares.environments import base
from ares.environments import specs
from ares.llms import llm_clients


class ConcreteCodeBaseEnv(base.CodeBaseEnv[str]):
    """Concrete implementation of CodeBaseEnv for testing."""

    def __init__(self, container: containers.Container, **kwargs):
        self._test_container = container
        self._test_task: str = "test task"
        self._current_task: str = self._test_task
        super().__init__(**kwargs)

    async def _reset_task(self) -> None:
        self._current_task = self._test_task

    async def _start_container(self) -> None:
        self._container = self._test_container

    async def _start_code_agent(self) -> None:
        # Create a simple code agent task that makes one LLM request
        async def agent_fn():
            request = llm_clients.LLMRequest(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self._current_task},
                ],
                temperature=0.7,
            )
            await self._llm_client(request)

        self._code_agent_task = asyncio.create_task(agent_fn())

    async def _compute_reward(self) -> float:
        return 1.0


def create_mock_container():
    """Create a mock container for testing."""
    container = mock.AsyncMock(spec=containers.Container)
    container.start = mock.AsyncMock()
    container.stop = mock.AsyncMock()
    container.stop_and_remove = mock.Mock()
    return container


def create_mock_container_factory(mock_container):
    """Create a mock container factory."""
    factory = mock.Mock(spec=containers.ContainerFactory)
    factory.from_image = mock.Mock(return_value=mock_container)
    return factory


def create_mock_agent_factory():
    """Create a mock code agent factory."""
    return mock.Mock(spec=code_agent_base.CodeAgentFactory)


def assert_spec_matches_value(spec, value, path=""):
    """Recursively validate that a value matches its spec structure.

    Args:
        spec: The spec (dict, list, or Array/StringArray/BoundedArray)
        value: The actual value to validate
        path: Current path for error messages
    """
    if isinstance(spec, dict):
        # Custom handling for dict containers - accept any object with the required attributes
        for key, child_spec in spec.items():
            assert hasattr(value, key) or (isinstance(value, dict) and key in value), (
                f"{path}.{key}: Missing key in value"
            )
            child_value = getattr(value, key, None) if hasattr(value, key) else value.get(key)
            assert_spec_matches_value(child_spec, child_value, f"{path}.{key}")

    elif isinstance(spec, list):
        # Custom handling for list containers
        assert hasattr(value, "__iter__"), f"{path}: Expected iterable, got {type(value)}"
        # Check at least one item if the list is non-empty
        value_list = list(value)
        if len(value_list) > 0 and len(spec) > 0:
            # Validate first item against spec
            assert_spec_matches_value(spec[0], value_list[0], f"{path}[0]")

    elif isinstance(spec, (specs.Array, specs.BoundedArray, specs.StringArray)):
        if value is None:
            raise AssertionError(f"{path}: Value is None")

        # Use the spec's built-in validate method for leaf specs
        try:
            spec.validate(value)
        except ValueError as e:
            raise AssertionError(f"{path}: Spec validation failed: {e}") from e
    else:
        raise AssertionError(f"{path}: Unknown spec type {type(spec)}")


def test_reward_spec_is_bounded_between_zero_and_one():
    """Test that reward_spec returns a bounded array [0, 1]."""
    mock_container = create_mock_container()
    mock_container_factory = create_mock_container_factory(mock_container)
    mock_agent_factory = create_mock_agent_factory()

    env = ConcreteCodeBaseEnv(
        container=mock_container,
        container_factory=mock_container_factory,
        code_agent_factory=mock_agent_factory,
    )

    reward_spec = env.reward_spec()

    assert isinstance(reward_spec, specs.BoundedArray)
    assert reward_spec.shape == ()
    assert reward_spec.dtype == np.dtype(float)
    assert reward_spec.minimum == 0.0
    assert reward_spec.maximum == 1.0


def test_discount_spec_is_bounded_between_zero_and_one():
    """Test that discount_spec returns a bounded array [0, 1]."""
    mock_container = create_mock_container()
    mock_container_factory = create_mock_container_factory(mock_container)
    mock_agent_factory = create_mock_agent_factory()

    env = ConcreteCodeBaseEnv(
        container=mock_container,
        container_factory=mock_container_factory,
        code_agent_factory=mock_agent_factory,
    )

    discount_spec = env.discount_spec()

    assert isinstance(discount_spec, specs.BoundedArray)
    assert discount_spec.shape == ()
    assert discount_spec.dtype == np.dtype(float)
    assert discount_spec.minimum == 0.0
    assert discount_spec.maximum == 1.0


@pytest.mark.asyncio
async def test_initial_observation_from_reset_matches_observation_spec():
    """Test that reset returns an observation matching the observation_spec."""
    mock_container = create_mock_container()
    mock_container_factory = create_mock_container_factory(mock_container)
    mock_agent_factory = create_mock_agent_factory()

    env = ConcreteCodeBaseEnv(
        container=mock_container,
        container_factory=mock_container_factory,
        code_agent_factory=mock_agent_factory,
    )

    async with env:
        timestep = await env.reset()

        # FIRST timesteps should have None reward/discount per dm_env spec
        assert timestep.step_type == "FIRST"
        assert timestep.reward is None
        assert timestep.discount is None
        assert timestep.observation is not None

        # Validate observation matches spec
        obs = timestep.observation
        assert isinstance(obs, llm_clients.LLMRequest)
        obs_spec = env.observation_spec()
        assert_spec_matches_value(obs_spec, obs, "observation")


@pytest.mark.asyncio
async def test_environment_accepts_action_matching_action_spec():
    """Test that the environment accepts actions that conform to action_spec."""
    mock_container = create_mock_container()
    mock_container_factory = create_mock_container_factory(mock_container)
    mock_agent_factory = create_mock_agent_factory()

    env = ConcreteCodeBaseEnv(
        container=mock_container,
        container_factory=mock_container_factory,
        code_agent_factory=mock_agent_factory,
    )

    async with env:
        await env.reset()

        # Create an action using the helper
        action = llm_clients.build_openai_compatible_llm_response(
            output_text="Test response",
            num_input_tokens=10,
            num_output_tokens=3,
            model="test-model",
            cost=0.001,
        )

        # Validate action matches spec
        action_spec = env.action_spec()
        assert_spec_matches_value(action_spec, action, "action")

        # Test that the environment actually accepts this action
        timestep = await env.step(action)

        # If step succeeds without error, the environment accepted the action
        assert timestep is not None
        assert timestep.step_type in ["MID", "LAST"]


@pytest.mark.asyncio
async def test_reward_and_discount_are_within_spec_bounds_after_step():
    """Test that step returns reward and discount within their spec bounds."""
    mock_container = create_mock_container()
    mock_container_factory = create_mock_container_factory(mock_container)
    mock_agent_factory = create_mock_agent_factory()

    env = ConcreteCodeBaseEnv(
        container=mock_container,
        container_factory=mock_container_factory,
        code_agent_factory=mock_agent_factory,
    )

    async with env:
        await env.reset()

        # Create and take action using helper
        action = llm_clients.build_openai_compatible_llm_response(
            output_text="Test response",
            num_input_tokens=10,
            num_output_tokens=3,
            model="test-model",
        )

        timestep = await env.step(action)

        # Check timestep has reward and discount
        assert timestep.step_type in ["MID", "LAST"]
        assert timestep.reward is not None
        assert timestep.discount is not None

        # Validate reward and discount are within bounds using spec's validate method
        reward_spec = env.reward_spec()
        discount_spec = env.discount_spec()
        reward_spec.validate(timestep.reward)
        discount_spec.validate(timestep.discount)
