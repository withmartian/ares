"""Tests for gymnasium-compatible wrappers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

from ares.environments.base import TimeStep
from ares.environments.gym_wrapper import AsyncGymWrapper
from ares.environments.gym_wrapper import GymWrapper
from ares.environments.gym_wrapper import wrap_as_gym


def _make_time_step(step_type: str, reward: float | None = None, obs: Any = "obs") -> TimeStep:
    return TimeStep(step_type=step_type, reward=reward, discount=None, observation=obs)


def _make_env(
    first_obs: Any = "first_obs",
    mid_obs: Any = "mid_obs",
    last_obs: Any = "last_obs",
    final_reward: float = 1.0,
) -> MagicMock:
    env = MagicMock()
    env.reset = AsyncMock(return_value=_make_time_step("FIRST", None, first_obs))
    env.step = AsyncMock(
        side_effect=[
            _make_time_step("MID", None, mid_obs),
            _make_time_step("LAST", final_reward, last_obs),
        ]
    )
    env.close = AsyncMock()
    return env


# ---------------------------------------------------------------------------
# AsyncGymWrapper
# ---------------------------------------------------------------------------


class TestAsyncGymWrapper:
    @pytest.mark.asyncio
    async def test_reset_returns_obs_and_empty_info(self) -> None:
        env = _make_env(first_obs="hello")
        wrapper = AsyncGymWrapper(env)
        obs, info = await wrapper.reset()
        assert obs == "hello"
        assert info == {}

    @pytest.mark.asyncio
    async def test_step_mid_not_terminated(self) -> None:
        env = _make_env()
        wrapper = AsyncGymWrapper(env)
        await wrapper.reset()
        obs, reward, terminated, truncated, _info = await wrapper.step("action")
        assert obs == "mid_obs"
        assert reward == 0.0
        assert terminated is False
        assert truncated is False

    @pytest.mark.asyncio
    async def test_step_last_terminated(self) -> None:
        env = _make_env(final_reward=0.5)
        wrapper = AsyncGymWrapper(env)
        await wrapper.reset()
        await wrapper.step("action")  # MID
        obs, reward, terminated, truncated, _info = await wrapper.step("action")  # LAST
        assert obs == "last_obs"
        assert reward == pytest.approx(0.5)
        assert terminated is True
        assert truncated is False

    @pytest.mark.asyncio
    async def test_step_info_contains_time_step(self) -> None:
        env = _make_env()
        wrapper = AsyncGymWrapper(env)
        await wrapper.reset()
        _, _, _, _, info = await wrapper.step("action")
        assert "time_step" in info
        assert isinstance(info["time_step"], TimeStep)

    @pytest.mark.asyncio
    async def test_close_delegates_to_env(self) -> None:
        env = _make_env()
        wrapper = AsyncGymWrapper(env)
        await wrapper.close()
        env.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_calls_close(self) -> None:
        env = _make_env()
        async with AsyncGymWrapper(env) as wrapper:
            await wrapper.reset()
        env.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_observation_and_action_space_none(self) -> None:
        env = _make_env()
        wrapper = AsyncGymWrapper(env)
        assert wrapper.observation_space is None
        assert wrapper.action_space is None

    @pytest.mark.asyncio
    async def test_reward_none_becomes_zero(self) -> None:
        env = MagicMock()
        env.reset = AsyncMock(return_value=_make_time_step("FIRST", None))
        env.step = AsyncMock(return_value=_make_time_step("MID", None))
        env.close = AsyncMock()
        wrapper = AsyncGymWrapper(env)
        await wrapper.reset()
        _, reward, _, _, _ = await wrapper.step("action")
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_reset_seed_and_options_ignored(self) -> None:
        env = _make_env()
        wrapper = AsyncGymWrapper(env)
        obs, _info = await wrapper.reset(seed=42, options={"foo": "bar"})
        assert obs == "first_obs"


# ---------------------------------------------------------------------------
# GymWrapper (sync bridge)
# ---------------------------------------------------------------------------


class TestGymWrapper:
    def test_reset_returns_obs_and_empty_info(self) -> None:
        env = _make_env(first_obs="hello")
        wrapper = GymWrapper(env)
        obs, info = wrapper.reset()
        assert obs == "hello"
        assert info == {}

    def test_step_mid_not_terminated(self) -> None:
        env = _make_env()
        wrapper = GymWrapper(env)
        wrapper.reset()
        obs, _reward, terminated, truncated, _info = wrapper.step("action")
        assert obs == "mid_obs"
        assert terminated is False
        assert truncated is False

    def test_step_last_terminated(self) -> None:
        env = _make_env(final_reward=1.0)
        wrapper = GymWrapper(env)
        wrapper.reset()
        wrapper.step("action")  # MID
        _obs, reward, terminated, _truncated, _info = wrapper.step("action")  # LAST
        assert terminated is True
        assert reward == pytest.approx(1.0)

    def test_close_delegates_to_env(self) -> None:
        env = _make_env()
        wrapper = GymWrapper(env)
        wrapper.close()
        env.close.assert_awaited_once()

    def test_context_manager_calls_close(self) -> None:
        env = _make_env()
        with GymWrapper(env) as wrapper:
            wrapper.reset()
        env.close.assert_awaited_once()

    def test_observation_and_action_space_none(self) -> None:
        env = _make_env()
        wrapper = GymWrapper(env)
        assert wrapper.observation_space is None
        assert wrapper.action_space is None


# ---------------------------------------------------------------------------
# wrap_as_gym factory
# ---------------------------------------------------------------------------


class TestWrapAsGym:
    def test_default_returns_sync_wrapper(self) -> None:
        env = _make_env()
        result = wrap_as_gym(env)
        assert isinstance(result, GymWrapper)

    def test_async_mode_returns_async_wrapper(self) -> None:
        env = _make_env()
        result = wrap_as_gym(env, async_mode=True)
        assert isinstance(result, AsyncGymWrapper)

    def test_sync_wrapper_is_usable(self) -> None:
        env = _make_env(first_obs="x")
        gym_env = wrap_as_gym(env)
        obs, _info = gym_env.reset()
        assert obs == "x"

    @pytest.mark.asyncio
    async def test_async_wrapper_is_usable(self) -> None:
        env = _make_env(first_obs="y")
        gym_env = wrap_as_gym(env, async_mode=True)
        obs, _info = await gym_env.reset()
        assert obs == "y"
