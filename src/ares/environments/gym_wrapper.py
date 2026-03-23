"""Gymnasium-like wrappers for ARES environments.

Provides :func:`wrap_as_gym` to adapt any ARES ``Environment`` to the
`gymnasium <https://gymnasium.farama.org>`_ API, making it accessible to
researchers and libraries already familiar with that interface.

Usage::

    import asyncio
    from ares.environments import wrap_as_gym

    async def main():
        async with MyAresEnv(...) as ares_env:
            env = wrap_as_gym(ares_env)
            obs, info = env.reset()
            while True:
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

For users already inside an async context, use :class:`AsyncGymWrapper`
directly to avoid the overhead of spinning up a nested event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ares.environments import base


def _run(coro: Any) -> Any:
    """Run a coroutine, re-using the running loop if one exists."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We are already inside an async context (e.g. Jupyter).
        # Callers should use AsyncGymWrapper directly in this case,
        # but we give a clear error rather than silently deadlocking.
        raise RuntimeError(
            "Cannot use GymWrapper from inside a running event loop. "
            "Use AsyncGymWrapper instead, or call wrap_as_gym() from a "
            "synchronous context."
        )
    return asyncio.run(coro)


class AsyncGymWrapper[ActionType, ObservationType]:
    """Async gymnasium-compatible wrapper for an ARES environment.

    Exposes the gymnasium ``reset`` / ``step`` / ``close`` interface as
    *async* methods, making it suitable for use inside ``asyncio`` event
    loops.

    Args:
        env: Any ARES :class:`~ares.environments.base.Environment`.
    """

    def __init__(
        self,
        env: base.Environment[ActionType, ObservationType, Any, Any],
    ) -> None:
        self._env = env
        # ARES environments operate on structured LLM objects rather than
        # array-based gym spaces, so we expose None here.  Libraries that
        # strictly require numpy spaces should subclass and override.
        self.observation_space: Any = None
        self.action_space: Any = None

    async def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Reset the environment and return the first observation.

        Args:
            seed: Ignored (ARES environments manage their own randomness).
            options: Ignored.

        Returns:
            A tuple of ``(observation, info)`` matching the gymnasium v26+
            interface.  ``info`` is an empty dict.
        """
        del seed, options  # Unused; kept for API compatibility.
        time_step = await self._env.reset()
        return time_step.observation, {}

    async def step(self, action: ActionType) -> tuple[ObservationType, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: The action to apply (an :class:`~ares.llms.response.LLMResponse`
                for code environments).

        Returns:
            A tuple of ``(observation, reward, terminated, truncated, info)``
            matching the gymnasium v26+ interface.

            * ``terminated`` is ``True`` when the episode ended because the task
              finished (the agent produced a final answer / the environment
              reached a terminal state).
            * ``truncated`` is always ``False``; ARES uses ``TimeoutError``
              rather than a truncation flag when a time limit is hit.
            * ``info`` carries the raw :class:`~ares.environments.base.TimeStep`
              under the key ``"time_step"`` so callers can inspect ``step_type``
              and ``discount`` if needed.
        """
        time_step = await self._env.step(action)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        terminated = time_step.last()
        info: dict[str, Any] = {"time_step": time_step}
        return time_step.observation, reward, terminated, False, info

    async def close(self) -> None:
        """Release resources held by the underlying ARES environment."""
        await self._env.close()

    async def __aenter__(self) -> AsyncGymWrapper[ActionType, ObservationType]:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()


class GymWrapper[ActionType, ObservationType]:
    """Synchronous gymnasium-compatible wrapper for an ARES environment.

    Bridges the fully-async ARES interface to a synchronous gymnasium-style
    API by running each coroutine in a new ``asyncio`` event loop.

    .. warning::
        This wrapper cannot be used from within an already-running event
        loop (e.g. inside ``async def`` functions or Jupyter notebooks).
        Use :class:`AsyncGymWrapper` in those contexts.

    Args:
        env: Any ARES :class:`~ares.environments.base.Environment`.
    """

    def __init__(
        self,
        env: base.Environment[ActionType, ObservationType, Any, Any],
    ) -> None:
        self._async = AsyncGymWrapper(env)
        self.observation_space: Any = None
        self.action_space: Any = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Reset the environment (synchronous).

        See :meth:`AsyncGymWrapper.reset` for full documentation.
        """
        return _run(self._async.reset(seed=seed, options=options))

    def step(self, action: ActionType) -> tuple[ObservationType, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment (synchronous).

        See :meth:`AsyncGymWrapper.step` for full documentation.
        """
        return _run(self._async.step(action))

    def close(self) -> None:
        """Release resources held by the underlying ARES environment."""
        _run(self._async.close())

    def __enter__(self) -> GymWrapper[ActionType, ObservationType]:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def wrap_as_gym[ActionType, ObservationType](
    env: base.Environment[ActionType, ObservationType, Any, Any],
    *,
    async_mode: bool = False,
) -> GymWrapper[ActionType, ObservationType] | AsyncGymWrapper[ActionType, ObservationType]:
    """Wrap an ARES environment in a gymnasium-compatible interface.

    Args:
        env: Any ARES :class:`~ares.environments.base.Environment`.
        async_mode: If ``True``, return an :class:`AsyncGymWrapper` with
            *async* ``reset``/``step``/``close`` methods.  If ``False``
            (default), return a synchronous :class:`GymWrapper`.

    Returns:
        A :class:`GymWrapper` (sync) or :class:`AsyncGymWrapper` (async).

    Example::

        # Synchronous usage (default)
        async with MyAresEnv(...) as ares_env:
            env = wrap_as_gym(ares_env)
            obs, info = env.reset()

        # Async usage (inside async context)
        async with MyAresEnv(...) as ares_env:
            env = wrap_as_gym(ares_env, async_mode=True)
            obs, info = await env.reset()
    """
    if async_mode:
        return AsyncGymWrapper(env)
    return GymWrapper(env)
