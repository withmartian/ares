"""Episode-based replay buffer for reinforcement learning.

This module provides a replay buffer that stores per-episode experiences and
supports n-step sampling with multi-agent capabilities.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
import random
from typing import Any

import numpy as np
import numpy.typing as npt


class EpisodeStatus(Enum):
    """Status of an episode in the replay buffer."""

    IN_PROGRESS = "in_progress"
    TERMINAL = "terminal"
    TRUNCATED = "truncated"


@dataclass
class NStepSample:
    """A batch of n-step samples from the replay buffer.

    Attributes:
        obs_t: Initial observations at time t, shape (batch_size, *obs_shape)
        action_t: Actions taken at time t, shape (batch_size, *action_shape)
        reward_seq: Sequence of rewards [r_t, r_{t+1}, ..., r_{t+m-1}],
                   shape (batch_size, m) where m <= n
        next_obs: Observations at time t+m, shape (batch_size, *obs_shape)
        m: Actual number of steps in each sample (batch_size,)
        terminal: Whether the episode terminated (batch_size,)
        truncated: Whether the episode was truncated (batch_size,)
    """

    obs_t: npt.NDArray[Any]
    action_t: npt.NDArray[Any]
    reward_seq: npt.NDArray[np.floating[Any]]
    next_obs: npt.NDArray[Any]
    m: npt.NDArray[np.integer[Any]]
    terminal: npt.NDArray[np.bool_]
    truncated: npt.NDArray[np.bool_]


@dataclass
class _Episode:
    """Internal representation of an episode."""

    episode_id: str
    agent_id: str
    observations: list[Any]
    actions: list[Any]
    rewards: list[float]
    status: EpisodeStatus


class ReplayBuffer:
    """Episode-based replay buffer with n-step sampling support.

    This replay buffer stores experiences organized by episodes, supporting
    multi-agent environments and n-step returns. It samples uniformly over
    all time steps across eligible episodes, weighted by episode length.

    Args:
        max_episodes: Maximum number of episodes to store (None for unlimited)
        max_steps: Maximum total number of steps to store (None for unlimited)
    """

    def __init__(
        self,
        max_episodes: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Initialize the replay buffer."""
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self._episodes: dict[str, _Episode] = {}
        self._finished_episodes: list[str] = []
        self._lock = asyncio.Lock()
        self._total_steps = 0
        self._next_episode_id = 0

    async def start_episode(
        self,
        agent_id: str,
        episode_id: str | None = None,
    ) -> str:
        """Start a new episode.

        Args:
            agent_id: Identifier for the agent
            episode_id: Optional episode identifier (auto-generated if None)

        Returns:
            The episode identifier

        Raises:
            ValueError: If episode_id already exists
        """
        async with self._lock:
            if episode_id is None:
                episode_id = f"{agent_id}_episode_{self._next_episode_id}"
                self._next_episode_id += 1

            if episode_id in self._episodes:
                raise ValueError(f"Episode {episode_id} already exists")

            self._episodes[episode_id] = _Episode(
                episode_id=episode_id,
                agent_id=agent_id,
                observations=[],
                actions=[],
                rewards=[],
                status=EpisodeStatus.IN_PROGRESS,
            )

            return episode_id

    async def append_step(
        self,
        episode_id: str,
        observation: Any,
        action: Any,
        reward: float,
    ) -> None:
        """Append a step to an episode.

        Args:
            episode_id: Episode identifier
            observation: Observation at this step
            action: Action taken at this step
            reward: Reward received at this step

        Raises:
            ValueError: If episode doesn't exist or is finished
        """
        async with self._lock:
            if episode_id not in self._episodes:
                raise ValueError(f"Episode {episode_id} does not exist")

            episode = self._episodes[episode_id]
            if episode.status != EpisodeStatus.IN_PROGRESS:
                raise ValueError(f"Cannot append to finished episode {episode_id} (status: {episode.status})")

            episode.observations.append(observation)
            episode.actions.append(action)
            episode.rewards.append(reward)
            self._total_steps += 1

            await self._maybe_evict()

    async def end_episode(
        self,
        episode_id: str,
        status: EpisodeStatus,
    ) -> None:
        """Mark an episode as finished.

        Args:
            episode_id: Episode identifier
            status: Final status (TERMINAL or TRUNCATED)

        Raises:
            ValueError: If episode doesn't exist or status is IN_PROGRESS
        """
        if status == EpisodeStatus.IN_PROGRESS:
            raise ValueError("Cannot end episode with IN_PROGRESS status")

        async with self._lock:
            if episode_id not in self._episodes:
                raise ValueError(f"Episode {episode_id} does not exist")

            episode = self._episodes[episode_id]
            episode.status = status
            self._finished_episodes.append(episode_id)

            await self._maybe_evict()

    async def _maybe_evict(self) -> None:
        """Evict oldest finished episodes if capacity is exceeded."""
        while self._finished_episodes and (
            (self.max_episodes and len(self._finished_episodes) > self.max_episodes)
            or (self.max_steps and self._total_steps > self.max_steps)
        ):
            episode_id = self._finished_episodes.pop(0)
            episode = self._episodes.pop(episode_id)
            self._total_steps -= len(episode.observations)

    def _get_eligible_episodes(self) -> list[_Episode]:
        """Get episodes eligible for sampling (finished episodes only)."""
        return [self._episodes[episode_id] for episode_id in self._finished_episodes if episode_id in self._episodes]

    async def sample_n_step(
        self,
        batch_size: int,
        n: int,
        _gamma: float = 0.99,
    ) -> NStepSample:
        """Sample n-step experiences from the buffer.

        Samples uniformly over all time steps across eligible (finished) episodes,
        weighted by episode length. Does not cross episode boundaries.

        Args:
            batch_size: Number of samples to return
            n: Number of steps for n-step returns
            gamma: Discount factor (included for API compatibility, not used)

        Returns:
            NStepSample containing batched n-step experiences

        Raises:
            ValueError: If buffer is empty or batch_size > available steps
        """
        async with self._lock:
            eligible_episodes = self._get_eligible_episodes()

            if not eligible_episodes:
                raise ValueError("No finished episodes available for sampling")

            total_steps = sum(len(ep.observations) for ep in eligible_episodes)
            if batch_size > total_steps:
                raise ValueError(f"Requested batch_size {batch_size} exceeds available steps {total_steps}")

            episode_weights = [len(ep.observations) for ep in eligible_episodes]
            total_weight = sum(episode_weights)
            episode_probs = [w / total_weight for w in episode_weights]

            samples = []
            for _ in range(batch_size):
                episode = random.choices(eligible_episodes, weights=episode_probs)[0]
                episode_len = len(episode.observations)

                t = random.randint(0, episode_len - 1)
                m = min(n, episode_len - t)

                obs_t = episode.observations[t]
                action_t = episode.actions[t]
                reward_seq = episode.rewards[t : t + m]

                if t + m < episode_len:
                    next_obs = episode.observations[t + m]
                    terminal = False
                    truncated = False
                else:
                    next_obs = episode.observations[-1]
                    terminal = episode.status == EpisodeStatus.TERMINAL
                    truncated = episode.status == EpisodeStatus.TRUNCATED

                samples.append(
                    {
                        "obs_t": obs_t,
                        "action_t": action_t,
                        "reward_seq": reward_seq,
                        "next_obs": next_obs,
                        "m": m,
                        "terminal": terminal,
                        "truncated": truncated,
                    }
                )

            obs_t_batch = np.array([s["obs_t"] for s in samples])
            action_t_batch = np.array([s["action_t"] for s in samples])

            max_m = max(s["m"] for s in samples)
            reward_seq_batch = np.zeros((batch_size, max_m), dtype=np.float32)
            for i, s in enumerate(samples):
                reward_seq_batch[i, : s["m"]] = s["reward_seq"]

            next_obs_batch = np.array([s["next_obs"] for s in samples])
            m_batch = np.array([s["m"] for s in samples], dtype=np.int32)
            terminal_batch = np.array([s["terminal"] for s in samples], dtype=bool)
            truncated_batch = np.array([s["truncated"] for s in samples], dtype=bool)

            return NStepSample(
                obs_t=obs_t_batch,
                action_t=action_t_batch,
                reward_seq=reward_seq_batch,
                next_obs=next_obs_batch,
                m=m_batch,
                terminal=terminal_batch,
                truncated=truncated_batch,
            )

    def get_num_episodes(self) -> int:
        """Get the number of episodes in the buffer."""
        return len(self._episodes)

    def get_num_finished_episodes(self) -> int:
        """Get the number of finished episodes in the buffer."""
        return len(self._finished_episodes)

    def get_total_steps(self) -> int:
        """Get the total number of steps in the buffer."""
        return self._total_steps
