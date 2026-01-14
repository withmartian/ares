"""Episode Replay Buffer for Multi-Agent Reinforcement Learning.

This module provides an asyncio-safe replay buffer that supports:
- Concurrent experience collection from multiple agents
- Episode-based storage with explicit start/end control
- N-step return sampling with configurable gamma
- Capacity management with automatic eviction of oldest episodes

Storage Design:
    Episodes store per-timestep arrays: observations[t], actions[t], rewards[t].
    We do NOT duplicate next_state; instead, next_obs is derived from
    observations[t+1] during sampling.

Usage Example:
    ```python
    import asyncio
    from ares.contrib.rl.replay_buffer import (
        EpisodeReplayBuffer,
        EpisodeStatus,
    )

    # Create buffer with capacity limits
    buffer = EpisodeReplayBuffer(max_episodes=1000, max_steps=100000)

    # Start an episode
    episode_id = await buffer.start_episode(agent_id="agent_0")

    # Collect experience (initial observation before first action)
    obs_0 = {"obs": [1, 2, 3]}

    # Take action, observe reward and next obs
    action_0 = 0
    reward_0 = 1.0
    obs_1 = {"obs": [4, 5, 6]}

    await buffer.append_observation_action_reward(
        episode_id, obs_0, action_0, reward_0
    )

    # Continue...
    action_1 = 1
    reward_1 = 2.0
    obs_2 = {"obs": [7, 8, 9]}

    await buffer.append_observation_action_reward(
        episode_id, obs_1, action_1, reward_1
    )

    # End episode (either terminal or truncated)
    await buffer.end_episode(episode_id, status=EpisodeStatus.TERMINAL)

    # Sample n-step batches
    samples = await buffer.sample_n_step(batch_size=32, n=3, gamma=0.99)
    for sample in samples:
        # sample.obs_t, sample.action_t, sample.rewards_seq
        # sample.next_obs, sample.done, sample.truncated, etc.
        discounted_return = compute_discounted_return(
            sample.rewards_seq, gamma=0.99
        )
    ```

Thread Safety and Async Usage:
    All public methods are async and use an internal asyncio.Lock to ensure
    safe concurrent mutations. This buffer is designed for asyncio-only usage
    and should NOT be used with threading.Thread. Multiple asyncio tasks can
    safely write to the buffer concurrently.

    Important: Do NOT mix asyncio with threading.Thread when using this buffer.
    Use asyncio.create_task() or asyncio.gather() for concurrency.
"""

import asyncio
import collections
import dataclasses
import enum
import random
import time
from typing import Any
import uuid


class EpisodeStatus(enum.Enum):
    """Status of an episode in the replay buffer."""

    IN_PROGRESS = "in_progress"
    TERMINAL = "terminal"  # Episode ended naturally (e.g., goal reached, death)
    TRUNCATED = "truncated"  # Episode ended due to time limit or external constraint


@dataclasses.dataclass
class Episode:
    """An episode containing sequences of observations, actions, and rewards.

    Storage format:
        - observations: [obs_0, obs_1, ..., obs_T]  (length T+1)
        - actions: [a_0, a_1, ..., a_{T-1}]          (length T)
        - rewards: [r_0, r_1, ..., r_{T-1}]          (length T)

    At time step t, we have obs_t, action_t, reward_t.
    The next observation obs_{t+1} is stored at observations[t+1].
    This avoids duplicating states as next_state.

    Attributes:
        episode_id: Unique identifier for this episode
        agent_id: Identifier of the agent that generated this episode
        observations: List of observations in temporal order
        actions: List of actions taken
        rewards: List of rewards received
        status: Current status of the episode
        start_time: Timestamp when episode started (for eviction policy)
    """

    episode_id: str
    agent_id: str
    observations: list[Any] = dataclasses.field(default_factory=list)
    actions: list[Any] = dataclasses.field(default_factory=list)
    rewards: list[float] = dataclasses.field(default_factory=list)
    status: EpisodeStatus = EpisodeStatus.IN_PROGRESS
    start_time: float = dataclasses.field(default_factory=time.time)

    def __len__(self) -> int:
        """Return the number of valid (obs, action, reward) tuples (i.e., len(actions))."""
        return len(self.actions)


@dataclasses.dataclass
class NStepSample:
    """A sampled n-step experience for training.

    The sample captures a trajectory segment starting at time t:
        obs_t, action_t, [r_t, r_{t+1}, ..., r_{t+m-1}], obs_{t+m}

    where m <= n is the actual number of steps (truncated at episode boundary).

    Attributes:
        episode_id: ID of the source episode
        agent_id: ID of the agent that generated this experience
        obs_t: The observation at time t
        action_t: The action taken at time t
        rewards_seq: Sequence of rewards [r_t, r_{t+1}, ..., r_{t+m-1}] (length m)
        next_obs: The observation at time t+m (obs_{t+m})
        done: True if episode ended within the n-step window
        truncated: True if episode was truncated (vs terminal) in window
        terminal: True if episode terminated naturally in window
        discount_powers: [gamma^0, gamma^1, ..., gamma^{m-1}] for computing returns
        start_step: The starting step index t
        actual_n: The actual number of steps m (may be < n if episode ends)
        gamma: The discount factor used
    """

    episode_id: str
    agent_id: str
    obs_t: Any
    action_t: Any
    rewards_seq: list[float]
    next_obs: Any
    done: bool
    truncated: bool
    terminal: bool
    discount_powers: list[float]
    start_step: int
    actual_n: int
    gamma: float


def compute_discounted_return(rewards: list[float], gamma: float) -> float:
    """Compute the discounted return from a sequence of rewards.

    G = sum_{k=0}^{n-1} gamma^k * r_k

    Args:
        rewards: Sequence of rewards [r_0, r_1, ..., r_{n-1}]
        gamma: Discount factor in (0, 1]

    Returns:
        The discounted return
    """
    return sum(gamma**k * r for k, r in enumerate(rewards))


class EpisodeReplayBuffer:
    """Asyncio-safe replay buffer for episodic reinforcement learning.

    This buffer stores complete episodes and supports n-step sampling with
    proper handling of episode boundaries. It manages capacity by evicting
    oldest finished episodes first, then oldest in-progress episodes if needed.

    Sampling:
        Uniform sampling over all valid time steps (experiences) across episodes.
        Each valid step (obs_t, action_t, reward_t) has equal probability.
        Current implementation uses O(num_episodes) scan; a TODO exists for
        Fenwick tree optimization if needed for large buffers.

    Concurrency:
        All public methods use an internal asyncio.Lock for thread-safety.
        Safe for concurrent use by multiple asyncio tasks.

        WARNING: This buffer is designed for asyncio ONLY. Do NOT use with
        threading.Thread. Use asyncio.create_task() for concurrency.

    Capacity Management:
        - max_episodes: Maximum number of episodes to store
        - max_steps: Maximum total number of transitions across all episodes
        - Eviction policy: oldest finished episodes first, then oldest in-progress
        - Eviction updates sampling counts to maintain uniform distribution
    """

    def __init__(
        self,
        max_episodes: int | None = None,
        max_steps: int | None = None,
    ):
        """Initialize the replay buffer.

        Args:
            max_episodes: Maximum number of episodes to store (None = unlimited)
            max_steps: Maximum total transitions to store (None = unlimited)
        """
        self._lock = asyncio.Lock()
        self._episodes: dict[str, Episode] = {}
        self._max_episodes = max_episodes
        self._max_steps = max_steps
        self._total_steps = 0

        # Track episodes by agent for potential future use
        self._agent_episodes: dict[str, list[str]] = collections.defaultdict(list)

    async def start_episode(
        self,
        agent_id: str,
        episode_id: str | None = None,
    ) -> str:
        """Start a new episode.

        Args:
            agent_id: Identifier for the agent
            episode_id: Optional custom episode ID (generated if None)

        Returns:
            The episode_id for this episode

        Raises:
            ValueError: If episode_id already exists
        """
        async with self._lock:
            if episode_id is None:
                episode_id = f"{agent_id}_{uuid.uuid4().hex[:8]}"

            if episode_id in self._episodes:
                raise ValueError(f"Episode {episode_id} already exists")

            episode = Episode(episode_id=episode_id, agent_id=agent_id)
            self._episodes[episode_id] = episode
            self._agent_episodes[agent_id].append(episode_id)

            # Check capacity and evict if needed
            await self._evict_if_needed()

            return episode_id

    async def append_observation_action_reward(
        self,
        episode_id: str,
        observation: Any,
        action: Any,
        reward: float,
    ) -> None:
        """Append an observation, action, and reward to an episode.

        At time step t, call this with (obs_t, action_t, reward_t).
        The observation obs_t should be the state in which action_t was taken,
        and reward_t is the immediate reward received.

        Note: You should also store the final observation after the last action
        by calling this method one more time or handling it specially when
        ending the episode. The typical pattern is:

        1. Observe obs_0 (initial state)
        2. Take action_0, receive reward_0, observe obs_1
           -> append(obs_0, action_0, reward_0)
        3. Take action_1, receive reward_1, observe obs_2
           -> append(obs_1, action_1, reward_1)
        ...
        T. Episode ends at obs_T
           -> Store obs_T in observations but no action/reward

        Args:
            episode_id: The episode to append to
            observation: The observation at time t
            action: The action taken at time t
            reward: The reward received at time t

        Raises:
            ValueError: If episode doesn't exist or is already finished
        """
        async with self._lock:
            if episode_id not in self._episodes:
                raise ValueError(f"Episode {episode_id} not found")

            episode = self._episodes[episode_id]

            if episode.status != EpisodeStatus.IN_PROGRESS:
                raise ValueError(f"Cannot append to finished episode {episode_id} (status: {episode.status})")

            # Store observation (if this is the first call, obs_0)
            # For subsequent calls, we're storing obs_t where action_t was taken
            if len(episode.observations) == len(episode.actions):
                # We need to add the observation for this timestep
                episode.observations.append(observation)

            episode.actions.append(action)
            episode.rewards.append(reward)
            self._total_steps += 1

            # Check step capacity
            await self._evict_if_needed()

    async def end_episode(
        self,
        episode_id: str,
        status: EpisodeStatus,
        final_observation: Any | None = None,
    ) -> None:
        """Mark an episode as finished.

        Args:
            episode_id: The episode to end
            status: EpisodeStatus.TERMINAL or EpisodeStatus.TRUNCATED
            final_observation: Optional final observation obs_T after last action.
                             If provided, appended to observations list.

        Raises:
            ValueError: If episode doesn't exist, is already finished, or
                       status is IN_PROGRESS
        """
        async with self._lock:
            if episode_id not in self._episodes:
                raise ValueError(f"Episode {episode_id} not found")

            episode = self._episodes[episode_id]

            if episode.status != EpisodeStatus.IN_PROGRESS:
                raise ValueError(f"Episode {episode_id} is already finished")

            if status == EpisodeStatus.IN_PROGRESS:
                raise ValueError("Cannot end episode with status IN_PROGRESS. Use TERMINAL or TRUNCATED.")

            episode.status = status

            # Store final observation if provided and needed
            # observations should be len(actions) + 1
            if final_observation is not None and len(episode.observations) == len(episode.actions):
                episode.observations.append(final_observation)

    async def sample_n_step(
        self,
        batch_size: int,
        n: int,
        gamma: float,
    ) -> list[NStepSample]:
        """Sample n-step experiences uniformly from the buffer.

        Sampling is uniform over all valid time steps across all episodes.
        Each step (obs_t, action_t, reward_t) has equal probability.

        N-step windows never cross episode boundaries. If fewer than n steps
        remain in the episode, the sample is truncated to the available steps.

        Args:
            batch_size: Number of samples to return
            n: Number of steps for n-step returns
            gamma: Discount factor for computing returns

        Returns:
            List of n-step samples (may be less than batch_size if insufficient data)

        Raises:
            ValueError: If n < 1 or gamma not in (0, 1]
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        async with self._lock:
            # Build a list of all valid starting positions
            # A position (episode_id, t) is valid if:
            #   - episode has at least t+1 observations (obs_t exists)
            #   - episode has action_t and reward_t
            #   - t < len(actions)
            # TODO: For very large buffers, consider using a Fenwick tree
            # (Binary Indexed Tree) to maintain cumulative step counts per episode,
            # enabling O(log n) sampling instead of O(num_episodes) scan.
            valid_positions: list[tuple[str, int]] = []

            for episode_id, episode in self._episodes.items():
                num_steps = len(episode.actions)
                if num_steps == 0:
                    continue

                # Each step index t in [0, num_steps-1] is a valid start
                for t in range(num_steps):
                    valid_positions.append((episode_id, t))

            if not valid_positions:
                return []

            # Sample uniformly from valid positions
            num_samples = min(batch_size, len(valid_positions))
            sampled_positions = random.sample(valid_positions, num_samples)

            # Build n-step samples
            samples: list[NStepSample] = []
            for episode_id, start_idx in sampled_positions:
                sample = self._build_n_step_sample(
                    episode_id=episode_id,
                    start_idx=start_idx,
                    n=n,
                    gamma=gamma,
                )
                samples.append(sample)

            return samples

    def _build_n_step_sample(
        self,
        episode_id: str,
        start_idx: int,
        n: int,
        gamma: float,
    ) -> NStepSample:
        """Build an n-step sample starting from a given position.

        Never crosses episode boundary; truncates if fewer than n steps remain.
        """
        episode = self._episodes[episode_id]

        num_steps = len(episode.actions)

        # Determine actual window size (truncate at episode boundary)
        end_idx = min(start_idx + n, num_steps)
        actual_n = end_idx - start_idx

        # Extract data for the window [start_idx, end_idx)
        obs_t = episode.observations[start_idx]
        action_t = episode.actions[start_idx]
        rewards_seq = episode.rewards[start_idx:end_idx]

        # next_obs is observation at end_idx
        # If end_idx < len(observations), we have it
        # Otherwise episode ended and we need the last observation
        if end_idx < len(episode.observations):
            next_obs = episode.observations[end_idx]
        else:
            # Episode ended; last observation should be at index end_idx-1+1 = end_idx
            # But if observations has length num_steps+1, then end_idx could equal num_steps
            # In that case, the last observation is observations[num_steps]
            # Let's ensure observations has the final obs
            if len(episode.observations) > end_idx:
                next_obs = episode.observations[end_idx]
            else:
                # Fallback: use the last available observation
                next_obs = episode.observations[-1]

        # Check if episode ended within the window
        done = (end_idx == num_steps) and (episode.status != EpisodeStatus.IN_PROGRESS)
        terminal = done and (episode.status == EpisodeStatus.TERMINAL)
        truncated = done and (episode.status == EpisodeStatus.TRUNCATED)

        # Compute discount powers
        discount_powers = [gamma**k for k in range(actual_n)]

        return NStepSample(
            episode_id=episode_id,
            agent_id=episode.agent_id,
            obs_t=obs_t,
            action_t=action_t,
            rewards_seq=rewards_seq,
            next_obs=next_obs,
            done=done,
            truncated=truncated,
            terminal=terminal,
            discount_powers=discount_powers,
            start_step=start_idx,
            actual_n=actual_n,
            gamma=gamma,
        )

    async def _evict_if_needed(self) -> None:
        """Evict oldest episodes if capacity limits are exceeded.

        Eviction policy:
        1. First evict oldest finished episodes (terminal or truncated)
        2. If still over capacity, evict oldest in-progress episodes

        Eviction updates the total step count to maintain correct uniform
        sampling statistics.
        """
        # Check episode capacity
        if self._max_episodes is not None:
            while len(self._episodes) > self._max_episodes:
                self._evict_oldest_episode()

        # Check step capacity
        if self._max_steps is not None:
            while self._total_steps > self._max_steps:
                if not self._episodes:
                    break
                self._evict_oldest_episode()

    def _evict_oldest_episode(self) -> None:
        """Evict the oldest episode from the buffer."""
        if not self._episodes:
            return

        # Separate finished and in-progress episodes
        finished_episodes: list[tuple[str, Episode]] = []
        in_progress_episodes: list[tuple[str, Episode]] = []

        for episode_id, episode in self._episodes.items():
            if episode.status == EpisodeStatus.IN_PROGRESS:
                in_progress_episodes.append((episode_id, episode))
            else:
                finished_episodes.append((episode_id, episode))

        # Evict oldest finished episode first
        if finished_episodes:
            oldest = min(finished_episodes, key=lambda x: x[1].start_time)
            episode_id = oldest[0]
        else:
            # No finished episodes, evict oldest in-progress
            oldest = min(in_progress_episodes, key=lambda x: x[1].start_time)
            episode_id = oldest[0]

        # Remove the episode
        episode = self._episodes.pop(episode_id)
        self._total_steps -= len(episode)

        # Update agent tracking
        agent_id = episode.agent_id
        if agent_id in self._agent_episodes:
            self._agent_episodes[agent_id].remove(episode_id)
            if not self._agent_episodes[agent_id]:
                del self._agent_episodes[agent_id]

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the replay buffer.

        Returns:
            Dictionary with buffer statistics
        """
        async with self._lock:
            num_in_progress = sum(1 for ep in self._episodes.values() if ep.status == EpisodeStatus.IN_PROGRESS)
            num_terminal = sum(1 for ep in self._episodes.values() if ep.status == EpisodeStatus.TERMINAL)
            num_truncated = sum(1 for ep in self._episodes.values() if ep.status == EpisodeStatus.TRUNCATED)

            return {
                "total_episodes": len(self._episodes),
                "in_progress": num_in_progress,
                "terminal": num_terminal,
                "truncated": num_truncated,
                "total_steps": self._total_steps,
                "num_agents": len(self._agent_episodes),
            }

    async def clear(self) -> None:
        """Clear all episodes from the buffer."""
        async with self._lock:
            self._episodes.clear()
            self._agent_episodes.clear()
            self._total_steps = 0
