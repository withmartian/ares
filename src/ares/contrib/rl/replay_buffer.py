"""Episode Replay Buffer for Multi-Agent Reinforcement Learning.

This module provides an asyncio-safe replay buffer that supports:
- Concurrent experience collection from multiple agents
- Episode-based storage with explicit start/end control
- N-step return sampling with configurable gamma
- Capacity management with automatic eviction of oldest episodes

Storage Design:
    Episodes store per-timestep arrays: observations[t], actions[t], rewards[t].

Usage Example:
    ```python
    import asyncio
    from ares.contrib.rl import replay_buffer

    # Create buffer with capacity limits
    buffer = replay_buffer.EpisodeReplayBuffer(max_episodes=1000, max_steps=100000)

    # Start an episode (episode_id is auto-generated)
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

    # End episode with final observation
    await buffer.end_episode(episode_id, status="COMPLETED", final_observation=obs_2)

    # Sample n-step batches
    samples = await buffer.sample_n_step(batch_size=32, n=3, gamma=0.99)
    for sample in samples:
        # sample.obs_t, sample.action_t, sample.rewards_seq
        # sample.next_obs, sample.terminal, etc.
        discounted_return = replay_buffer.compute_discounted_return(
            sample.rewards_seq, gamma=0.99
        )
    ```

Thread Safety and Async Usage:
    All public methods are async. This buffer is designed for single-threaded
    asyncio usage and does not provide internal synchronization. If you need
    concurrent access from multiple asyncio tasks, you should manage
    synchronization externally.
"""

import collections
from collections import deque
import dataclasses
import random
import time
from typing import Any, Literal
import uuid

# Type of episode status
EpisodeStatus = Literal["IN_PROGRESS", "COMPLETED"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Episode[ObservationType, ActionType]:
    """An episode containing sequences of observations, actions, and rewards.

    Storage format:
        - observations: [obs_0, obs_1, ..., obs_T]  (length T+1)
        - actions: [a_0, a_1, ..., a_{T-1}]          (length T)
        - rewards: [r_0, r_1, ..., r_{T-1}]          (length T)

    A full transition at timestep t consists of (obs_t, action_t, reward_t, obs_{t+1}).
    The next observation obs_{t+1} is stored at observations[t+1].

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
    observations: list[ObservationType] = dataclasses.field(default_factory=list)
    actions: list[ActionType] = dataclasses.field(default_factory=list)
    rewards: list[float] = dataclasses.field(default_factory=list)
    status: EpisodeStatus = "IN_PROGRESS"
    start_time: float = dataclasses.field(default_factory=time.time)

    def __len__(self) -> int:
        """Return the number of complete transitions (with both obs_t and obs_{t+1} available).

        A complete transition requires observations[t] and observations[t+1].
        Returns max(len(observations) - 1, 0).
        """
        return max(len(self.observations) - 1, 0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReplaySample[ObservationType, ActionType]:
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
        terminal: True if episode terminated within the n-step window
        next_discount: Discount factor to apply to bootstrap value at next_obs.
                      This is gamma^m where m is actual_n. When terminal=True, this
                      should be 0 (no bootstrap). When terminal=False, this is the
                      discount to apply to the value estimate at next_obs.
        discount_powers: [gamma^0, gamma^1, ..., gamma^{m-1}] for computing returns
        start_step: The starting step index t
        actual_n: The actual number of steps m (may be < n if episode ends)
        gamma: The discount factor used
    """

    episode_id: str
    agent_id: str
    obs_t: ObservationType
    action_t: ActionType
    rewards_seq: list[float]
    next_obs: ObservationType
    terminal: bool
    next_discount: float
    discount_powers: list[float]
    start_step: int
    actual_n: int
    gamma: float

    @property
    def reward(self) -> float:
        """Return the computed discounted return for this sample.

        This is a convenience property that computes the discounted return
        from the rewards sequence using the stored gamma value.

        Returns:
            The discounted return: sum_{k=0}^{n-1} gamma^k * r_k
        """
        return compute_discounted_return(self.rewards_seq, self.gamma)


# Backward compatibility alias
NStepSample = ReplaySample


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
    """Replay buffer for episodic reinforcement learning.

    This buffer stores complete episodes and supports n-step sampling with
    proper handling of episode boundaries. It manages capacity by evicting
    oldest finished episodes first, then oldest in-progress episodes if needed.

    Sampling:
        Uniform sampling over all valid time steps (experiences) across episodes.
        Each valid step (obs_t, action_t, reward_t) has equal probability.
        The implementation uses O(num_episodes) scan to build episode weights,
        then O(num_episodes) weighted sampling for each sample, avoiding the
        O(num_episodes * steps_per_episode) cost of enumerating all positions.

    Concurrency:
        This buffer is designed for single-threaded usage. If you need
        concurrent access from multiple asyncio tasks, you should manage
        synchronization externally.

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
        self._episodes: dict[str, Episode[Any, Any]] = {}
        self._max_episodes = max_episodes
        self._max_steps = max_steps
        self._total_steps = 0

        # Track episodes by agent for potential future use
        self._episodes_by_agent: dict[str, list[str]] = collections.defaultdict(list)

        # Track episode IDs in insertion order for efficient sampling and eviction
        # This deque parallels self._episodes and enables O(1) oldest episode access
        self._episode_order: deque[str] = deque()

    async def start_episode(
        self,
        agent_id: str,
    ) -> str:
        """Start a new episode.

        Args:
            agent_id: Identifier for the agent

        Returns:
            The episode_id for this episode
        """
        episode_id = str(uuid.uuid4())

        episode = Episode(episode_id=episode_id, agent_id=agent_id)
        self._episodes[episode_id] = episode
        self._episodes_by_agent[agent_id].append(episode_id)
        self._episode_order.append(episode_id)

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
        if episode_id not in self._episodes:
            raise ValueError(f"Episode {episode_id} not found")

        episode = self._episodes[episode_id]

        if episode.status != "IN_PROGRESS":
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
        status: EpisodeStatus = "COMPLETED",
        final_observation: Any = None,
    ) -> None:
        """Mark an episode as finished.

        Args:
            episode_id: The episode to end
            status: Episode status (should be "COMPLETED")
            final_observation: Final observation obs_T after last action. Required if not
                             already appended via append_observation_action_reward.

        Raises:
            ValueError: If episode doesn't exist, is already finished,
                       status is IN_PROGRESS, or final_observation is required but not provided
        """
        if episode_id not in self._episodes:
            raise ValueError(f"Episode {episode_id} not found")

        episode = self._episodes[episode_id]

        if episode.status != "IN_PROGRESS":
            raise ValueError(f"Episode {episode_id} is already finished")

        if status == "IN_PROGRESS":
            raise ValueError("Cannot end episode with status IN_PROGRESS")

        # Validation: If observations length equals actions length,
        # the final observation hasn't been added yet, so it must be provided
        if len(episode.observations) == len(episode.actions):
            if final_observation is None:
                raise ValueError(
                    f"Episode {episode_id} requires final_observation: "
                    f"observations length ({len(episode.observations)}) equals "
                    f"actions length ({len(episode.actions)})"
                )
            episode.observations.append(final_observation)
        elif final_observation is not None:
            # If final_observation is provided but not needed, append it anyway
            episode.observations.append(final_observation)

        # Update status using object.__setattr__ since Episode dataclass is frozen
        object.__setattr__(episode, "status", status)

    def _get_valid_step_count(self, episode: Episode) -> int:
        """Get the number of valid starting positions for sampling in an episode.

        Args:
            episode: The episode to check

        Returns:
            Number of valid starting positions (0 if none)
        """
        num_steps = len(episode.actions)
        if num_steps == 0:
            return 0

        # For COMPLETED episodes, all steps are valid
        if episode.status == "COMPLETED":
            return num_steps

        # For IN_PROGRESS episodes, only steps with next observation available
        # A step t is valid if observations[t+1] exists
        # Since len(observations) can be at most len(actions) + 1,
        # valid steps are those where t+1 < len(observations)
        valid_count = max(0, len(episode.observations) - 1)
        return min(valid_count, num_steps)

    async def sample_n_step(
        self,
        batch_size: int,
        n: int,
        gamma: float,
    ) -> list[ReplaySample]:
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

        # Build episode ranges using the deque for iteration order
        # This avoids iterating over self._episodes.items() directly
        episode_ranges: list[tuple[int, int, str]] = []  # (start_pos, end_pos, episode_id)
        cumulative_pos = 0

        for episode_id in self._episode_order:
            episode = self._episodes[episode_id]
            valid_count = self._get_valid_step_count(episode)
            if valid_count > 0:
                episode_ranges.append((cumulative_pos, cumulative_pos + valid_count, episode_id))
                cumulative_pos += valid_count

        if not episode_ranges:
            return []

        total_valid_positions = cumulative_pos

        # Sample uniformly without replacement
        num_samples = min(batch_size, total_valid_positions)

        # Generate unique random positions
        sampled_global_positions = random.sample(range(total_valid_positions), num_samples)

        # Build n-step samples by mapping global positions back to (episode_id, step_idx)
        samples: list[ReplaySample] = []
        for global_pos in sampled_global_positions:
            # Find the episode containing this position
            for start_pos, end_pos, episode_id in episode_ranges:
                if start_pos <= global_pos < end_pos:
                    # Convert global position to local step index within episode
                    start_idx = global_pos - start_pos

                    sample = self._build_n_step_sample(
                        episode_id=episode_id,
                        start_idx=start_idx,
                        n=n,
                        gamma=gamma,
                    )
                    samples.append(sample)
                    break

        return samples

    def _build_n_step_sample(
        self,
        episode_id: str,
        start_idx: int,
        n: int,
        gamma: float,
    ) -> ReplaySample:
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
        # We can safely access this because _get_valid_step_count ensures
        # that only positions with next_obs available are sampled
        next_obs = episode.observations[end_idx]

        # Check if episode ended within the window
        terminal = (end_idx == num_steps) and (episode.status != "IN_PROGRESS")

        # Compute next_discount: gamma^m when not terminal, 0 when terminal
        next_discount = 0.0 if terminal else gamma**actual_n

        # Compute discount powers
        discount_powers = [gamma**k for k in range(actual_n)]

        return ReplaySample(
            episode_id=episode_id,
            agent_id=episode.agent_id,
            obs_t=obs_t,
            action_t=action_t,
            rewards_seq=rewards_seq,
            next_obs=next_obs,
            terminal=terminal,
            next_discount=next_discount,
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
            if episode.status == "IN_PROGRESS":
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
        if agent_id in self._episodes_by_agent:
            self._episodes_by_agent[agent_id].remove(episode_id)
            if not self._episodes_by_agent[agent_id]:
                del self._episodes_by_agent[agent_id]

        # Remove from episode order deque
        self._episode_order.remove(episode_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the replay buffer.

        Returns:
            Dictionary with buffer statistics
        """
        num_in_progress = sum(1 for ep in self._episodes.values() if ep.status == "IN_PROGRESS")
        num_completed = sum(1 for ep in self._episodes.values() if ep.status == "COMPLETED")

        return {
            "total_episodes": len(self._episodes),
            "in_progress": num_in_progress,
            "completed": num_completed,
            "total_steps": self._total_steps,
            "num_agents": len(self._episodes_by_agent),
        }

    async def clear(self) -> None:
        """Clear all episodes from the buffer."""
        self._episodes.clear()
        self._episodes_by_agent.clear()
        self._episode_order.clear()
        self._total_steps = 0
