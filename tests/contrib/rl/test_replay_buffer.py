"""Unit tests for the Episode Replay Buffer."""

import asyncio
import random

import pytest

import ares.contrib.rl.replay_buffer


class TestComputeDiscountedReturn:
    """Test the helper function for computing discounted returns."""

    def test_single_reward(self):
        """Test with a single reward."""
        result = ares.contrib.rl.replay_buffer.compute_discounted_return([5.0], gamma=0.99)
        assert result == 5.0

    def test_multiple_rewards(self):
        """Test with multiple rewards."""
        rewards = [1.0, 2.0, 3.0]
        gamma = 0.9
        expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0
        result = ares.contrib.rl.replay_buffer.compute_discounted_return(rewards, gamma)
        assert abs(result - expected) < 1e-6

    def test_gamma_one(self):
        """Test with gamma=1 (undiscounted)."""
        rewards = [1.0, 2.0, 3.0]
        result = ares.contrib.rl.replay_buffer.compute_discounted_return(rewards, gamma=1.0)
        assert result == 6.0

    def test_empty_rewards(self):
        """Test with empty reward sequence."""
        result = ares.contrib.rl.replay_buffer.compute_discounted_return([], gamma=0.99)
        assert result == 0.0


class TestEpisodeLifecycle:
    """Test basic episode lifecycle operations."""

    @pytest.mark.asyncio
    async def test_start_episode(self):
        """Test starting a new episode."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        assert episode_id.startswith("agent_0_")
        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["in_progress"] == 1

    @pytest.mark.asyncio
    async def test_start_episode_custom_id(self):
        """Test starting an episode with a custom ID."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0", episode_id="custom_episode")

        assert episode_id == "custom_episode"

    @pytest.mark.asyncio
    async def test_start_duplicate_episode_id(self):
        """Test that starting an episode with duplicate ID raises error."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        await buffer.start_episode(agent_id="agent_0", episode_id="ep1")

        with pytest.raises(ValueError, match="already exists"):
            await buffer.start_episode(agent_id="agent_0", episode_id="ep1")

    @pytest.mark.asyncio
    async def test_append_observation_action_reward(self):
        """Test appending experience to an episode."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        # Add first transition
        await buffer.append_observation_action_reward(episode_id, observation=[1, 2, 3], action=0, reward=1.0)

        stats = await buffer.get_stats()
        assert stats["total_steps"] == 1

        # Add second transition
        await buffer.append_observation_action_reward(episode_id, observation=[4, 5, 6], action=1, reward=2.0)

        stats = await buffer.get_stats()
        assert stats["total_steps"] == 2

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_episode(self):
        """Test appending to a non-existent episode raises error."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        with pytest.raises(ValueError, match="not found"):
            await buffer.append_observation_action_reward("nonexistent", observation=[1], action=0, reward=0.0)

    @pytest.mark.asyncio
    async def test_end_episode_terminal(self):
        """Test ending an episode as terminal."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        await buffer.append_observation_action_reward(episode_id, observation=[1], action=0, reward=1.0)

        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[2]
        )

        stats = await buffer.get_stats()
        assert stats["terminal"] == 1
        assert stats["in_progress"] == 0

    @pytest.mark.asyncio
    async def test_end_episode_truncated(self):
        """Test ending an episode as truncated."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        await buffer.append_observation_action_reward(episode_id, observation=[1], action=0, reward=1.0)

        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TRUNCATED, final_observation=[2]
        )

        stats = await buffer.get_stats()
        assert stats["truncated"] == 1
        assert stats["in_progress"] == 0

    @pytest.mark.asyncio
    async def test_end_episode_prevents_further_appends(self):
        """Test that ending an episode prevents further appends."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        await buffer.append_observation_action_reward(episode_id, observation=[1], action=0, reward=1.0)

        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[2]
        )

        # Try to append after ending
        with pytest.raises(ValueError, match="Cannot append to finished episode"):
            await buffer.append_observation_action_reward(episode_id, observation=[3], action=1, reward=2.0)

    @pytest.mark.asyncio
    async def test_end_episode_already_finished(self):
        """Test that ending an already finished episode raises error."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        await buffer.append_observation_action_reward(episode_id, observation=[1], action=0, reward=1.0)

        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[2]
        )

        with pytest.raises(ValueError, match="already finished"):
            await buffer.end_episode(
                episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[3]
            )

    @pytest.mark.asyncio
    async def test_end_episode_with_in_progress_status(self):
        """Test that ending with IN_PROGRESS status raises error."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        with pytest.raises(ValueError, match="Cannot end episode with status IN_PROGRESS"):
            await buffer.end_episode(episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.IN_PROGRESS)


class TestStorageFormat:
    """Test that storage format avoids duplication of states."""

    @pytest.mark.asyncio
    async def test_no_state_duplication(self):
        """Test that next_obs is derived from subsequent observation, not duplicated."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        # Build a simple episode
        obs_0 = {"data": [0]}
        obs_1 = {"data": [1]}
        obs_2 = {"data": [2]}

        await buffer.append_observation_action_reward(episode_id, observation=obs_0, action=0, reward=1.0)
        await buffer.append_observation_action_reward(episode_id, observation=obs_1, action=1, reward=2.0)

        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=obs_2
        )

        # Sample and verify next_obs matches subsequent observation
        samples = await buffer.sample_n_step(batch_size=2, n=1, gamma=0.99)

        # Find sample starting at t=0
        sample_0 = next(s for s in samples if s.start_step == 0)
        assert sample_0.obs_t == obs_0
        assert sample_0.next_obs == obs_1  # Derived from observations[1]

        # Find sample starting at t=1
        sample_1 = next(s for s in samples if s.start_step == 1)
        assert sample_1.obs_t == obs_1
        assert sample_1.next_obs == obs_2  # Derived from observations[2]


class TestConcurrency:
    """Test concurrent episode appending."""

    @pytest.mark.asyncio
    async def test_concurrent_episode_appends(self):
        """Test multiple episodes appended concurrently via asyncio tasks."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        async def fill_episode(agent_id: str, num_steps: int):
            """Fill an episode with num_steps transitions."""
            episode_id = await buffer.start_episode(agent_id=agent_id)
            for t in range(num_steps):
                await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=float(t))
            await buffer.end_episode(
                episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[num_steps]
            )

        # Run multiple episodes concurrently
        tasks = [
            fill_episode("agent_0", 10),
            fill_episode("agent_1", 20),
            fill_episode("agent_2", 15),
        ]
        await asyncio.gather(*tasks)

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3
        assert stats["total_steps"] == 10 + 20 + 15
        assert stats["terminal"] == 3

    @pytest.mark.asyncio
    async def test_concurrent_writes_and_reads(self):
        """Test concurrent writes (appends) and reads (sampling)."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Pre-fill some episodes
        for i in range(3):
            episode_id = await buffer.start_episode(agent_id=f"agent_{i}")
            for t in range(10):
                await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=1.0)
            await buffer.end_episode(
                episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[10]
            )

        async def writer():
            """Write new episodes."""
            for i in range(3, 6):
                episode_id = await buffer.start_episode(agent_id=f"agent_{i}")
                for t in range(5):
                    await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=1.0)
                await buffer.end_episode(
                    episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
                )
                await asyncio.sleep(0.001)  # Small delay to allow interleaving

        async def reader():
            """Sample from the buffer."""
            for _ in range(10):
                samples = await buffer.sample_n_step(batch_size=5, n=3, gamma=0.99)
                assert len(samples) > 0
                await asyncio.sleep(0.001)

        # Run writer and reader concurrently
        await asyncio.gather(writer(), reader())

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 6


class TestUniformSampling:
    """Test uniform sampling over experiences."""

    @pytest.mark.asyncio
    async def test_uniform_over_steps_not_episodes(self):
        """
        Test that sampling is uniform over steps, not episodes.

        Create episodes with different lengths and verify that longer episodes
        contribute proportionally more samples.
        """
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Episode 1: 10 steps
        ep1 = await buffer.start_episode(agent_id="agent_0")
        for t in range(10):
            await buffer.append_observation_action_reward(ep1, observation={"ep": 1, "t": t}, action=t, reward=1.0)
        await buffer.end_episode(
            ep1, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation={"ep": 1, "t": 10}
        )

        # Episode 2: 30 steps (3x longer)
        ep2 = await buffer.start_episode(agent_id="agent_1")
        for t in range(30):
            await buffer.append_observation_action_reward(ep2, observation={"ep": 2, "t": t}, action=t, reward=1.0)
        await buffer.end_episode(
            ep2, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation={"ep": 2, "t": 30}
        )

        # Sample many times and count samples from each episode
        num_samples = 1000
        samples = await buffer.sample_n_step(batch_size=num_samples, n=1, gamma=0.99)

        ep1_count = sum(1 for s in samples if s.episode_id == ep1)
        ep2_count = sum(1 for s in samples if s.episode_id == ep2)

        # Expect ratio close to 1:3 (episode 2 is 3x longer)
        # Allow some variance due to randomness
        ratio = ep2_count / ep1_count if ep1_count > 0 else 0
        assert 2.0 < ratio < 4.0, f"Expected ratio ~3, got {ratio}"

    @pytest.mark.asyncio
    async def test_all_steps_have_equal_probability(self):
        """Test that all steps across episodes are equally likely."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Create 3 episodes with 10 steps each
        episode_ids = []
        for i in range(3):
            ep = await buffer.start_episode(agent_id=f"agent_{i}")
            episode_ids.append(ep)
            for t in range(10):
                await buffer.append_observation_action_reward(ep, observation=[i, t], action=t, reward=1.0)
            await buffer.end_episode(
                ep, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[i, 10]
            )

        # Sample exhaustively (all 30 steps)
        samples = await buffer.sample_n_step(batch_size=30, n=1, gamma=0.99)

        # Check we got all 30 unique steps
        assert len(samples) == 30

        # Verify each episode contributes 10 samples
        for ep_id in episode_ids:
            count = sum(1 for s in samples if s.episode_id == ep_id)
            assert count == 10


class TestNStepSampling:
    """Test n-step sampling with boundary handling."""

    @pytest.mark.asyncio
    async def test_n_step_basic(self):
        """Test basic n-step sampling."""
        random.seed(42)
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        # Create episode with 5 steps
        for t in range(5):
            await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=float(t + 1))
        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        # Sample with n=3 starting from t=0
        samples = await buffer.sample_n_step(batch_size=1, n=3, gamma=0.9)
        sample = samples[0]

        # Should get 3 steps: rewards [1, 2, 3]
        assert sample.obs_t == [0]
        assert sample.action_t == 0
        assert sample.rewards_seq == [1.0, 2.0, 3.0]
        assert sample.next_obs == [3]
        assert sample.actual_n == 3
        assert not sample.done

    @pytest.mark.asyncio
    async def test_n_step_truncation_at_boundary(self):
        """Test that n-step sample truncates at episode boundary."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        # Create episode with only 3 steps
        for t in range(3):
            await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=float(t + 1))
        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[3]
        )

        # Request n=5 but only 3 steps available from t=0
        # Should get all 3 steps and truncate
        samples = await buffer.sample_n_step(batch_size=10, n=5, gamma=0.9)

        # Find sample starting at t=0
        sample_0 = next(s for s in samples if s.start_step == 0)
        assert sample_0.actual_n == 3
        assert sample_0.rewards_seq == [1.0, 2.0, 3.0]
        assert sample_0.done  # Episode ended
        assert sample_0.terminal

    @pytest.mark.asyncio
    async def test_n_step_never_crosses_episode_boundary(self):
        """Test that n-step sampling never crosses episode boundaries."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Create two short episodes
        ep1 = await buffer.start_episode(agent_id="agent_0")
        for t in range(3):
            await buffer.append_observation_action_reward(ep1, observation={"ep": 1, "t": t}, action=t, reward=1.0)
        await buffer.end_episode(
            ep1, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation={"ep": 1, "t": 3}
        )

        ep2 = await buffer.start_episode(agent_id="agent_1")
        for t in range(3):
            await buffer.append_observation_action_reward(ep2, observation={"ep": 2, "t": t}, action=t, reward=2.0)
        await buffer.end_episode(
            ep2, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation={"ep": 2, "t": 3}
        )

        # Sample with large n
        samples = await buffer.sample_n_step(batch_size=10, n=10, gamma=0.9)

        # Verify no sample crosses episodes
        for sample in samples:
            # Check that all rewards come from the same episode
            if sample.episode_id == ep1:
                # All observations should have ep=1
                assert sample.obs_t["ep"] == 1
                assert sample.next_obs["ep"] == 1
            else:
                assert sample.obs_t["ep"] == 2
                assert sample.next_obs["ep"] == 2

            # No sample should exceed 3 steps (episode length)
            assert sample.actual_n <= 3

    @pytest.mark.asyncio
    async def test_n_step_near_end_truncates(self):
        """Test n-step sampling near episode end truncates properly."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        # Create episode with 5 steps
        for t in range(5):
            await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=float(t + 1))
        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        # Sample starting from t=3 with n=3
        # Should only get 2 steps (t=3, t=4) because episode has only 5 steps total
        samples = await buffer.sample_n_step(batch_size=10, n=3, gamma=0.9)

        # Find sample starting at t=3
        sample_3 = next(s for s in samples if s.start_step == 3)
        assert sample_3.actual_n == 2
        assert sample_3.rewards_seq == [4.0, 5.0]
        assert sample_3.done

    @pytest.mark.asyncio
    async def test_n_step_discount_powers(self):
        """Test that discount powers are correctly computed."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        episode_id = await buffer.start_episode(agent_id="agent_0")

        for t in range(5):
            await buffer.append_observation_action_reward(episode_id, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            episode_id, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        gamma = 0.9
        samples = await buffer.sample_n_step(batch_size=1, n=4, gamma=gamma)
        sample = samples[0]

        expected_powers = [gamma**k for k in range(sample.actual_n)]
        assert sample.discount_powers == expected_powers

    @pytest.mark.asyncio
    async def test_n_step_terminal_vs_truncated(self):
        """Test that terminal and truncated flags are set correctly."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Terminal episode
        ep1 = await buffer.start_episode(agent_id="agent_0")
        for t in range(3):
            await buffer.append_observation_action_reward(ep1, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep1, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[3]
        )

        # Truncated episode
        ep2 = await buffer.start_episode(agent_id="agent_1")
        for t in range(3):
            await buffer.append_observation_action_reward(ep2, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep2, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TRUNCATED, final_observation=[3]
        )

        # Sample with n that includes the end
        samples = await buffer.sample_n_step(batch_size=10, n=5, gamma=0.9)

        # Find sample from terminal episode starting at end
        terminal_samples = [s for s in samples if s.episode_id == ep1 and s.start_step == 2]
        if terminal_samples:
            assert terminal_samples[0].done
            assert terminal_samples[0].terminal
            assert not terminal_samples[0].truncated

        # Find sample from truncated episode
        truncated_samples = [s for s in samples if s.episode_id == ep2 and s.start_step == 2]
        if truncated_samples:
            assert truncated_samples[0].done
            assert truncated_samples[0].truncated
            assert not truncated_samples[0].terminal


class TestCapacityAndEviction:
    """Test capacity management and eviction behavior."""

    @pytest.mark.asyncio
    async def test_max_episodes_eviction(self):
        """Test that max_episodes limit triggers eviction."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer(max_episodes=3)

        # Add 3 episodes (at capacity)
        for i in range(3):
            ep = await buffer.start_episode(agent_id=f"agent_{i}")
            await buffer.append_observation_action_reward(ep, observation=[i], action=i, reward=1.0)
            await buffer.end_episode(
                ep, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[i + 1]
            )

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3

        # Add 4th episode, should evict oldest
        ep4 = await buffer.start_episode(agent_id="agent_3")
        await buffer.append_observation_action_reward(ep4, observation=[3], action=3, reward=1.0)
        await buffer.end_episode(
            ep4, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[4]
        )

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3  # Still at max

    @pytest.mark.asyncio
    async def test_max_steps_eviction(self):
        """Test that max_steps limit triggers eviction."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer(max_steps=10)

        # Add episodes totaling 10 steps
        ep1 = await buffer.start_episode(agent_id="agent_0")
        for t in range(5):
            await buffer.append_observation_action_reward(ep1, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep1, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        ep2 = await buffer.start_episode(agent_id="agent_1")
        for t in range(5):
            await buffer.append_observation_action_reward(ep2, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep2, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        stats = await buffer.get_stats()
        assert stats["total_steps"] == 10

        # Add more steps, should trigger eviction
        ep3 = await buffer.start_episode(agent_id="agent_2")
        for t in range(3):
            await buffer.append_observation_action_reward(ep3, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep3, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[3]
        )

        stats = await buffer.get_stats()
        # Should have evicted ep1, keeping ep2 and ep3
        assert stats["total_steps"] <= 10

    @pytest.mark.asyncio
    async def test_eviction_prefers_finished_episodes(self):
        """Test that eviction prefers finished episodes over in-progress."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer(max_episodes=3)

        # Add 2 finished episodes
        for i in range(2):
            ep = await buffer.start_episode(agent_id=f"agent_{i}")
            await buffer.append_observation_action_reward(ep, observation=[i], action=i, reward=1.0)
            await buffer.end_episode(
                ep, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[i + 1]
            )

        # Add 1 in-progress episode
        ep_in_progress = await buffer.start_episode(agent_id="agent_in_progress")
        await buffer.append_observation_action_reward(ep_in_progress, observation=[99], action=99, reward=1.0)

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3
        assert stats["in_progress"] == 1
        assert stats["terminal"] == 2

        # Add another episode, should evict oldest finished, not in-progress
        ep_new = await buffer.start_episode(agent_id="agent_new")
        await buffer.append_observation_action_reward(ep_new, observation=[100], action=100, reward=1.0)
        await buffer.end_episode(
            ep_new, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[101]
        )

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3
        assert stats["in_progress"] == 1  # In-progress episode should still be there

    @pytest.mark.asyncio
    async def test_eviction_updates_sampling_counts(self):
        """Test that eviction correctly updates total_steps for sampling."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer(max_episodes=2)

        # Add 2 episodes
        ep1 = await buffer.start_episode(agent_id="agent_0")
        for t in range(10):
            await buffer.append_observation_action_reward(ep1, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep1, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[10]
        )

        ep2 = await buffer.start_episode(agent_id="agent_1")
        for t in range(5):
            await buffer.append_observation_action_reward(ep2, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep2, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[5]
        )

        stats = await buffer.get_stats()
        assert stats["total_steps"] == 15

        # Add 3rd episode, should evict ep1
        ep3 = await buffer.start_episode(agent_id="agent_2")
        for t in range(7):
            await buffer.append_observation_action_reward(ep3, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(
            ep3, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[7]
        )

        stats = await buffer.get_stats()
        # Should have ep2 (5 steps) + ep3 (7 steps) = 12 steps
        assert stats["total_steps"] == 12

        # Sampling should still work correctly
        samples = await buffer.sample_n_step(batch_size=12, n=1, gamma=0.9)
        assert len(samples) == 12


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_sample_empty_buffer(self):
        """Test sampling from an empty buffer returns empty list."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        samples = await buffer.sample_n_step(batch_size=10, n=3, gamma=0.9)
        assert samples == []

    @pytest.mark.asyncio
    async def test_sample_with_only_empty_episodes(self):
        """Test sampling when episodes have no steps."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        await buffer.start_episode(agent_id="agent_0")

        samples = await buffer.sample_n_step(batch_size=10, n=3, gamma=0.9)
        assert samples == []

    @pytest.mark.asyncio
    async def test_sample_n_less_than_one(self):
        """Test that n < 1 raises ValueError."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        with pytest.raises(ValueError, match="n must be >= 1"):
            await buffer.sample_n_step(batch_size=10, n=0, gamma=0.9)

    @pytest.mark.asyncio
    async def test_sample_invalid_gamma(self):
        """Test that invalid gamma raises ValueError."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()
        with pytest.raises(ValueError, match="gamma must be in"):
            await buffer.sample_n_step(batch_size=10, n=3, gamma=0.0)
        with pytest.raises(ValueError, match="gamma must be in"):
            await buffer.sample_n_step(batch_size=10, n=3, gamma=1.5)

    @pytest.mark.asyncio
    async def test_clear_buffer(self):
        """Test clearing the buffer."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        # Add some episodes
        for i in range(3):
            ep = await buffer.start_episode(agent_id=f"agent_{i}")
            await buffer.append_observation_action_reward(ep, observation=[i], action=i, reward=1.0)
            await buffer.end_episode(
                ep, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[i + 1]
            )

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 3

        # Clear
        await buffer.clear()

        stats = await buffer.get_stats()
        assert stats["total_episodes"] == 0
        assert stats["total_steps"] == 0

    @pytest.mark.asyncio
    async def test_sample_batch_size_larger_than_available(self):
        """Test that sampling returns fewer samples if not enough data."""
        buffer = ares.contrib.rl.replay_buffer.EpisodeReplayBuffer()

        ep = await buffer.start_episode(agent_id="agent_0")
        for t in range(3):
            await buffer.append_observation_action_reward(ep, observation=[t], action=t, reward=1.0)
        await buffer.end_episode(ep, status=ares.contrib.rl.replay_buffer.EpisodeStatus.TERMINAL, final_observation=[3])

        # Request 100 samples but only 3 available
        samples = await buffer.sample_n_step(batch_size=100, n=1, gamma=0.9)
        assert len(samples) == 3
