"""Tests for episode-based replay buffer."""

import asyncio
from collections import Counter

import numpy as np
import pytest

from ares.contrib.rl.replay_buffer import EpisodeStatus
from ares.contrib.rl.replay_buffer import ReplayBuffer


@pytest.mark.asyncio
async def test_start_episode():
    """Test starting a new episode."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    assert episode_id == "agent_1_episode_0"
    assert buffer.get_num_episodes() == 1

    custom_id = await buffer.start_episode("agent_1", "custom_episode")
    assert custom_id == "custom_episode"
    assert buffer.get_num_episodes() == 2


@pytest.mark.asyncio
async def test_duplicate_episode_id():
    """Test that duplicate episode IDs raise an error."""
    buffer = ReplayBuffer()

    await buffer.start_episode("agent_1", "episode_1")
    with pytest.raises(ValueError, match="already exists"):
        await buffer.start_episode("agent_1", "episode_1")


@pytest.mark.asyncio
async def test_append_step():
    """Test appending steps to an episode."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    await buffer.append_step(episode_id, [1, 2, 3], [0], 1.0)
    await buffer.append_step(episode_id, [4, 5, 6], [1], 2.0)

    assert buffer.get_total_steps() == 2


@pytest.mark.asyncio
async def test_append_to_nonexistent_episode():
    """Test that appending to a non-existent episode raises an error."""
    buffer = ReplayBuffer()

    with pytest.raises(ValueError, match="does not exist"):
        await buffer.append_step("nonexistent", [1, 2, 3], [0], 1.0)


@pytest.mark.asyncio
async def test_end_episode():
    """Test ending an episode."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    await buffer.append_step(episode_id, [1, 2, 3], [0], 1.0)
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    assert buffer.get_num_finished_episodes() == 1


@pytest.mark.asyncio
async def test_end_episode_prevents_further_appends():
    """Test that ending an episode prevents further appends."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    await buffer.append_step(episode_id, [1, 2, 3], [0], 1.0)
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    with pytest.raises(ValueError, match="Cannot append to finished episode"):
        await buffer.append_step(episode_id, [4, 5, 6], [1], 2.0)


@pytest.mark.asyncio
async def test_end_episode_with_in_progress_status():
    """Test that ending with IN_PROGRESS status raises an error."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    with pytest.raises(ValueError, match="Cannot end episode with IN_PROGRESS status"):
        await buffer.end_episode(episode_id, EpisodeStatus.IN_PROGRESS)


@pytest.mark.asyncio
async def test_concurrent_appends_multiple_episodes():
    """Test concurrent appends to multiple episodes."""
    buffer = ReplayBuffer()

    episode_1 = await buffer.start_episode("agent_1")
    episode_2 = await buffer.start_episode("agent_2")

    async def add_steps_to_episode(episode_id: str, num_steps: int) -> None:
        for i in range(num_steps):
            await buffer.append_step(
                episode_id,
                [i] * 3,
                [i % 2],
                float(i),
            )
            await asyncio.sleep(0.001)

    await asyncio.gather(
        add_steps_to_episode(episode_1, 10),
        add_steps_to_episode(episode_2, 15),
    )

    assert buffer.get_total_steps() == 25
    assert buffer.get_num_episodes() == 2

    await buffer.end_episode(episode_1, EpisodeStatus.TERMINAL)
    await buffer.end_episode(episode_2, EpisodeStatus.TRUNCATED)

    assert buffer.get_num_finished_episodes() == 2


@pytest.mark.asyncio
async def test_sample_empty_buffer():
    """Test that sampling from an empty buffer raises an error."""
    buffer = ReplayBuffer()

    with pytest.raises(ValueError, match="No finished episodes"):
        await buffer.sample_n_step(batch_size=1, n=3)


@pytest.mark.asyncio
async def test_sample_no_finished_episodes():
    """Test that sampling with only in-progress episodes raises an error."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    await buffer.append_step(episode_id, [1, 2, 3], [0], 1.0)

    with pytest.raises(ValueError, match="No finished episodes"):
        await buffer.sample_n_step(batch_size=1, n=3)


@pytest.mark.asyncio
async def test_sample_n_step_basic():
    """Test basic n-step sampling."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    for i in range(5):
        await buffer.append_step(episode_id, [i] * 3, i % 2, float(i))
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    sample = await buffer.sample_n_step(batch_size=2, n=3)

    assert sample.obs_t.shape == (2, 3)
    assert sample.action_t.shape == (2,)
    assert sample.reward_seq.shape[0] == 2
    assert sample.next_obs.shape == (2, 3)
    assert sample.m.shape == (2,)
    assert sample.terminal.shape == (2,)
    assert sample.truncated.shape == (2,)


@pytest.mark.asyncio
async def test_sample_n_step_truncation_near_end():
    """Test that n-step sampling truncates properly near episode end."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    for i in range(5):
        await buffer.append_step(episode_id, [i], [i % 2], float(i))
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    samples = []
    for _ in range(100):
        sample = await buffer.sample_n_step(batch_size=1, n=10)
        samples.append(
            {
                "obs_t": sample.obs_t[0, 0],
                "m": sample.m[0],
                "terminal": sample.terminal[0],
            }
        )

    truncated_samples = [s for s in samples if s["m"] < 10]
    assert len(truncated_samples) > 0

    for s in samples:
        if s["obs_t"] == 4:
            assert s["m"] == 1
            assert s["terminal"]
        elif s["obs_t"] == 3:
            assert s["m"] <= 2


@pytest.mark.asyncio
async def test_sample_uniform_over_steps():
    """Test that sampling is approximately uniform over steps."""
    buffer = ReplayBuffer()

    episode_1 = await buffer.start_episode("agent_1")
    for i in range(10):
        await buffer.append_step(episode_1, [1, i], [0], float(i))
    await buffer.end_episode(episode_1, EpisodeStatus.TERMINAL)

    episode_2 = await buffer.start_episode("agent_2")
    for i in range(20):
        await buffer.append_step(episode_2, [2, i], [0], float(i))
    await buffer.end_episode(episode_2, EpisodeStatus.TERMINAL)

    episode_samples = []
    num_samples = 3000
    for _ in range(num_samples):
        sample = await buffer.sample_n_step(batch_size=1, n=1)
        episode_id = sample.obs_t[0, 0]
        episode_samples.append(episode_id)

    counter = Counter(episode_samples)
    ratio = counter[2] / counter[1]

    assert 1.5 < ratio < 2.5


@pytest.mark.asyncio
async def test_sample_batch_size_exceeds_available():
    """Test that sampling more than available steps raises an error."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    for i in range(5):
        await buffer.append_step(episode_id, [i], [0], float(i))
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    with pytest.raises(ValueError, match="exceeds available steps"):
        await buffer.sample_n_step(batch_size=10, n=3)


@pytest.mark.asyncio
async def test_eviction_max_episodes():
    """Test eviction based on max_episodes."""
    buffer = ReplayBuffer(max_episodes=2)

    for i in range(5):
        episode_id = await buffer.start_episode(f"agent_{i}")
        await buffer.append_step(episode_id, [i], [0], float(i))
        await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    assert buffer.get_num_finished_episodes() <= 2
    assert buffer.get_num_episodes() <= 3


@pytest.mark.asyncio
async def test_eviction_max_steps():
    """Test eviction based on max_steps."""
    buffer = ReplayBuffer(max_steps=15)

    for i in range(5):
        episode_id = await buffer.start_episode(f"agent_{i}")
        for j in range(5):
            await buffer.append_step(episode_id, [i, j], [0], float(j))
        await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    assert buffer.get_total_steps() <= 20


@pytest.mark.asyncio
async def test_eviction_oldest_first():
    """Test that eviction removes oldest episodes first."""
    buffer = ReplayBuffer(max_episodes=2)

    ep1 = await buffer.start_episode("agent_1")
    await buffer.append_step(ep1, [1], [0], 1.0)
    await buffer.end_episode(ep1, EpisodeStatus.TERMINAL)

    ep2 = await buffer.start_episode("agent_2")
    await buffer.append_step(ep2, [2], [0], 2.0)
    await buffer.end_episode(ep2, EpisodeStatus.TERMINAL)

    ep3 = await buffer.start_episode("agent_3")
    await buffer.append_step(ep3, [3], [0], 3.0)
    await buffer.end_episode(ep3, EpisodeStatus.TERMINAL)

    sample = await buffer.sample_n_step(batch_size=2, n=1)
    sampled_obs = set(sample.obs_t[:, 0].tolist())

    assert 1 not in sampled_obs
    assert 2 in sampled_obs or 3 in sampled_obs


@pytest.mark.asyncio
async def test_terminal_vs_truncated_status():
    """Test that terminal and truncated flags are set correctly."""
    buffer = ReplayBuffer()

    ep1 = await buffer.start_episode("agent_1")
    for i in range(3):
        await buffer.append_step(ep1, [1, i], [0], float(i))
    await buffer.end_episode(ep1, EpisodeStatus.TERMINAL)

    ep2 = await buffer.start_episode("agent_2")
    for i in range(3):
        await buffer.append_step(ep2, [2, i], [0], float(i))
    await buffer.end_episode(ep2, EpisodeStatus.TRUNCATED)

    terminal_count = 0
    truncated_count = 0

    for _ in range(100):
        sample = await buffer.sample_n_step(batch_size=1, n=10)
        episode_id = sample.obs_t[0, 0]

        if sample.terminal[0]:
            assert episode_id == 1
            terminal_count += 1
        if sample.truncated[0]:
            assert episode_id == 2
            truncated_count += 1

    assert terminal_count > 0
    assert truncated_count > 0


@pytest.mark.asyncio
async def test_n_step_reward_sequence():
    """Test that reward sequences are correctly extracted."""
    buffer = ReplayBuffer()

    episode_id = await buffer.start_episode("agent_1")
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, r in enumerate(rewards):
        await buffer.append_step(episode_id, [i], [0], r)
    await buffer.end_episode(episode_id, EpisodeStatus.TERMINAL)

    found_start_sample = False
    for _ in range(50):
        sample = await buffer.sample_n_step(batch_size=1, n=3)
        if sample.obs_t[0, 0] == 0:
            found_start_sample = True
            assert sample.m[0] == 3
            np.testing.assert_array_equal(
                sample.reward_seq[0, : sample.m[0]],
                [1.0, 2.0, 3.0],
            )
            break

    assert found_start_sample


@pytest.mark.asyncio
async def test_concurrent_read_write():
    """Test concurrent reading and writing to the buffer."""
    buffer = ReplayBuffer()

    episode_1 = await buffer.start_episode("agent_1")
    for i in range(20):
        await buffer.append_step(episode_1, [i], [0], float(i))
    await buffer.end_episode(episode_1, EpisodeStatus.TERMINAL)

    async def write_episodes() -> None:
        for i in range(5):
            ep = await buffer.start_episode(f"writer_{i}")
            for j in range(10):
                await buffer.append_step(ep, [100 + i, j], [0], float(j))
                await asyncio.sleep(0.001)
            await buffer.end_episode(ep, EpisodeStatus.TERMINAL)

    async def read_samples() -> None:
        for _ in range(50):
            try:
                await buffer.sample_n_step(batch_size=2, n=3)
                await asyncio.sleep(0.001)
            except ValueError:
                pass

    await asyncio.gather(write_episodes(), read_samples())

    assert buffer.get_num_finished_episodes() >= 1
