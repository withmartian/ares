"""Tests for EnvironmentCheckpoint and CheckpointableEnvironment."""

import pytest

from ares.environments import base


@pytest.fixture
def simple_timestep() -> base.TimeStep:
    """Create a simple TimeStep for testing."""
    return base.TimeStep(step_type="MID", reward=0.0, discount=1.0, observation="test_obs")


class TestEnvironmentCheckpoint:
    """Tests for EnvironmentCheckpoint."""

    @pytest.mark.asyncio
    async def test_restore_returns_env_and_timestep(self, simple_timestep):
        """Test that restore() returns the env from restore_fn and the stored timestep."""
        mock_env = object()

        async def restore_fn():
            return mock_env

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=5,
            timestep=simple_timestep,
        )

        env, ts = await checkpoint.restore()
        assert env is mock_env
        assert ts is simple_timestep

    @pytest.mark.asyncio
    async def test_restore_creates_independent_envs(self, simple_timestep):
        """Test that multiple restores create independent environments."""
        call_count = 0

        async def restore_fn():
            nonlocal call_count
            call_count += 1
            return {"id": call_count}

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        env_a, _ = await checkpoint.restore()
        env_b, _ = await checkpoint.restore()
        assert env_a != env_b
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_release_calls_release_fn(self, simple_timestep):
        """Test that release() calls the release function."""
        released = False

        async def release_fn():
            nonlocal released
            released = True

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            release_fn=release_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        await checkpoint.release()
        assert released
        assert checkpoint.is_released

    @pytest.mark.asyncio
    async def test_release_is_idempotent(self, simple_timestep):
        """Test that release() can be called multiple times safely."""
        call_count = 0

        async def release_fn():
            nonlocal call_count
            call_count += 1

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            release_fn=release_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        await checkpoint.release()
        await checkpoint.release()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_restore_after_release_raises(self, simple_timestep):
        """Test that restore() raises after release()."""

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        await checkpoint.release()
        with pytest.raises(RuntimeError, match="released"):
            await checkpoint.restore()

    @pytest.mark.asyncio
    async def test_no_release_fn_is_ok(self, simple_timestep):
        """Test that checkpoints without release_fn work fine."""

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        # Release should be a no-op
        await checkpoint.release()
        assert checkpoint.is_released

    def test_step_count_accessible(self, simple_timestep):
        """Test that step_count is publicly accessible."""

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=42,
            timestep=simple_timestep,
        )

        assert checkpoint.step_count == 42

    def test_timestep_accessible(self, simple_timestep):
        """Test that timestep is publicly accessible."""

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        assert checkpoint.timestep is simple_timestep


class TestCheckpointJanitor:
    """Tests for _CheckpointJanitor."""

    def test_register_and_unregister(self, simple_timestep):
        """Test that register/unregister tracks checkpoints."""
        janitor = base._CheckpointJanitor()

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            step_count=0,
            timestep=simple_timestep,
        )

        # Checkpoint auto-registers with the global janitor, but
        # we test the local janitor instance here.
        janitor.register(checkpoint)
        assert id(checkpoint) in janitor._checkpoints

        janitor.unregister(checkpoint)
        assert id(checkpoint) not in janitor._checkpoints

    def test_cleanup_calls_sync_release(self, simple_timestep):
        """Test that janitor cleanup calls release_fn_sync."""
        janitor = base._CheckpointJanitor()
        released = False

        def release_sync():
            nonlocal released
            released = True

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            release_fn_sync=release_sync,
            step_count=0,
            timestep=simple_timestep,
        )

        janitor.register(checkpoint)
        janitor._cleanup()

        assert released
        assert len(janitor._checkpoints) == 0

    def test_cleanup_handles_exceptions(self, simple_timestep):
        """Test that janitor cleanup handles exceptions gracefully."""
        janitor = base._CheckpointJanitor()

        def release_sync():
            raise RuntimeError("cleanup failed")

        async def restore_fn():
            return object()

        checkpoint = base.EnvironmentCheckpoint(
            restore_fn=restore_fn,
            release_fn_sync=release_sync,
            step_count=0,
            timestep=simple_timestep,
        )

        janitor.register(checkpoint)
        # Should not raise
        janitor._cleanup()
        assert len(janitor._checkpoints) == 0
