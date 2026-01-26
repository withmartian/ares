"""Tests for environment state snapshotting."""

import pathlib
import tempfile

import pytest

from ares.environments import snapshot
from ares.environments import swebench_env
from ares.testing import mock_container

# Mock task for testing
_MOCK_SWEBENCH_TASK = swebench_env.SwebenchTask(
    repo="test/repo",
    instance_id="test-instance-1",
    base_commit="abc123",
    patch="diff --git a/file.py",
    test_patch="diff --git a/test_file.py",
    problem_statement="Fix the bug",
    hints_text="",
    created_at="2024-01-01",
    version="1.0",
    FAIL_TO_PASS='["test_case_1"]',
    PASS_TO_PASS='["test_case_2"]',
    environment_setup_commit="def456",
)


@pytest.mark.asyncio
async def test_snapshot_dataclass_save_and_load(tmp_path: pathlib.Path):
    """Test EnvironmentSnapshot can be saved to and loaded from disk."""
    snap = snapshot.EnvironmentSnapshot(
        snapshot_id="test-123",
        created_at="2024-01-01T00:00:00",
        snapshot_dir=tmp_path / "snapshots" / "test-123",
        step_count=5,
        step_limit=100,
        requires_reset=False,
        task_type="swebench",
        task_data={"repo": "test/repo", "instance_id": "test-1"},
        container_type="docker",
        container_image="python:3.12",
        container_dockerfile_path=None,
        container_resources={"cpu": 2, "memory": 4096},
        agent_messages=[{"role": "user", "content": "Hello"}],
    )

    snap.snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_file = snap.snapshot_dir / "snapshot.json"

    # Save
    snap.save_to_file(snapshot_file)
    assert snapshot_file.exists()

    # Load
    loaded_snap = snapshot.EnvironmentSnapshot.load_from_file(snapshot_file)
    assert loaded_snap.snapshot_id == snap.snapshot_id
    assert loaded_snap.step_count == snap.step_count
    assert loaded_snap.task_data == snap.task_data
    assert loaded_snap.snapshot_dir == snap.snapshot_dir


def test_swebench_task_serialization():
    """Test SwebenchTask can be serialized and deserialized via SweBenchEnv methods."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    # Serialize via env method (handles JSON string conversion)
    task_data = env._serialize_task(_MOCK_SWEBENCH_TASK)

    # Verify serialization produces a dict with JSON strings
    assert isinstance(task_data, dict)
    assert task_data["instance_id"] == "test-instance-1"

    # Verify deserialization recreates the task
    restored_task = swebench_env.SweBenchEnv._deserialize_task(task_data, "swebench")
    assert restored_task.instance_id == _MOCK_SWEBENCH_TASK.instance_id
    assert restored_task.repo == _MOCK_SWEBENCH_TASK.repo


@pytest.mark.asyncio
async def test_validate_snapshot_allowed_raises_during_active_episode():
    """Test that _validate_snapshot_allowed raises when agent task is running."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
        step_limit=100,
    )

    # Create a mock running task (not done)
    import asyncio

    async def mock_running_task():
        await asyncio.sleep(100)  # Never completes

    env._code_agent_task = asyncio.create_task(mock_running_task())

    # Should raise because task is running
    with pytest.raises(RuntimeError, match="Cannot snapshot during active episode"):
        env._validate_snapshot_allowed()

    # Cleanup
    env._code_agent_task.cancel()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError):
        await env._code_agent_task


@pytest.mark.asyncio
async def test_validate_snapshot_allowed_succeeds_when_task_done():
    """Test that _validate_snapshot_allowed succeeds when agent task is done."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
        step_limit=100,
    )

    # Create a completed task
    import asyncio

    async def mock_completed_task():
        return None

    env._code_agent_task = asyncio.create_task(mock_completed_task())
    await env._code_agent_task  # Wait for completion

    # Should not raise
    env._validate_snapshot_allowed()


@pytest.mark.asyncio
async def test_validate_snapshot_allowed_succeeds_when_no_task():
    """Test that _validate_snapshot_allowed succeeds when no agent task exists."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
        step_limit=100,
    )

    # No agent task set
    env._code_agent_task = None

    # Should not raise
    env._validate_snapshot_allowed()


def test_swebench_env_serialize_task():
    """Test SweBenchEnv._serialize_task produces correct dict."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    serialized = env._serialize_task(_MOCK_SWEBENCH_TASK)

    assert isinstance(serialized, dict)
    assert serialized["instance_id"] == "test-instance-1"
    assert serialized["repo"] == "test/repo"


def test_swebench_env_deserialize_task():
    """Test SweBenchEnv._deserialize_task recreates task from dict."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    # Serialize first (this will convert lists to JSON strings)
    task_data = env._serialize_task(_MOCK_SWEBENCH_TASK)

    # Then deserialize
    restored_task = swebench_env.SweBenchEnv._deserialize_task(task_data, "swebench")

    assert isinstance(restored_task, swebench_env.SwebenchTask)
    assert restored_task.instance_id == "test-instance-1"
    assert restored_task.repo == "test/repo"


def test_swebench_env_get_task_type():
    """Test SweBenchEnv._get_task_type returns 'swebench'."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    assert env._get_task_type() == "swebench"


def test_get_container_type_daytona():
    """Test _get_container_type identifies Daytona containers."""
    from ares.containers import daytona

    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    # Mock a Daytona container
    mock_daytona = type("MockDaytona", (daytona.DaytonaContainer,), {})()

    container_type = env._get_container_type(mock_daytona)
    assert container_type == "daytona"


def test_get_container_type_docker():
    """Test _get_container_type identifies Docker containers."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    # Mock a Docker container (not Daytona)
    mock_docker = mock_container.MockContainer()

    container_type = env._get_container_type(mock_docker)
    assert container_type == "docker"


@pytest.mark.asyncio
async def test_export_state_basic_metadata(tmp_path: pathlib.Path):
    """Test export_state creates snapshot with correct metadata."""

    # Create a mock container with download_dir support
    class MockContainerWithDownload(mock_container.MockContainer):
        def __init__(self):
            super().__init__()
            self.resources = None  # Add resources attribute

        async def download_dir(self, remote_path: str, local_path: pathlib.Path):
            """Mock download_dir that creates an empty tarball."""
            del remote_path  # Unused in mock
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("mock tarball content")

    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
        step_limit=50,
    )

    # Set up minimal state
    container = MockContainerWithDownload()
    await container.start()
    env._container = container
    env._current_task = _MOCK_SWEBENCH_TASK
    env._step_count = 10
    env._requires_reset = False
    env._code_agent_task = None  # No running task

    # Export state
    snap = await env.export_state(tmp_path, snapshot_id="test-export-123")

    # Verify metadata
    assert snap.snapshot_id == "test-export-123"
    assert snap.step_count == 10
    assert snap.step_limit == 50
    assert snap.requires_reset is False
    assert snap.task_type == "swebench"
    assert snap.task_data["instance_id"] == "test-instance-1"
    assert snap.container_type == "docker"

    # Verify files were created
    assert (snap.snapshot_dir / "snapshot.json").exists()
    assert (snap.snapshot_dir / "container_fs.tar.gz").exists()


@pytest.mark.asyncio
async def test_export_state_auto_generates_snapshot_id(tmp_path: pathlib.Path):
    """Test export_state auto-generates UUID when snapshot_id not provided."""

    # Create a mock container with download_dir support
    class MockContainerWithDownload(mock_container.MockContainer):
        def __init__(self):
            super().__init__()
            self.resources = None  # Add resources attribute

        async def download_dir(self, remote_path: str, local_path: pathlib.Path):
            del remote_path  # Unused in mock
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("mock tarball")

    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    # Set up minimal state
    container = MockContainerWithDownload()
    await container.start()
    env._container = container
    env._current_task = _MOCK_SWEBENCH_TASK
    env._code_agent_task = None

    # Export without snapshot_id
    snap = await env.export_state(tmp_path)

    # Should have a UUID-like snapshot_id
    assert snap.snapshot_id is not None
    assert len(snap.snapshot_id) > 0


@pytest.mark.asyncio
async def test_export_state_raises_if_no_container():
    """Test export_state raises if container not available."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    env._container = None
    env._current_task = _MOCK_SWEBENCH_TASK
    env._code_agent_task = None

    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(RuntimeError, match="Container is not available"):
        await env.export_state(pathlib.Path(tmp_dir))


@pytest.mark.asyncio
async def test_export_state_raises_if_no_task():
    """Test export_state raises if current task not available."""
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
    )

    container = mock_container.MockContainer()
    await container.start()
    env._container = container
    env._current_task = None
    env._code_agent_task = None

    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(RuntimeError, match="No current task set"):
        await env.export_state(pathlib.Path(tmp_dir))


@pytest.mark.asyncio
async def test_load_from_state_creates_valid_env(tmp_path: pathlib.Path):
    """Test load_from_state creates a properly initialized environment."""

    # Create a mock container with download_dir and upload_dir support
    class MockContainerWithDirOps(mock_container.MockContainer):
        def __init__(self):
            super().__init__()
            self.resources = None
            self.image = "python:3.12"  # Add image attribute for snapshot

        async def download_dir(self, remote_path: str, local_path: pathlib.Path):
            del remote_path  # Unused in mock
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("mock tarball")

        async def upload_dir(self, local_path: pathlib.Path, remote_path: str):
            """Mock upload_dir for container restoration."""
            del local_path, remote_path  # Unused in mock

    # Create and export state
    env = swebench_env.SweBenchEnv(
        tasks=[_MOCK_SWEBENCH_TASK],
        container_factory=mock_container.MockContainerFactory,
        step_limit=42,
    )

    container = MockContainerWithDirOps()
    await container.start()
    env._container = container
    env._current_task = _MOCK_SWEBENCH_TASK
    env._step_count = 7
    env._requires_reset = False
    env._code_agent_task = None

    snap = await env.export_state(tmp_path, snapshot_id="test-load")

    # Load from snapshot
    class MockContainerFactory:
        @classmethod
        def from_image(cls, *, image: str, name: str | None = None, resources=None):
            del image, name, resources
            return MockContainerWithDirOps()

        @classmethod
        def from_dockerfile(cls, *, dockerfile_path, name: str | None = None, resources=None):
            del dockerfile_path, name, resources
            return MockContainerWithDirOps()

    restored_env = await swebench_env.SweBenchEnv.load_from_state(snap, container_factory=MockContainerFactory)

    # Verify restoration
    assert restored_env._step_count == 7
    assert restored_env._step_limit == 42
    assert restored_env._requires_reset is False
    assert restored_env._current_task.instance_id == _MOCK_SWEBENCH_TASK.instance_id
    assert restored_env._container is not None

    # Cleanup
    await restored_env.close()
