"""Tests for Docker container snapshotting."""

import unittest.mock

import pytest

from ares.containers import docker


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    with unittest.mock.patch.object(docker, "_make_docker_client") as mock_fn:
        client = unittest.mock.MagicMock()
        mock_fn.return_value = client
        yield client


@pytest.mark.asyncio
async def test_snapshot_creates_image(mock_docker_client):  # noqa: ARG001
    """Test that snapshot() commits the container and returns an image ID."""
    container = docker.DockerContainer(image="test:latest")

    # Set up mock container
    mock_inner = unittest.mock.MagicMock()
    container._container = mock_inner

    mock_image = unittest.mock.MagicMock()
    mock_image.id = "sha256:abc123"
    mock_inner.commit.return_value = mock_image

    snapshot_id = await container.snapshot()

    assert snapshot_id == "sha256:abc123"
    mock_inner.commit.assert_called_once()
    call_kwargs = mock_inner.commit.call_args
    assert call_kwargs[1]["repository"] == "ares-go-explore"
    assert call_kwargs[1]["conf"]["Labels"]["ares-go-explore"] == "true"


@pytest.mark.asyncio
async def test_snapshot_raises_if_not_started():
    """Test that snapshot() raises if container isn't started."""
    container = docker.DockerContainer(image="test:latest")
    with pytest.raises(RuntimeError, match="not started"):
        await container.snapshot()


def test_from_snapshot_creates_container():
    """Test that from_snapshot() creates a DockerContainer with the snapshot as image."""
    container = docker.DockerContainer.from_snapshot(
        "sha256:abc123",
        name="restored",
        default_workdir="/workspace",
    )

    assert isinstance(container, docker.DockerContainer)
    assert container.image == "sha256:abc123"
    assert container.name == "restored"
    assert container.default_workdir == "/workspace"


@pytest.mark.asyncio
async def test_delete_snapshot(mock_docker_client):
    """Test that delete_snapshot() removes the Docker image."""
    container = docker.DockerContainer(image="test:latest")
    await container.delete_snapshot("sha256:abc123")
    mock_docker_client.images.remove.assert_called_once_with("sha256:abc123")


@pytest.mark.asyncio
async def test_delete_snapshot_ignores_not_found(mock_docker_client):
    """Test that delete_snapshot() handles already-deleted images."""
    import docker as docker_lib

    mock_docker_client.images.remove.side_effect = docker_lib.errors.ImageNotFound("not found")
    container = docker.DockerContainer(image="test:latest")
    # Should not raise
    await container.delete_snapshot("sha256:abc123")


def test_delete_snapshot_sync(mock_docker_client):
    """Test synchronous snapshot deletion for atexit cleanup."""
    container = docker.DockerContainer(image="test:latest")
    container.delete_snapshot_sync("sha256:abc123")
    mock_docker_client.images.remove.assert_called_once_with("sha256:abc123")
