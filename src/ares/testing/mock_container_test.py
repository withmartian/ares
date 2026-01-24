"""Tests for mock container implementation."""

import pathlib

import pytest

from ares.containers import containers
from ares.testing import mock_container


@pytest.mark.asyncio
async def test_mock_container_start():
    """Test that start() marks container as started."""
    container = mock_container.MockContainer()
    assert not container.started

    await container.start()
    assert container.started


@pytest.mark.asyncio
async def test_mock_container_stop():
    """Test that stop() marks container as stopped."""
    container = mock_container.MockContainer()
    assert not container.stopped

    await container.stop()
    assert container.stopped


@pytest.mark.asyncio
async def test_mock_container_exec_run_records_commands():
    """Test that exec_run records executed commands."""
    container = mock_container.MockContainer()

    await container.exec_run("ls -la")
    await container.exec_run("pwd")

    assert container.exec_commands == ["ls -la", "pwd"]


@pytest.mark.asyncio
async def test_mock_container_exec_run_default_response():
    """Test that exec_run returns success by default."""
    container = mock_container.MockContainer()

    result = await container.exec_run("echo 'hello'")

    assert result.exit_code == 0
    assert result.output == ""


@pytest.mark.asyncio
async def test_mock_container_exec_run_configured_response():
    """Test that exec_run uses configured responses."""
    container = mock_container.MockContainer()
    container.exec_responses["ls -la"] = containers.ExecResult(
        output="file1.txt\nfile2.txt",
        exit_code=0,
    )

    result = await container.exec_run("ls -la")

    assert result.exit_code == 0
    assert result.output == "file1.txt\nfile2.txt"


@pytest.mark.asyncio
async def test_mock_container_exec_run_custom_handler():
    """Test that exec_run uses custom handler when provided."""
    container = mock_container.MockContainer()

    def handler(command: str) -> containers.ExecResult:
        if "error" in command:
            return containers.ExecResult(output="Error!", exit_code=1)
        return containers.ExecResult(output=f"Executed: {command}", exit_code=0)

    container.exec_handler = handler

    success_result = await container.exec_run("echo 'test'")
    assert success_result.exit_code == 0
    assert "Executed" in success_result.output

    error_result = await container.exec_run("error command")
    assert error_result.exit_code == 1
    assert error_result.output == "Error!"


@pytest.mark.asyncio
async def test_mock_container_upload_files():
    """Test that upload_files records uploaded files."""
    container = mock_container.MockContainer()

    await container.upload_files(
        [pathlib.Path("/local/file1.txt"), pathlib.Path("/local/file2.txt")],
        ["/remote/file1.txt", "/remote/file2.txt"],
    )

    assert len(container.uploaded_files) == 2
    assert container.uploaded_files[0] == (pathlib.Path("/local/file1.txt"), "/remote/file1.txt")
    assert container.uploaded_files[1] == (pathlib.Path("/local/file2.txt"), "/remote/file2.txt")


@pytest.mark.asyncio
async def test_mock_container_download_files():
    """Test that download_files records downloaded files."""
    container = mock_container.MockContainer()

    await container.download_files(
        ["/remote/result.txt", "/remote/output.log"],
        [pathlib.Path("/local/result.txt"), pathlib.Path("/local/output.log")],
    )

    assert len(container.downloaded_files) == 2
    assert container.downloaded_files[0] == ("/remote/result.txt", pathlib.Path("/local/result.txt"))
    assert container.downloaded_files[1] == ("/remote/output.log", pathlib.Path("/local/output.log"))


@pytest.mark.asyncio
async def test_mock_container_upload_file_inherited():
    """Test that upload_file (inherited from Container) works correctly."""
    container = mock_container.MockContainer()

    # upload_file is a concrete method inherited from Container protocol
    # It should internally call upload_files with a single-element list
    await container.upload_file(pathlib.Path("/local/config.json"), "/remote/config.json")

    assert len(container.uploaded_files) == 1
    assert container.uploaded_files[0] == (pathlib.Path("/local/config.json"), "/remote/config.json")


@pytest.mark.asyncio
async def test_mock_container_download_file_inherited():
    """Test that download_file (inherited from Container) works correctly."""
    container = mock_container.MockContainer()

    # download_file is a concrete method inherited from Container protocol
    # It should internally call download_files with a single-element list
    await container.download_file("/remote/result.json", pathlib.Path("/local/result.json"))

    assert len(container.downloaded_files) == 1
    assert container.downloaded_files[0] == ("/remote/result.json", pathlib.Path("/local/result.json"))


def test_mock_container_stop_and_remove():
    """Test synchronous stop_and_remove method."""
    container = mock_container.MockContainer()
    assert not container.stopped

    container.stop_and_remove()
    assert container.stopped


def test_mock_container_factory_from_image():
    """Test factory creates containers from images."""
    container = mock_container.MockContainerFactory.from_image(
        image="python:3.12",
        name="test-container",
    )

    assert isinstance(container, mock_container.MockContainer)
    assert not container.started


def test_mock_container_factory_from_dockerfile():
    """Test factory creates containers from dockerfiles."""
    container = mock_container.MockContainerFactory.from_dockerfile(
        dockerfile_path=pathlib.Path("/path/to/Dockerfile"),
        name="test-container",
    )

    assert isinstance(container, mock_container.MockContainer)
    assert not container.started
