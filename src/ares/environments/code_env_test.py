"""Tests for the code environment."""

import pathlib
import types
from typing import Any, cast

import pytest

from ares.containers import containers
from ares.environments import code_env
from ares.testing import mock_container


@pytest.mark.asyncio
async def test_persist_artifacts_downloads_harbor_log_tree(tmp_path: pathlib.Path) -> None:
    attempts = 0

    def exec_handler(command: str) -> containers.ExecResult:
        nonlocal attempts
        if command == "find /logs -type f":
            attempts += 1
            if attempts < 3:
                raise TimeoutError("transient download failure")
            return containers.ExecResult(
                output=("/logs/agent/opencode.jsonl\n/logs/agent/mediated.jsonl\n/logs/verifier/test-stdout.txt\n"),
                exit_code=0,
            )
        return containers.ExecResult(output="", exit_code=0)

    container = mock_container.MockContainer(exec_handler=exec_handler)
    env = code_env.CodeEnvironment(tasks=[], artifact_root=tmp_path)
    env._container = container
    env._episode_artifact_dir = tmp_path / "episode"
    try:
        assert env.artifact_dir is None
        await env._persist_artifacts()

        assert [remote for remote, _ in container.downloaded_files] == [
            "/logs/agent/opencode.jsonl",
            "/logs/agent/mediated.jsonl",
            "/logs/verifier/test-stdout.txt",
        ]
        assert env._artifacts_persisted is True
        assert env.artifact_dir == tmp_path / "episode"
        assert attempts == 3
        assert (tmp_path / "episode" / "agent").is_dir()
        assert (tmp_path / "episode" / "verifier").is_dir()
        assert all(isinstance(local, pathlib.Path) for _, local in container.downloaded_files)
    finally:
        code_env._ENVIRONMENT_JANITOR.unregister_for_cleanup(env)


@pytest.mark.asyncio
async def test_stop_container_propagates_artifact_failure_and_still_stops(tmp_path: pathlib.Path) -> None:
    def exec_handler(command: str) -> containers.ExecResult:
        if command == "find /logs -type f":
            raise ValueError("permanent artifact failure")
        return containers.ExecResult(output="", exit_code=0)

    container = mock_container.MockContainer(exec_handler=exec_handler)
    env = code_env.CodeEnvironment(tasks=[], artifact_root=tmp_path)
    env._container = container
    env._episode_artifact_dir = tmp_path / "episode"
    try:
        with pytest.raises(ValueError, match="permanent artifact failure"):
            await env._stop_container()

        assert container.stopped is True
        assert env._container is None
        assert env.artifact_dir is None
    finally:
        code_env._ENVIRONMENT_JANITOR.unregister_for_cleanup(env)


@pytest.mark.asyncio
async def test_start_container_cleans_up_failed_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    def exec_handler(command: str) -> containers.ExecResult:
        if command == "mkdir -p /logs/agent /logs/verifier /logs/artifacts":
            return containers.ExecResult(output="permission denied", exit_code=1)
        return containers.ExecResult(output="", exit_code=0)

    container = mock_container.MockContainer(exec_handler=exec_handler)

    async def create_container(**kwargs: Any) -> containers.Container:
        del kwargs
        return container

    monkeypatch.setattr(code_env.base, "create_container", create_container)
    env = code_env.CodeEnvironment(tasks=[])
    env._current_task = cast(
        Any,
        types.SimpleNamespace(
            name="test-task",
            config=types.SimpleNamespace(
                environment=types.SimpleNamespace(
                    docker_image="test:latest",
                    cpus=1,
                    memory_mb=1024,
                    storage_mb=1024,
                )
            ),
            paths=types.SimpleNamespace(environment_dir=tmp_path),
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="Failed to create container log directories"):
            await env._start_container()

        assert container.started is True
        assert container.stopped is True
        assert env._container is None
    finally:
        code_env._ENVIRONMENT_JANITOR.unregister_for_cleanup(env)


@pytest.mark.asyncio
async def test_compute_reward_persists_verifier_output(tmp_path: pathlib.Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_path = tests_dir / "test.sh"
    test_path.write_text("#!/bin/bash\nexit 1\n")

    def exec_handler(command: str) -> containers.ExecResult:
        if command == "cat /logs/verifier/reward.txt":
            return containers.ExecResult(output="0\n", exit_code=0)
        return containers.ExecResult(output="", exit_code=0)

    container = mock_container.MockContainer(exec_handler=exec_handler)
    env = code_env.CodeEnvironment(tasks=[])
    env._container = container
    env._current_task = cast(
        Any,
        types.SimpleNamespace(
            name="test-task",
            paths=types.SimpleNamespace(tests_dir=tests_dir, test_path=test_path),
        ),
    )
    try:
        reward = await env._compute_reward()

        assert reward == 0.0
        assert "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1" in container.exec_commands
    finally:
        code_env._ENVIRONMENT_JANITOR.unregister_for_cleanup(env)
