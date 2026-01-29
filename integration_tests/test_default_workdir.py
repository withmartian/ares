"""Integration tests for default working directory behavior across benchmarks."""

import pytest

import ares
from ares.containers import daytona
from ares.environments import code_env


@pytest.mark.parametrize(
    "preset,expected_workdir",
    [
        ("sbv-mswea:0", "/testbed"),
        ("tbench-mswea:0", "/app"),
    ],
)
@pytest.mark.asyncio
async def test_default_workdir(preset: str, expected_workdir: str):
    """Test that environments use the correct default working directory for each benchmark.

    This test verifies that:
    1. Containers are created with the appropriate default_workdir for each benchmark
    2. The working directory is accessible and correct after environment reset

    Note: Uses Daytona containers for consistency and to avoid local Docker image issues.
    """
    # Create environment using preset with Daytona container
    env = ares.make(preset, container_factory=daytona.DaytonaContainer)
    assert isinstance(env, code_env.CodeEnvironment), "Expected CodeEnvironment from ares.make"

    async with env:
        # Reset environment to initialize container
        await env.reset()

        # Container should be initialized after reset
        assert env._container is not None, "Container should be initialized after reset"

        # Access internal container and check working directory
        result = await env._container.exec_run("pwd")

        # Verify the working directory matches expected
        assert result.output.strip() == expected_workdir, (
            f"Expected workdir {expected_workdir} for {preset}, got {result.output.strip()}"
        )

        # Also verify the directory exists and is accessible
        ls_result = await env._container.exec_run("ls -la")
        assert ls_result.exit_code == 0, f"Failed to list directory {expected_workdir}: {ls_result.output}"
