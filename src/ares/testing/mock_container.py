"""Mock container implementation for testing."""

from collections.abc import Callable
import dataclasses
import pathlib

from ares.containers import containers


@dataclasses.dataclass
class MockContainer(containers.Container):
    """Mock container for testing without real container backends.

    This mock allows tests to verify container interactions without requiring
    Docker or Daytona. It records all method calls and allows configuring
    responses for exec_run commands.

    Attributes:
        started: Whether start() has been called.
        stopped: Whether stop() has been called.
        exec_commands: List of commands executed via exec_run.
        uploaded_files: List of (local_path, remote_path) tuples uploaded.
        downloaded_files: List of (remote_path, local_path) tuples downloaded.
        exec_responses: Dict mapping command strings to ExecResult responses.
        exec_handler: Optional function to dynamically generate responses.
    """

    started: bool = False
    stopped: bool = False
    exec_commands: list[str] = dataclasses.field(default_factory=list)
    uploaded_files: list[tuple[pathlib.Path, str]] = dataclasses.field(default_factory=list)
    downloaded_files: list[tuple[str, pathlib.Path]] = dataclasses.field(default_factory=list)
    exec_responses: dict[str, containers.ExecResult] = dataclasses.field(default_factory=dict)
    exec_handler: Callable[[str], containers.ExecResult] | None = None

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Mark container as started."""
        del env  # Unused in mock
        self.started = True

    async def stop(self) -> None:
        """Mark container as stopped."""
        self.stopped = True

    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> containers.ExecResult:
        """Record command and return configured response.

        Args:
            command: The command to execute.
            workdir: Working directory (recorded but not used).
            env: Environment variables (recorded but not used).
            timeout_s: Timeout in seconds (recorded but not used).

        Returns:
            ExecResult from exec_responses dict, exec_handler, or default success.
        """
        del workdir, env, timeout_s  # Unused in mock
        self.exec_commands.append(command)

        # Try custom handler first
        if self.exec_handler:
            return self.exec_handler(command)

        # Try configured responses
        if command in self.exec_responses:
            return self.exec_responses[command]

        # Default to successful empty response
        return containers.ExecResult(output="", exit_code=0)

    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]) -> None:
        """Record uploaded files."""
        for local_path, remote_path in zip(local_paths, remote_paths, strict=True):
            self.uploaded_files.append((local_path, remote_path))

    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]) -> None:
        """Record downloaded files."""
        for remote_path, local_path in zip(remote_paths, local_paths, strict=True):
            self.downloaded_files.append((remote_path, local_path))

    def stop_and_remove(self) -> None:
        """Synchronous stop for atexit cleanup."""
        self.stopped = True


class MockContainerFactory:
    """Factory for creating mock containers.

    Usage:
        factory = MockContainerFactory()
        container = factory.from_image(image="test:latest")
    """

    def __init__(self):
        """Initialize with empty container list."""
        self.created_containers: list[MockContainer] = []

    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> MockContainer:
        """Create mock container from image."""
        del image, name, resources  # Unused in mock
        container = MockContainer()
        return container

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> MockContainer:
        """Create mock container from dockerfile."""
        del dockerfile_path, name, resources  # Unused in mock
        container = MockContainer()
        return container
