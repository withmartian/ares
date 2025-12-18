"""An interface for interacting with containers."""

import abc
import dataclasses
import pathlib
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class ExecResult:
    # TODO: Maybe stdout/stderr?
    output: str
    exit_code: int


@dataclasses.dataclass(frozen=True)
class Resources:
    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None
    gpu: int | None = None


class Container(Protocol):
    @abc.abstractmethod
    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the container.

        Args:
            env: The environment variables to set in the container.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop the container."""

    @abc.abstractmethod
    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Execute a command in the container.

        Args:
            command: The command to execute.
            workdir: The working directory to execute the command in.
                If None, uses the default working directory.
            env: The environment variables to set in the container, if any.
            timeout_s: The timeout in seconds for the command, if any.

        Returns:
            ExecResult: The result of the command execution.

        Raises:
            TimeoutError: If a timeout is specified and the command timed out.
        """

    @abc.abstractmethod
    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]) -> None:
        """Upload a file to the container.

        Args:
            local_path: The path to the local file.
            remote_path: The path to the remote file.
        """

    @abc.abstractmethod
    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]) -> None:
        """Download files from the container.

        Args:
            remote_paths: The path to the remote files.
            local_paths: The path to the local files.
        """

    async def upload_file(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Upload a single file to the container."""
        await self.upload_files([local_path], [remote_path])

    async def upload_dir(self, local_path: pathlib.Path, remote_path: str) -> None:
        """Upload a full directory to the container."""
        local_path_uploads = []
        remote_path_uploads = []

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                destination_path = str(remote_path / relative_path)

                local_path_uploads.append(str(file_path))
                remote_path_uploads.append(destination_path)

        await self.upload_files(local_path_uploads, remote_path_uploads)

    async def download_file(self, remote_path: str, local_path: pathlib.Path) -> None:
        """Download a single file from the container."""
        await self.download_files([remote_path], [local_path])

    async def download_dir(self, remote_path: str, local_path: pathlib.Path) -> None:
        """Download a full directory from the container."""
        remote_path_downloads = []
        local_path_downloads = []

        local_path.mkdir(parents=True, exist_ok=True)

        all_remote_file_paths = (await self.exec_run(f"find {remote_path} -type f")).output.splitlines()

        for file_path in all_remote_file_paths:
            path_obj = pathlib.Path(file_path)
            relative_path = path_obj.relative_to(remote_path)
            local_file_path = local_path / relative_path

            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            remote_path_downloads.append(file_path)
            local_path_downloads.append(str(local_file_path))

        if remote_path_downloads:
            await self.download_files(remote_path_downloads, local_path_downloads)

    def stop_and_remove(self) -> None:
        """Stop and remove the container.

        This is the one synchronous method, since it's intended to be used with atexit
        to clean up resources no matter how the program exits.
        """


class ContainerFactory(Protocol):
    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: Resources | None = None,
    ) -> Container: ...

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: Resources | None = None,
    ) -> Container: ...
