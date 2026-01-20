"""An interface for interacting with local Docker containers."""

import asyncio
import dataclasses
import functools
import io
import pathlib
import tarfile

import docker

from ares.containers import containers


@dataclasses.dataclass(kw_only=True)
class DockerContainer(containers.Container):
    image: str | None = None
    dockerfile_path: pathlib.Path | str | None = None
    name: str | None = None
    # TODO: Figure out a way to set resources for docker containers.
    resources: containers.Resources | None = None

    @functools.cached_property
    def _client(self) -> docker.DockerClient:
        return docker.from_env()

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the container."""
        if self.image is None:
            assert self.dockerfile_path is not None, "Must specify one of image or dockerfile_path"
            # TODO: Some kind of cache for dockerfile directory to avoid
            #       rebuilding same image over and over again?
            image_obj, _ = await asyncio.to_thread(
                self._client.images.build,
                path=self.dockerfile_path.parent,
                tag=self.name,
            )
            self.image = image_obj.id

        self._container = await asyncio.to_thread(
            self._client.containers.run,
            image=self.image,
            name=self.name,
            # Ensure the container stays running.
            command="tail -f /dev/null",
            detach=True,
            environment=env,
        )

    async def stop(self) -> None:
        """Stop the container."""
        await asyncio.to_thread(self._container.stop)
        await asyncio.to_thread(self._container.remove, force=True)

    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> containers.ExecResult:
        # We have to run the command with `sh -c` to handle multiple commands in a single string.
        result = await asyncio.wait_for(
            asyncio.to_thread(
                self._container.exec_run,
                ["sh", "-c", command],
                workdir=workdir,
                environment=env,
            ),
            timeout=timeout_s,
        )
        result_str = result.output.decode("utf-8", errors="replace")
        return containers.ExecResult(output=result_str, exit_code=result.exit_code)

    def stop_and_remove(self) -> None:
        """Stop and remove the container."""
        self._container.stop()
        self._container.remove(force=True)

    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]) -> None:
        """Upload files to the container."""
        if len(local_paths) != len(remote_paths):
            raise ValueError("local_paths and remote_paths must have the same length")

        for local_path, remote_path in zip(local_paths, remote_paths, strict=True):
            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(str(local_path), arcname=pathlib.Path(remote_path).name)
            tar_stream.seek(0)

            # Determine the destination directory
            remote_dir = str(pathlib.Path(remote_path).parent)

            # Upload the tar archive to the container
            await asyncio.to_thread(
                self._container.put_archive,
                path=remote_dir,
                data=tar_stream.getvalue(),
            )

    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]) -> None:
        """Download files from the container."""
        if len(remote_paths) != len(local_paths):
            raise ValueError("remote_paths and local_paths must have the same length")

        for remote_path, local_path in zip(remote_paths, local_paths, strict=True):
            # Ensure the local directory exists
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Get the tar archive from the container
            tar_stream, _ = await asyncio.to_thread(
                self._container.get_archive,
                path=remote_path,
            )

            # Extract the file from the tar archive
            tar_bytes = b"".join(tar_stream)
            with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar_file:
                # Extract the file
                members = tar_file.getmembers()
                if members:
                    # Get the first member (the file we want)
                    member = members[0]
                    file_data = tar_file.extractfile(member)
                    if file_data:
                        with open(local_path, "wb") as f:
                            f.write(file_data.read())

    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "DockerContainer":
        return DockerContainer(image=image, name=name, resources=resources)

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "DockerContainer":
        return DockerContainer(dockerfile_path=dockerfile_path, name=name, resources=resources)
