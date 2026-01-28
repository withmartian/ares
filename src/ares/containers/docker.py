"""An interface for interacting with local Docker containers."""

import asyncio
import dataclasses
import functools
import io
import pathlib
import tarfile
from typing import cast

import docker
import docker.errors
import docker.models.containers
import docker.models.images

from ares.containers import containers


def _make_docker_client() -> docker.DockerClient:
    try:
        return docker.from_env()
    except docker.errors.DockerException as e:
        raise RuntimeError("Failed to connect to Docker daemon; is docker running?") from e


@dataclasses.dataclass(kw_only=True)
class DockerContainer(containers.Container):
    image: str | None = None
    dockerfile_path: pathlib.Path | str | None = None
    name: str | None = None
    # TODO: Figure out a way to set resources for docker containers.
    resources: containers.Resources | None = None
    _container: docker.models.containers.Container | None = dataclasses.field(default=None, init=False)

    @functools.cached_property
    def _client(self) -> docker.DockerClient:
        return _make_docker_client()

    @functools.cached_property
    def _client_with_no_auth(self) -> docker.DockerClient:
        client = _make_docker_client()
        client.images.client.api._auth_configs["credHelpers"] = {}  # type: ignore
        return client

    def _build_image(self, path: str, tag: str | None) -> docker.models.images.Image:
        try:
            return self._client.images.build(path=path, tag=tag)[0]
        except docker.errors.DockerException as e:
            if "StoreError" in str(e):
                # Try again after removing auth sources, since they might be broken.
                # Fix for https://github.com/docker/docker-py/issues/3379 - if auth sources fail,
                # we the build() call will fail even if the image is publicly accessible.
                return self._client_with_no_auth.images.build(path=path, tag=tag)[0]
            else:
                raise

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the container."""
        if self.image is None:
            if self.dockerfile_path is None:
                raise ValueError("Must specify one of image or dockerfile_path")

            dockerfile_path = pathlib.Path(self.dockerfile_path)
            # TODO: Some kind of cache for dockerfile directory to avoid
            #       rebuilding same image over and over again?
            image_obj = await asyncio.to_thread(
                self._build_image,
                path=dockerfile_path.parent.as_posix(),
                tag=self.name,
            )
            self.image = image_obj.id
            assert self.image is not None, f"Image ID is None for container {self.name}"

        # TODO: Work out why this cast is necessary.
        self._container = cast(
            docker.models.containers.Container,
            await asyncio.to_thread(
                self._client.containers.run,
                image=self.image,
                name=self.name,
                # Ensure the container stays running.
                command="tail -f /dev/null",
                detach=True,
                environment=env,
            ),
        )

    async def stop(self) -> None:
        """Stop the container."""
        if self._container is not None:
            await asyncio.to_thread(self._container.stop)
            await asyncio.to_thread(self._container.remove, force=True)
            self._container = None

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
                # TODO: Consolidate this implementation into some container base class/helper fn
                # NOTE: Many code agents expect things like `.bashrc` to be loaded, so we use `bash -lc` here
                ["bash", "-lc", command],
                workdir=workdir,
                environment=env,
            ),
            timeout=timeout_s,
        )
        result_str = result.output.decode("utf-8", errors="replace")
        return containers.ExecResult(output=result_str, exit_code=result.exit_code)

    def stop_and_remove(self) -> None:
        """Stop and remove the container."""
        if self._container is not None:
            self._container.stop()
            self._container.remove(force=True)
            self._container = None

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

            # Create dirs if they don't exist (since otherwise Docker complains)
            await asyncio.to_thread(
                self._container.exec_run,
                cmd=f"mkdir -p {remote_dir}",
            )

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
