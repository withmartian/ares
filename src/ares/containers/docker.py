"""An interface for interacting with local Docker containers."""

import asyncio
import dataclasses
import functools
import hashlib
import io
import logging
import pathlib
import tarfile
from typing import cast

import docker
import docker.errors
import docker.models.containers

from ares import config
from ares.containers import containers

_LOGGER = logging.getLogger(__name__)


class DockerBuildError(RuntimeError):
    """Raised when Docker image build or container start fails."""

    pass


# Cache: maps (dockerfile_path, dockerfile_hash) -> image_id
# This avoids rebuilding the same Dockerfile repeatedly across environment resets.
_DOCKERFILE_IMAGE_CACHE: dict[tuple[str, str], str] = {}


def _compute_dockerfile_hash(dockerfile_path: pathlib.Path) -> str:
    """Compute a hash of the Dockerfile contents for cache keying.

    Args:
        dockerfile_path: Path to the Dockerfile or its parent directory.

    Returns:
        SHA256 hash of the Dockerfile contents.
    """
    # Handle both file path and directory path
    dockerfile_file = dockerfile_path / "Dockerfile" if dockerfile_path.is_dir() else dockerfile_path
    content = dockerfile_file.read_bytes()
    return hashlib.sha256(content).hexdigest()[:16]  # Use first 16 chars for brevity


def _get_cached_image(dockerfile_path: pathlib.Path) -> str | None:
    """Check if we have a cached image for this Dockerfile.

    Args:
        dockerfile_path: Path to the Dockerfile or its parent directory.

    Returns:
        Cached image ID if found, None otherwise.
    """
    try:
        dockerfile_hash = _compute_dockerfile_hash(dockerfile_path)
        cache_key = (str(dockerfile_path.resolve()), dockerfile_hash)
        return _DOCKERFILE_IMAGE_CACHE.get(cache_key)
    except Exception as e:
        _LOGGER.debug("Error checking image cache: %s", e)
        return None


def _cache_image(dockerfile_path: pathlib.Path, image_id: str) -> None:
    """Cache an image ID for a Dockerfile.

    Args:
        dockerfile_path: Path to the Dockerfile or its parent directory.
        image_id: The built image ID.
    """
    try:
        dockerfile_hash = _compute_dockerfile_hash(dockerfile_path)
        cache_key = (str(dockerfile_path.resolve()), dockerfile_hash)
        _DOCKERFILE_IMAGE_CACHE[cache_key] = image_id
        _LOGGER.debug("Cached image %s for %s (hash: %s)", image_id, dockerfile_path, dockerfile_hash)
    except Exception as e:
        _LOGGER.debug("Error caching image: %s", e)


def _clear_docker_credentials() -> None:
    """Clear stored Docker credentials to allow anonymous pulls.

    This removes the auth configuration from ~/.docker/config.json for Docker Hub,
    allowing Docker to make unauthenticated requests (which don't require email verification).
    """
    import json

    docker_config_path = pathlib.Path.home() / ".docker" / "config.json"
    if not docker_config_path.exists():
        _LOGGER.debug("No Docker config file found at %s", docker_config_path)
        return

    try:
        with open(docker_config_path) as f:
            docker_config = json.load(f)

        # Check if there are any auths to clear
        if docker_config.get("auths"):
            # Clear the auths for Docker Hub
            docker_hub_keys = [k for k in docker_config["auths"] if "docker.io" in k or "index.docker.io" in k]
            if docker_hub_keys:
                for key in docker_hub_keys:
                    del docker_config["auths"][key]
                with open(docker_config_path, "w") as f:
                    json.dump(docker_config, f, indent=2)
                _LOGGER.info("Cleared Docker Hub credentials from %s to enable anonymous pulls", docker_config_path)
    except Exception as e:
        _LOGGER.warning("Failed to clear Docker credentials: %s", e)


def _ensure_docker_login(client: docker.DockerClient) -> None:
    """Attempt Docker registry login if credentials are configured.

    This is a no-op if no credentials are set in config or if docker_skip_auth is True.

    Args:
        client: The Docker client to authenticate.

    Raises:
        DockerBuildError: If authentication fails.
    """
    cfg = config.CONFIG

    # If skip_auth is enabled, clear any stored credentials to ensure anonymous pulls work
    if cfg.docker_skip_auth:
        _LOGGER.debug("DOCKER_SKIP_AUTH is enabled, clearing any stored Docker credentials.")
        _clear_docker_credentials()
        return

    if not cfg.docker_registry_username or not cfg.docker_registry_password:
        _LOGGER.debug("No Docker registry credentials configured, skipping programmatic login.")
        return

    registry = cfg.docker_registry_url or "https://index.docker.io/v1/"
    _LOGGER.debug("Attempting Docker login to %s as %s", registry, cfg.docker_registry_username)

    try:
        client.login(
            username=cfg.docker_registry_username,
            password=cfg.docker_registry_password,
            registry=registry,
        )
        _LOGGER.info("Docker login successful for user: %s", cfg.docker_registry_username)
    except docker.errors.APIError as e:
        raise DockerBuildError(
            f"Docker login failed for user '{cfg.docker_registry_username}'.\n"
            f"Check your DOCKER_REGISTRY_USERNAME and DOCKER_REGISTRY_PASSWORD environment variables.\n"
            f"Original error: {e}"
        ) from e


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
        try:
            return docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Failed to connect to Docker daemon; is docker running?") from e

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the container.

        Raises:
            DockerBuildError: If image build or container start fails.
            ValueError: If neither image nor dockerfile_path is specified.
        """
        # Attempt Docker login if credentials are configured
        _ensure_docker_login(self._client)

        if self.image is None:
            if self.dockerfile_path is None:
                raise ValueError("Must specify one of image or dockerfile_path")

            dockerfile_path = pathlib.Path(self.dockerfile_path)

            # Check cache first to avoid rebuilding the same Dockerfile
            cached_image = _get_cached_image(dockerfile_path)
            if cached_image is not None:
                _LOGGER.info("Using cached image %s for %s", cached_image, dockerfile_path)
                self.image = cached_image
            else:
                _LOGGER.debug("Building image from Dockerfile: %s", dockerfile_path)
                try:
                    image_obj, _ = await asyncio.to_thread(
                        self._client.images.build,
                        path=dockerfile_path.parent.as_posix(),
                        tag=self.name,
                    )
                except docker.errors.BuildError as e:
                    error_msg = str(e)
                    if "email must be verified" in error_msg.lower():
                        raise DockerBuildError(
                            f"Docker Hub requires email verification for your account.\n\n"
                            f"Solutions (choose one):\n"
                            f"  1. RECOMMENDED: Set DOCKER_SKIP_AUTH=true to use anonymous pulls "
                            f"(no account needed)\n"
                            f"  2. Verify your email at https://hub.docker.com/settings/general\n"
                            f"  3. Run 'docker logout' in your terminal to use anonymous pulls\n\n"
                            f"Original error: {error_msg}"
                        ) from e
                    elif "authentication required" in error_msg.lower():
                        raise DockerBuildError(
                            f"Docker authentication failed while building image.\n"
                            f"This typically means Docker Hub requires you to log in or verify your email.\n\n"
                            f"Solutions (choose one):\n"
                            f"  1. Set DOCKER_SKIP_AUTH=true to use anonymous pulls (no account needed)\n"
                            f"  2. Run 'docker login' in your terminal\n"
                            f"  3. Set DOCKER_REGISTRY_USERNAME and DOCKER_REGISTRY_PASSWORD env vars\n\n"
                            f"Original error: {error_msg}"
                        ) from e
                    elif "pull access denied" in error_msg.lower():
                        raise DockerBuildError(
                            f"Docker pull access denied.\n"
                            f"The base image may not exist or requires authentication.\n"
                            f"Check that the base image in your Dockerfile is correct and accessible.\n\n"
                            f"Original error: {error_msg}"
                        ) from e
                    else:
                        raise DockerBuildError(
                            f"Docker build failed: {error_msg}\nCheck the Dockerfile at: {dockerfile_path}"
                        ) from e
                except docker.errors.APIError as e:
                    raise DockerBuildError(
                        f"Docker API error during build: {e}\nIs the Docker daemon running? Try: docker info"
                    ) from e

                image_id = image_obj.id
                assert image_id is not None, f"Image ID is None for container {self.name}"
                self.image = image_id
                _cache_image(dockerfile_path, image_id)
                _LOGGER.debug("Image built successfully: %s", image_id)

        # Start the container
        try:
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
        except docker.errors.ImageNotFound as e:
            raise DockerBuildError(
                f"Docker image not found: {self.image}\nEnsure the image exists or provide a Dockerfile path."
            ) from e
        except docker.errors.APIError as e:
            raise DockerBuildError(f"Docker API error while starting container: {e}") from e

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
