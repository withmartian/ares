"""PubSub container management for distributed agent communication."""

import asyncio
import dataclasses
import functools
import logging
import pathlib

import docker

from ares.containers import containers

_LOGGER = logging.getLogger(__name__)

# Path to the Dockerfile for the pubsub proxy
_PROXY_DOCKERFILE_DIR = pathlib.Path(__file__).parent


@dataclasses.dataclass(kw_only=True)
class PubSubContainer(containers.Container):
    """Container running the FastAPI-based PubSub proxy.

    This container runs a simple HTTP server that bridges agent LLM requests
    to the local RL training loop. Agents make HTTP requests to this container,
    which queues them for the local machine to consume via polling.

    The container runs a single FastAPI process with in-memory state.
    """

    name: str | None = None
    port: int = 8000
    resources: containers.Resources | None = None

    @functools.cached_property
    def _client(self) -> docker.DockerClient:
        return docker.from_env()

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the PubSub proxy container.

        This builds the Docker image from the Dockerfile in this module
        and starts a container running the FastAPI proxy.

        Args:
            env: Optional environment variables (unused for proxy)
        """
        _LOGGER.info("Building PubSub proxy Docker image...")

        # Build the image
        image_obj, build_logs = await asyncio.to_thread(
            self._client.images.build,
            path=str(_PROXY_DOCKERFILE_DIR),
            tag=f"{self.name}:latest" if self.name else "ares-pubsub-proxy:latest",
            rm=True,  # Remove intermediate containers
        )

        # Log build output for debugging
        for log in build_logs:
            if "stream" in log:
                _LOGGER.debug("Build: %s", log["stream"].strip())

        _LOGGER.info("PubSub proxy image built: %s", image_obj.id[:12])

        # Start the container
        # Note: Container always listens on port 8000 internally (from Dockerfile)
        # We map that to the requested host port
        self._container = await asyncio.to_thread(
            self._client.containers.run,
            image=image_obj.id,
            name=self.name,
            detach=True,
            ports={"8000/tcp": self.port},  # Map container's 8000 to host port
            environment=env,
        )

        _LOGGER.info("PubSub proxy container started: %s (port %d)", self.name, self.port)

        # Wait for the service to be ready
        await self._wait_for_ready()

    async def _wait_for_ready(self, timeout: float = 30.0) -> None:
        """Wait for the FastAPI app to be ready to accept requests.

        Args:
            timeout: Maximum time to wait (seconds)

        Raises:
            RuntimeError: If service doesn't become ready within timeout
        """
        import time

        import httpx

        start = time.time()
        url = f"http://localhost:{self.port}/health"

        _LOGGER.debug("Waiting for PubSub proxy to be ready at %s...", url)

        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        _LOGGER.info("PubSub proxy is ready")
                        return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as e:
                _LOGGER.debug("Health check failed (retrying): %s", e)
                await asyncio.sleep(0.5)
                continue
            except Exception as e:
                _LOGGER.warning("Unexpected error during health check: %s", e)
                await asyncio.sleep(0.5)
                continue

        # If we get here, we timed out. Get container logs for debugging
        try:
            logs = self._container.logs(tail=50).decode("utf-8", errors="replace")
            _LOGGER.error("Container logs:\n%s", logs)
        except Exception as e:
            _LOGGER.error("Failed to get container logs: %s", e)

        raise RuntimeError(f"PubSub proxy did not become ready within {timeout}s")

    async def stop(self) -> None:
        """Stop the PubSub proxy container."""
        if not hasattr(self, "_container"):
            _LOGGER.info("PubSub container not started, stop is a no-op.")
            return

        _LOGGER.info("Stopping PubSub proxy container %s", self.name)
        await asyncio.to_thread(self._container.stop)
        await asyncio.to_thread(self._container.remove, force=True)

    def stop_and_remove(self) -> None:
        """Stop and remove the container (synchronous for atexit)."""
        if not hasattr(self, "_container"):
            _LOGGER.info("PubSub container not started, stop_and_remove is a no-op.")
            return

        _LOGGER.info("Stopping and removing PubSub proxy container %s", self.name)
        self._container.stop()
        self._container.remove(force=True)

    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> containers.ExecResult:
        """Execute a command in the container.

        Note: This is not typically used for the PubSub proxy, but is
        implemented for compatibility with the Container protocol.
        """
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

    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]) -> None:
        """Upload files to the container (not typically used for proxy)."""
        raise NotImplementedError("File upload not implemented for PubSubContainer")

    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]) -> None:
        """Download files from the container (not typically used for proxy)."""
        raise NotImplementedError("File download not implemented for PubSubContainer")

    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "PubSubContainer":
        """Not supported - PubSubContainer always builds from Dockerfile."""
        raise NotImplementedError("PubSubContainer must be built from Dockerfile")

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "PubSubContainer":
        """Not supported - PubSubContainer uses built-in Dockerfile."""
        raise NotImplementedError("PubSubContainer uses built-in Dockerfile")

    def get_base_url(self) -> str:
        """Get the base URL for agents to connect to this proxy.

        Returns:
            Base URL in format http://host:port
        """
        # For local Docker, containers can reach host via special hostname
        # For Daytona, this would be the container name or IP
        return f"http://localhost:{self.port}"
