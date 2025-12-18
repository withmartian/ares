"""An interface for interacting with daytona containers."""

import asyncio
import contextlib
import dataclasses
import functools
import logging
import pathlib

import daytona
import daytona.common.errors
import daytona.common.process
import tenacity

from ares import config
from ares.containers import containers

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache
def _get_daytona_client() -> daytona.AsyncDaytona:
    return daytona.AsyncDaytona()


@functools.lru_cache
def _get_daytona_client_sync() -> daytona.Daytona:
    return daytona.Daytona()


@tenacity.retry(
    retry=tenacity.retry_if_exception_type(daytona.common.errors.DaytonaError),
    stop=tenacity.stop_after_attempt(10),
    wait=tenacity.wait_exponential(multiplier=1.2, min=1, max=60) + tenacity.wait_random(min=0, max=1),
    before_sleep=tenacity.before_sleep_log(_LOGGER, logging.INFO),
)
async def _exec_with_retry(
    sbx: daytona.AsyncSandbox,
    command: str,
    *,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> daytona.common.process.ExecuteResponse:
    try:
        return await asyncio.wait_for(
            sbx.process.exec(command, cwd=workdir, env=env),
            timeout=timeout_s,
        )
    except daytona.common.errors.DaytonaError as e:
        _LOGGER.warning("Error executing command in sandbox %s in state [%s]: %s", sbx.id, sbx.state, e)
        raise


@dataclasses.dataclass(kw_only=True)
class DaytonaContainer(containers.Container):
    image: str | None = None
    dockerfile_path: pathlib.Path | str | None = None
    name: str | None = None
    resources: containers.Resources | None = None

    def __post_init__(self):
        self._sbx: daytona.AsyncSandbox | None = None

        self._daytona_resources: daytona.Resources | None = None
        if self.resources is not None:
            self._daytona_resources = daytona.Resources(
                cpu=self.resources.cpu,
                memory=self.resources.memory,
                disk=self.resources.disk,
                gpu=self.resources.gpu,
            )

    async def start(self, env: dict[str, str] | None = None) -> None:
        """Start the container."""
        # TODO: Look into using snapshots here - I believe it's way faster than images
        if self.image is not None:
            _LOGGER.debug("Using prebuilt image: %s", self.image)
            img = daytona.Image.base(self.image)
        elif self.dockerfile_path is not None:
            _LOGGER.debug("Building environment from Dockerfile")
            img = daytona.Image.from_dockerfile(self.dockerfile_path)
        else:
            raise ValueError("Must specify one of image or dockerfile_path")

        params = daytona.CreateSandboxFromImageParams(
            name=self.name,
            image=img,
            env_vars=env,
            auto_stop_interval=config.CONFIG.daytona_auto_stop_interval,
            auto_delete_interval=0 if config.CONFIG.daytona_delete_on_stop else None,
            labels={"user": config.CONFIG.user},
            resources=self._daytona_resources,
        )
        self._sbx = await _get_daytona_client().create(params=params)

    async def stop(self) -> None:
        """Stop the container."""
        if self._sbx is None:
            _LOGGER.info("Sandbox not started, stop is a no-op.")
            return

        # For now, we delete the sandbox. This also stops it.
        # This is an easy way to prevent users going over quota with Daytona.
        # Note that if the sandbox is already deleted, this will do nothing.
        with contextlib.suppress(daytona.common.errors.DaytonaNotFoundError):
            await self._sbx.delete()

        self._sbx = None

    async def exec_run(
        self,
        command: str,
        *,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> containers.ExecResult:
        if self._sbx is None:
            raise RuntimeError("Sandbox not started, exec_run is not possible.")

        _LOGGER.debug("[%d] Executing command: %s", id(self), command)
        # Note: we only retry on DaytonaErrors.
        # If we get a Timeout, it gets re-raised.
        result = await _exec_with_retry(
            self._sbx,
            command,
            workdir=workdir,
            env=env,
            timeout_s=timeout_s,
        )

        exit_code = result.exit_code
        int_exit_code = int(exit_code)

        if float(int_exit_code) != exit_code:
            raise ValueError(f"Exit code is not an integer: {exit_code}")

        return containers.ExecResult(output=result.result, exit_code=int_exit_code)

    def stop_and_remove(self) -> None:
        """Stop and remove the container."""
        if self._sbx is None:
            _LOGGER.info("Sandbox not started, stop_and_remove is a no-op.")
            return

        client = _get_daytona_client_sync()

        _LOGGER.info("Stopping and removing sandbox %s", self._sbx.id)
        try:
            sync_sbx = client.get(self._sbx.id)
        except daytona.common.errors.DaytonaNotFoundError:
            # The container doesn't exist.
            _LOGGER.debug("Sandbox %s not found, stop_and_remove is a no-op.", self._sbx.id)
            return

        # It is sufficient to delete the sandbox to stop it.
        try:
            sync_sbx.delete(timeout=10)
        except daytona.DaytonaError as e:
            _LOGGER.error("Error deleting sandbox %s: %s", sync_sbx.id, e)
        _LOGGER.info("Sandbox %s deleted", sync_sbx.id)

    async def upload_files(self, local_paths: list[pathlib.Path], remote_paths: list[str]) -> None:
        """Upload files to the container."""
        if self._sbx is None:
            raise RuntimeError("Sandbox not started, upload_files is not possible.")

        file_uploads = [
            daytona.FileUpload(
                source=str(local_path),
                destination=remote_path,
            )
            for local_path, remote_path in zip(local_paths, remote_paths, strict=True)
        ]

        if file_uploads:
            await self._sbx.fs.upload_files(files=file_uploads)

    async def download_files(self, remote_paths: list[str], local_paths: list[pathlib.Path]) -> None:
        """Download files from the container."""
        if self._sbx is None:
            raise RuntimeError("Sandbox not started, download_files is not possible.")

        file_downloads = [
            daytona.FileDownloadRequest(
                source=remote_path,
                destination=str(local_path),
            )
            for remote_path, local_path in zip(remote_paths, local_paths, strict=True)
        ]

        if file_downloads:
            await self._sbx.fs.download_files(files=file_downloads)

    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "DaytonaContainer":
        return DaytonaContainer(image=image, name=name, resources=resources)

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: containers.Resources | None = None,
    ) -> "DaytonaContainer":
        return DaytonaContainer(dockerfile_path=dockerfile_path, name=name, resources=resources)
