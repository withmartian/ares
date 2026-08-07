"""Build and control ares-proxy inside a sandbox."""

import asyncio
import dataclasses
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import threading
from typing import Any

from ares.containers import containers

PORT = 8080
PATH = "/usr/local/bin/ares-proxy"
LOG_PATH = "/logs/agent/ares-proxy.log"

_BINARY_CACHE: dict[str, pathlib.Path] = {}
_BUILD_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class PendingRequest:
    id: str
    endpoint: str
    request: dict[str, Any]


def _parse_request(value: Any) -> PendingRequest:
    if not isinstance(value, dict):
        raise ValueError(f"Proxy request must be an object, got {type(value).__name__}")

    request_id = value.get("id")
    endpoint = value.get("endpoint")
    request_body = value.get("request")
    if not isinstance(request_id, str):
        raise ValueError("Proxy request field 'id' must be a string")
    if not isinstance(endpoint, str):
        raise ValueError("Proxy request field 'endpoint' must be a string")
    if not isinstance(request_body, dict):
        raise ValueError("Proxy request field 'request' must be an object")
    return PendingRequest(id=request_id, endpoint=endpoint, request=request_body)


def _source_dir() -> pathlib.Path:
    package_root = pathlib.Path(__file__).resolve().parents[1]
    for candidate in [package_root / "ares-proxy", package_root.parents[1] / "ares-proxy"]:
        if (candidate / "go.mod").is_file():
            return candidate
    raise RuntimeError("Could not locate packaged ares-proxy source")


def _build_sync(source_dir: pathlib.Path, output_path: pathlib.Path, goarch: str) -> None:
    env = {**os.environ, "GOOS": "linux", "GOARCH": goarch, "CGO_ENABLED": "0"}
    if go := shutil.which("go"):
        result = subprocess.run(
            [go, "build", "-o", str(output_path), "."],
            cwd=source_dir,
            env=env,
            capture_output=True,
            text=True,
        )
    elif docker := shutil.which("docker"):
        result = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "-e",
                "GOOS=linux",
                "-e",
                f"GOARCH={goarch}",
                "-e",
                "CGO_ENABLED=0",
                "-v",
                f"{source_dir}:/src:ro",
                "-v",
                f"{output_path.parent}:/out",
                "-w",
                "/src",
                "golang:1.24-alpine",
                "go",
                "build",
                "-o",
                f"/out/{output_path.name}",
                ".",
            ],
            capture_output=True,
            text=True,
        )
    else:
        raise RuntimeError("Building ares-proxy requires either Go or Docker on the ARES client")

    if result.returncode != 0:
        raise RuntimeError(f"Failed to build ares-proxy: {result.stderr}")


def _get_binary_sync(goarch: str) -> pathlib.Path:
    with _BUILD_LOCK:
        if goarch not in _BINARY_CACHE:
            output_path = pathlib.Path(tempfile.mkdtemp(prefix="ares-proxy-")) / "ares-proxy"
            _build_sync(_source_dir(), output_path, goarch)
            _BINARY_CACHE[goarch] = output_path
        return _BINARY_CACHE[goarch]


async def _get_binary(goarch: str) -> pathlib.Path:
    return await asyncio.to_thread(_get_binary_sync, goarch)


@dataclasses.dataclass(kw_only=True)
class SandboxProxy:
    container: containers.Container
    timeout_minutes: int = 15

    def __post_init__(self) -> None:
        self._pid: int | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{PORT}"

    async def install(self) -> None:
        result = await self.container.exec_run("uname -m")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to determine sandbox architecture: {result.output}")
        goarch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(result.output.strip())
        if goarch is None:
            raise RuntimeError(f"Unsupported sandbox architecture: {result.output.strip()}")

        await self.container.upload_file(await _get_binary(goarch), PATH)
        result = await self.container.exec_run(f"chmod +x {PATH}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to install ares-proxy: {result.output}")

    async def start(self) -> None:
        result = await self.container.exec_run("mkdir -p /logs/agent")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to create agent log directory: {result.output}")
        result = await self.container.exec_run(
            f"nohup env PORT={PORT} TIMEOUT_MINUTES={self.timeout_minutes} {PATH} > {LOG_PATH} 2>&1 & echo $!"
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to start ares-proxy: {result.output}")
        try:
            self._pid = int(result.output.strip().splitlines()[-1])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"Failed to parse ares-proxy PID: {result.output}") from error

        result = await self.container.exec_run(
            f"for _ in $(seq 1 50); do curl -fsS {self.url}/poll >/dev/null && exit 0; "
            f"sleep 0.1; done; cat {LOG_PATH}; exit 1"
        )
        if result.exit_code != 0:
            raise RuntimeError(f"ares-proxy failed to start: {result.output}")

    async def poll(self) -> list[PendingRequest]:
        result = await self.container.exec_run(f"curl -fsS {self.url}/poll", timeout_s=10)
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to poll ares-proxy: {result.output}")
        try:
            values = json.loads(result.output)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"ares-proxy returned invalid poll JSON: {result.output}") from error
        if not isinstance(values, list):
            raise RuntimeError(f"ares-proxy poll response must be a list, got {type(values).__name__}")
        return [_parse_request(value) for value in values]

    async def respond(self, request_id: str, body: str, *, content_type: str) -> None:
        payload = shlex.quote(
            json.dumps(
                {"id": request_id, "response": body, "content_type": content_type},
                separators=(",", ":"),
            )
        )
        result = await self.container.exec_run(
            f"curl -fsS -X POST {self.url}/respond -H 'Content-Type: application/json' --data-binary {payload}",
            timeout_s=10,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to respond through ares-proxy: {result.output}")

    async def stop(self) -> None:
        if self._pid is not None:
            await self.container.exec_run(f"kill {self._pid} 2>/dev/null || true")
            self._pid = None
