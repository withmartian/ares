"""Tests for the in-sandbox ARES proxy runtime."""

import concurrent.futures
import json
import pathlib

import pytest

from ares.code_agents import ares_proxy
from ares.containers import containers
from ares.testing import mock_container


def test_proxy_binary_build_is_thread_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    architecture = "test-thread-architecture"
    ares_proxy._BINARY_CACHE.pop(architecture, None)
    build_count = 0

    def build_proxy(source_dir: pathlib.Path, output_path: pathlib.Path, goarch: str) -> None:
        nonlocal build_count
        assert source_dir == tmp_path
        assert goarch == architecture
        build_count += 1
        output_path.write_bytes(b"proxy")

    monkeypatch.setattr(ares_proxy, "_source_dir", lambda: tmp_path)
    monkeypatch.setattr(ares_proxy, "_build_sync", build_proxy)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        binaries = list(executor.map(ares_proxy._get_binary_sync, [architecture] * 4))

    assert len(set(binaries)) == 1
    assert build_count == 1
    ares_proxy._BINARY_CACHE.pop(architecture, None)


@pytest.mark.asyncio
async def test_sandbox_proxy_lifecycle(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    binary = tmp_path / "ares-proxy"
    binary.write_bytes(b"proxy")
    response_payload = None

    def exec_handler(command: str) -> containers.ExecResult:
        nonlocal response_payload
        if command == "uname -m":
            return containers.ExecResult(output="x86_64\n", exit_code=0)
        if command.startswith("nohup env PORT="):
            return containers.ExecResult(output="123\n", exit_code=0)
        if command == "curl -fsS http://localhost:8080/poll":
            return containers.ExecResult(
                output=json.dumps([{"id": "1", "endpoint": "/v1/responses", "request": {}}]),
                exit_code=0,
            )
        if "--data-binary" in command:
            response_payload = command
        return containers.ExecResult(output="", exit_code=0)

    async def get_binary(goarch: str) -> pathlib.Path:
        assert goarch == "amd64"
        return binary

    monkeypatch.setattr(ares_proxy, "_get_binary", get_binary)
    container = mock_container.MockContainer(exec_handler=exec_handler)
    proxy = ares_proxy.SandboxProxy(container=container)

    await proxy.install()
    await proxy.start()
    pending = await proxy.poll()
    await proxy.respond(pending[0].id, "stream", content_type="text/event-stream")
    await proxy.stop()

    assert container.uploaded_files == [(binary, ares_proxy.PATH)]
    assert pending[0].endpoint == "/v1/responses"
    assert response_payload is not None
