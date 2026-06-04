"""Tests for Daytona container implementation."""

import pathlib
import types
from typing import cast

import daytona
import pytest

from ares.containers import daytona as ares_daytona
from ares.environments import base


def test_daytona_container_methods_preserve_explicit_config() -> None:
    """Test that Daytona container methods pass explicit config to containers."""
    daytona_config = daytona.DaytonaConfig(api_key="test-key", api_url="https://daytona.example")

    image_container = ares_daytona.DaytonaContainer.from_image(
        image="python:3.12",
        daytona_config=daytona_config,
    )
    dockerfile_container = ares_daytona.DaytonaContainer.from_dockerfile(
        dockerfile_path=pathlib.Path("Dockerfile"),
        daytona_config=daytona_config,
    )

    assert image_container.daytona_config is daytona_config
    assert dockerfile_container.daytona_config is daytona_config


def test_daytona_container_factory_preserves_explicit_config() -> None:
    """Test that DaytonaContainerFactory passes explicit config to containers."""
    daytona_config = daytona.DaytonaConfig(api_key="test-key", api_url="https://daytona.example")
    factory = ares_daytona.DaytonaContainerFactory(daytona_config=daytona_config)

    image_container = factory.from_image(
        image="python:3.12",
        default_workdir="/workspace",
    )
    dockerfile_container = factory.from_dockerfile(
        dockerfile_path=pathlib.Path("Dockerfile"),
        default_workdir="/workspace",
    )

    assert image_container.daytona_config is daytona_config
    assert image_container.default_workdir == "/workspace"
    assert dockerfile_container.daytona_config is daytona_config
    assert dockerfile_container.default_workdir == "/workspace"


@pytest.mark.asyncio
async def test_start_uses_explicit_daytona_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that start() creates the async Daytona client with explicit config."""
    daytona_config = daytona.DaytonaConfig(api_key="test-key", api_url="https://daytona.example")
    created_configs: list[daytona.DaytonaConfig | None] = []

    class FakeAsyncDaytona:
        def __init__(self, config: daytona.DaytonaConfig | None = None) -> None:
            self.config = config

        async def create(self, *, params: daytona.CreateSandboxFromImageParams) -> types.SimpleNamespace:
            del params
            created_configs.append(self.config)
            return types.SimpleNamespace(id="sandbox-id", state="started")

    monkeypatch.setattr(ares_daytona.daytona, "AsyncDaytona", FakeAsyncDaytona)

    container = ares_daytona.DaytonaContainer.from_image(
        image="python:3.12",
        daytona_config=daytona_config,
    )
    await container.start()

    assert created_configs == [daytona_config]
    assert container._sbx is not None
    assert container._sbx.id == "sandbox-id"


def test_stop_and_remove_uses_explicit_daytona_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that stop_and_remove() creates the sync Daytona client with explicit config."""
    daytona_config = daytona.DaytonaConfig(api_key="test-key", api_url="https://daytona.example")
    created_configs: list[daytona.DaytonaConfig | None] = []
    get_ids: list[str] = []
    deleted_timeouts: list[int] = []

    class FakeSandbox:
        id = "sandbox-id"

        def delete(self, timeout: int) -> None:
            deleted_timeouts.append(timeout)

    class FakeDaytona:
        def __init__(self, config: daytona.DaytonaConfig | None = None) -> None:
            created_configs.append(config)

        def get(self, sandbox_id: str) -> FakeSandbox:
            get_ids.append(sandbox_id)
            return FakeSandbox()

    monkeypatch.setattr(ares_daytona.daytona, "Daytona", FakeDaytona)

    container = ares_daytona.DaytonaContainer.from_image(
        image="python:3.12",
        daytona_config=daytona_config,
    )
    container._sbx = cast(daytona.AsyncSandbox, types.SimpleNamespace(id="sandbox-id"))

    container.stop_and_remove()

    assert created_configs == [daytona_config]
    assert get_ids == ["sandbox-id"]
    assert deleted_timeouts == [10]


@pytest.mark.asyncio
async def test_create_container_accepts_configured_daytona_factory() -> None:
    """Test that create_container accepts DaytonaContainerFactory."""
    daytona_config = daytona.DaytonaConfig(api_key="test-key", api_url="https://daytona.example")
    container_factory = ares_daytona.DaytonaContainerFactory(
        daytona_config=daytona_config,
    )

    container = await base.create_container(
        container_factory=container_factory,
        container_prefix="test",
        image_name="python:3.12",
    )

    assert isinstance(container, ares_daytona.DaytonaContainer)
    assert container.image == "python:3.12"
    assert container.daytona_config is daytona_config
    assert container.name is not None
    assert container.name.startswith("ares.test.12")
