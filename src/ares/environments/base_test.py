"""Tests for environment utilities."""

import pathlib

import pytest

from ares.containers import containers
from ares.environments import base
from ares.testing import mock_container


class _CapturingContainerFactory:
    name: str | None = None

    @classmethod
    def from_image(
        cls,
        *,
        image: str,
        name: str | None = None,
        resources: containers.Resources | None = None,
        default_workdir: str | None = None,
    ) -> mock_container.MockContainer:
        del image, resources, default_workdir
        cls.name = name
        return mock_container.MockContainer()

    @classmethod
    def from_dockerfile(
        cls,
        *,
        dockerfile_path: pathlib.Path | str,
        name: str | None = None,
        resources: containers.Resources | None = None,
        default_workdir: str | None = None,
    ) -> mock_container.MockContainer:
        del dockerfile_path, resources, default_workdir
        cls.name = name
        return mock_container.MockContainer()


@pytest.mark.asyncio
async def test_create_container_sanitizes_image_tag_in_name() -> None:
    await base.create_container(
        container_factory=_CapturingContainerFactory,
        container_prefix="harbor_env",
        image_name="alexgshaw/gpt2-codegolf:20251031",
    )

    assert _CapturingContainerFactory.name is not None
    assert ":" not in _CapturingContainerFactory.name
    assert "gpt2-codegolf-20251031" in _CapturingContainerFactory.name
