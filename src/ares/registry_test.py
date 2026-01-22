"""Tests for the registry system."""

import dataclasses
from typing import Any

import pytest

from ares import info
from ares import make
from ares import registry
from ares.containers import containers
from ares.containers import docker
from ares.experiment_tracking import stat_tracker


@dataclasses.dataclass(frozen=True)
class _MockEnvSpec:
    """Test environment spec for unit tests.

    Attributes:
        name: Name for the preset.
        description: Description for the preset.
        num_tasks: Number of tasks to report.
    """

    name: str
    description: str
    num_tasks: int = 1

    def get_info(self) -> registry.EnvironmentInfo:
        """Return test metadata."""
        return registry.EnvironmentInfo(
            name=self.name,
            description=self.description,
            num_tasks=self.num_tasks,
        )

    def get_env(
        self,
        *,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> Any:
        """Return test data with received parameters."""
        return {
            "container_factory": container_factory,
            "tracker": tracker,
        }


def test_list_presets():
    """Test that default presets are registered."""
    presets = registry._list_presets()
    assert len(presets) == 1
    assert "sbv-mswea" in presets


def test_info_all_presets():
    """Test info() without arguments returns all presets."""
    result = info()
    assert "Available presets:" in result
    assert "sbv-mswea" in result


def test_info_specific_preset():
    """Test info() with a specific preset name."""
    result = info("sbv-mswea")
    assert "sbv-mswea" in result


def test_info_missing_preset():
    """Test info() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        info("nonexistent:preset")


def test_register_custom_preset():
    """Test registering a custom preset."""
    spec = _MockEnvSpec(name="test-custom", description="Test preset", num_tasks=10)
    registry.register_preset("test-custom", spec)

    # Verify it's registered
    assert "test-custom" in registry._list_presets()

    # Verify info includes task count
    result = info("test-custom")
    assert "test-custom" in result
    assert "10" in result

    # Clean up
    registry.unregister_preset("test-custom")


def test_register_duplicate_preset():
    """Test that registering a duplicate preset raises ValueError."""
    spec = _MockEnvSpec(name="test-duplicate", description="Test preset")
    registry.register_preset("test-duplicate", spec)

    with pytest.raises(ValueError, match="already registered"):
        registry.register_preset("test-duplicate", spec)

    # Clean up
    registry.unregister_preset("test-duplicate")


def test_register_preset_with_colon():
    """Test that registering a preset with a colon raises ValueError."""
    spec = _MockEnvSpec(name="invalid:name", description="Test preset")

    with pytest.raises(ValueError, match="cannot contain colons"):
        registry.register_preset("invalid:name", spec)


def test_unregister_missing_preset():
    """Test that unregistering a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister_preset("nonexistent:preset")


def test_make_missing_preset():
    """Test that make() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        make("nonexistent-preset")


def test_make_with_params():
    """Test make() passes parameters to the spec's get_env() method."""
    spec = _MockEnvSpec(name="test-params", description="Test preset")
    registry.register_preset("test-params", spec)

    # Test with default container_factory
    result: Any = make("test-params")
    assert result["container_factory"] == docker.DockerContainer
    assert result["tracker"] is None

    # Test with explicit tracker
    test_tracker = stat_tracker.NullStatTracker()
    result = make("test-params", tracker=test_tracker)
    assert result["container_factory"] == docker.DockerContainer
    assert result["tracker"] == test_tracker

    # Clean up
    registry.unregister_preset("test-params")


def test_clear_registry():
    """Test clearing the registry."""
    # Save original presets
    original_presets = set(registry._list_presets())

    # Clear registry
    registry.clear_registry()
    assert len(registry._list_presets()) == 0

    # Re-register default presets
    from ares import presets

    presets._register_default_presets()

    # Verify defaults are back
    assert set(registry._list_presets()) == original_presets
