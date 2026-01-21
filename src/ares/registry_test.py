"""Tests for the registry system."""

import pytest

from ares import info
from ares import make
from ares import registry


def test_list_presets():
    """Test that default presets are registered."""
    presets = registry.list_presets()
    assert len(presets) > 0
    assert "swebench:verified" in presets
    assert "swebench:lite" in presets
    assert "harbor:easy" in presets


def test_info_all_presets():
    """Test info() without arguments returns all presets."""
    result = info()
    assert "Available presets:" in result
    assert "swebench:verified" in result
    assert "swebench:lite" in result


def test_info_specific_preset():
    """Test info() with a specific preset name."""
    result = info("swebench:lite")
    assert "swebench:lite" in result


def test_info_missing_preset():
    """Test info() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        info("nonexistent:preset")


def test_register_custom_preset():
    """Test registering a custom preset."""

    def my_factory(**kwargs):
        return {"custom": True, **kwargs}

    registry.register_preset("test:custom", my_factory, "Test preset")  # type: ignore[arg-type]

    # Verify it's registered
    assert "test:custom" in registry.list_presets()

    # Clean up
    registry.unregister_preset("test:custom")


def test_register_duplicate_preset():
    """Test that registering a duplicate preset raises ValueError."""

    def my_factory(**_kwargs):
        return {"custom": True}

    registry.register_preset("test:duplicate", my_factory)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="already registered"):
        registry.register_preset("test:duplicate", my_factory)  # type: ignore[arg-type]

    # Clean up
    registry.unregister_preset("test:duplicate")


def test_unregister_missing_preset():
    """Test that unregistering a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister_preset("nonexistent:preset")


def test_make_missing_preset():
    """Test that make() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        make("nonexistent:preset")


def test_make_with_kwargs():
    """Test make() passes kwargs to the factory."""

    def my_factory(**kwargs):
        return kwargs

    registry.register_preset("test:kwargs", my_factory)  # type: ignore[arg-type]

    result = make("test:kwargs", step_limit=50, custom_arg="test")
    assert result["step_limit"] == 50
    assert result["custom_arg"] == "test"

    # Clean up
    registry.unregister_preset("test:kwargs")


def test_make_with_task_index_suffix():
    """Test make() with :N suffix for task selection."""

    def my_factory(task_index=None, **kwargs):
        return {"task_index": task_index, **kwargs}

    registry.register_preset("test:indexed", my_factory)  # type: ignore[arg-type]

    # Test with suffix
    result = make("test:indexed:5")
    assert result["task_index"] == 5

    # Test with suffix and kwargs
    result = make("test:indexed:10", step_limit=100)
    assert result["task_index"] == 10
    assert result["step_limit"] == 100

    # Clean up
    registry.unregister_preset("test:indexed")


def test_clear_registry():
    """Test clearing the registry."""
    # Save original presets
    original_presets = set(registry.list_presets())

    # Clear registry
    registry.clear_registry()
    assert len(registry.list_presets()) == 0

    # Re-register default presets
    from ares import presets

    presets._register_default_presets()

    # Verify defaults are back
    assert set(registry.list_presets()) == original_presets
