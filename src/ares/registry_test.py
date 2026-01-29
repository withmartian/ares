"""Tests for the registry system."""

from collections.abc import Sequence
import dataclasses
from typing import Any

import pytest

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
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> Any:
        """Return test data with received parameters."""
        return {
            "selector": selector,
            "container_factory": container_factory,
            "tracker": tracker,
        }


def test_list_presets():
    """Test that default presets are registered."""
    presets = registry._list_presets()
    assert len(presets) > 1
    assert "sbv-mswea" in presets
    assert "tbench-mswea" in presets


def test_info_all_presets():
    """Test info() without arguments returns all presets."""
    result = registry.info()
    assert isinstance(result, Sequence)
    assert len(result) >= 1
    assert any(env_info.name == "sbv-mswea" for env_info in result)


def test_info_specific_preset():
    """Test info() with a specific preset name."""
    result = registry.info("sbv-mswea")
    assert isinstance(result, registry.EnvironmentInfo)
    assert result.name == "sbv-mswea"


def test_info_missing_preset():
    """Test info() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        registry.info("nonexistent:preset")


def test_register_custom_preset():
    """Test registering a custom preset."""
    spec = _MockEnvSpec(name="test-custom", description="Test preset", num_tasks=10)
    registry.register_preset("test-custom", spec)

    # Verify it's registered
    assert "test-custom" in registry._list_presets()

    # Verify info includes task count
    result = registry.info("test-custom")
    assert isinstance(result, registry.EnvironmentInfo)
    assert result.name == "test-custom"
    assert result.num_tasks == 10

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

    with pytest.raises(ValueError, match="contains invalid characters"):
        registry.register_preset("invalid:name", spec)


def test_unregister_missing_preset():
    """Test that unregistering a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister_preset("nonexistent:preset")


def test_make_missing_preset():
    """Test that make() with a non-existent preset raises KeyError."""
    with pytest.raises(KeyError, match="not found"):
        registry.make("nonexistent-preset")


def test_make_with_params():
    """Test make() passes parameters to the spec's get_env() method."""
    spec = _MockEnvSpec(name="test-params", description="Test preset")
    registry.register_preset("test-params", spec)

    # Test with default container_factory
    result: Any = registry.make("test-params")
    assert result["container_factory"] == docker.DockerContainer
    assert result["tracker"] is None
    assert isinstance(result["selector"], registry.SliceSelector)
    assert result["selector"].start is None
    assert result["selector"].end is None

    # Test with explicit tracker
    test_tracker = stat_tracker.NullStatTracker()
    result = registry.make("test-params", tracker=test_tracker)
    assert result["container_factory"] == docker.DockerContainer
    assert result["tracker"] == test_tracker

    # Clean up
    registry.unregister_preset("test-params")


def test_clear_registry():
    """Test clearing the registry."""
    # Clear registry
    registry.clear_registry()
    assert len(registry._list_presets()) == 0

    # Re-register default presets
    from ares import presets

    presets._register_default_presets()

    # Verify defaults are back (should only have default presets)
    default_presets = set(registry._list_presets())
    assert "sbv-mswea" in default_presets


# Selector parsing tests


def test_parse_selector_no_selector():
    """Test parsing preset ID without selector."""
    preset_id, selector = registry._parse_selector("test-preset")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.SliceSelector)
    assert selector.start is None
    assert selector.end is None


def test_parse_selector_single_index():
    """Test parsing single index selector."""
    preset_id, selector = registry._parse_selector("test-preset:5")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.IndexSelector)
    assert selector.index == 5


def test_parse_selector_slice():
    """Test parsing slice selector."""
    preset_id, selector = registry._parse_selector("test-preset:0:10")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.SliceSelector)
    assert selector.start == 0
    assert selector.end == 10


def test_parse_selector_slice_start_only():
    """Test parsing slice with start only (e.g., :5:)."""
    preset_id, selector = registry._parse_selector("test-preset:5:")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.SliceSelector)
    assert selector.start == 5
    assert selector.end is None


def test_parse_selector_slice_end_only():
    """Test parsing slice with end only (e.g., ::10)."""
    preset_id, selector = registry._parse_selector("test-preset::10")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.SliceSelector)
    assert selector.start is None
    assert selector.end == 10


def test_parse_selector_shard():
    """Test parsing shard selector."""
    preset_id, selector = registry._parse_selector("test-preset@2/8")
    assert preset_id == "test-preset"
    assert isinstance(selector, registry.ShardSelector)
    assert selector.shard_index == 2
    assert selector.total_shards == 8


def test_parse_selector_invalid_empty_index():
    """Test that empty index raises ValueError."""
    with pytest.raises(ValueError, match="Index cannot be empty"):
        registry._parse_selector("test-preset:")


def test_parse_selector_invalid_negative_index():
    """Test that negative index raises ValueError."""
    with pytest.raises(ValueError, match="must be non-negative"):
        registry._parse_selector("test-preset:-1")


def test_parse_selector_invalid_slice_order():
    """Test that start >= end raises ValueError."""
    with pytest.raises(ValueError, match="Start must be less than end"):
        registry._parse_selector("test-preset:10:5")


def test_parse_selector_invalid_shard_index():
    """Test that shard_index >= total_shards raises ValueError."""
    with pytest.raises(ValueError, match="must be less than total shards"):
        registry._parse_selector("test-preset@8/8")


def test_parse_selector_invalid_shard_empty():
    """Test that empty shard specification raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        registry._parse_selector("test-preset@")


def test_parse_selector_invalid_shard_missing_slash():
    """Test that shard without slash raises ValueError."""
    with pytest.raises(ValueError, match="Expected 'dataset@shard/total'"):
        registry._parse_selector("test-preset@2")


# Selector execution tests


def test_index_selector():
    """Test IndexSelector selects single item."""
    selector = registry.IndexSelector(index=2)
    tasks = ["a", "b", "c", "d", "e"]
    result = selector(tasks)
    assert result == ["c"]


def test_slice_selector_full():
    """Test SliceSelector with both start and end."""
    selector = registry.SliceSelector(start=1, end=4)
    tasks = ["a", "b", "c", "d", "e"]
    result = selector(tasks)
    assert result == ["b", "c", "d"]


def test_slice_selector_start_only():
    """Test SliceSelector with start only."""
    selector = registry.SliceSelector(start=2, end=None)
    tasks = ["a", "b", "c", "d", "e"]
    result = selector(tasks)
    assert result == ["c", "d", "e"]


def test_slice_selector_end_only():
    """Test SliceSelector with end only."""
    selector = registry.SliceSelector(start=None, end=3)
    tasks = ["a", "b", "c", "d", "e"]
    result = selector(tasks)
    assert result == ["a", "b", "c"]


def test_slice_selector_all():
    """Test SliceSelector with no bounds selects all."""
    selector = registry.SliceSelector(start=None, end=None)
    tasks = ["a", "b", "c", "d", "e"]
    result = selector(tasks)
    assert result == ["a", "b", "c", "d", "e"]


def test_shard_selector_even_distribution():
    """Test ShardSelector distributes tasks evenly."""
    tasks = list(range(100))

    # 4 shards should each get 25 tasks
    shard0 = registry.ShardSelector(shard_index=0, total_shards=4)(tasks)
    shard1 = registry.ShardSelector(shard_index=1, total_shards=4)(tasks)
    shard2 = registry.ShardSelector(shard_index=2, total_shards=4)(tasks)
    shard3 = registry.ShardSelector(shard_index=3, total_shards=4)(tasks)

    assert len(shard0) == 25
    assert len(shard1) == 25
    assert len(shard2) == 25
    assert len(shard3) == 25

    # Verify all tasks are included and no duplicates
    all_tasks = list(shard0) + list(shard1) + list(shard2) + list(shard3)
    assert sorted(all_tasks) == tasks


def test_shard_selector_uneven_distribution():
    """Test ShardSelector with uneven task count."""
    tasks = list(range(10))

    # 3 shards: should be [3, 3, 4] or similar
    shard0 = registry.ShardSelector(shard_index=0, total_shards=3)(tasks)
    shard1 = registry.ShardSelector(shard_index=1, total_shards=3)(tasks)
    shard2 = registry.ShardSelector(shard_index=2, total_shards=3)(tasks)

    # All shards should be within 1 of each other
    lengths = [len(shard0), len(shard1), len(shard2)]
    assert max(lengths) - min(lengths) <= 1

    # Verify all tasks are included and no duplicates
    all_tasks = list(shard0) + list(shard1) + list(shard2)
    assert sorted(all_tasks) == tasks


def test_register_preset_invalid_characters():
    """Test that registering preset with invalid characters raises ValueError."""
    spec = _MockEnvSpec(name="invalid@name", description="Test preset")

    with pytest.raises(ValueError, match="contains invalid characters"):
        registry.register_preset("invalid@name", spec)


def test_make_with_selector():
    """Test make() with selector syntax."""
    spec = _MockEnvSpec(name="test-selector", description="Test preset", num_tasks=10)
    registry.register_preset("test-selector", spec)

    # Test single index
    result: Any = registry.make("test-selector:5")
    assert isinstance(result["selector"], registry.IndexSelector)
    assert result["selector"].index == 5

    # Test slice
    result = registry.make("test-selector:0:5")
    assert isinstance(result["selector"], registry.SliceSelector)
    assert result["selector"].start == 0
    assert result["selector"].end == 5

    # Test shard
    result = registry.make("test-selector@2/4")
    assert isinstance(result["selector"], registry.ShardSelector)
    assert result["selector"].shard_index == 2
    assert result["selector"].total_shards == 4

    # Clean up
    registry.unregister_preset("test-selector")


# Decorator tests


def test_register_env_decorator_basic():
    """Test basic @register_env decorator functionality."""

    @registry.register_env(
        name="test-decorator",
        description="Test decorator registration",
        num_tasks=42,
    )
    def create_test_env(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> dict[str, Any]:
        """Test environment factory."""
        return {
            "selector": selector,
            "container_factory": container_factory,
            "tracker": tracker,
        }

    # Verify preset is registered
    assert "test-decorator" in registry._list_presets()

    # Verify info is correct
    info_result = registry.info("test-decorator")
    assert isinstance(info_result, registry.EnvironmentInfo)
    assert info_result.name == "test-decorator"
    assert info_result.num_tasks == 42
    assert info_result.description == "Test decorator registration"

    # Clean up
    registry.unregister_preset("test-decorator")


def test_register_env_decorator_make():
    """Test that environments created via make() work with decorator-registered presets."""

    @registry.register_env(
        name="test-decorator-make",
        description="Test make with decorator",
        num_tasks=10,
    )
    def create_test_env(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> dict[str, Any]:
        """Test environment factory."""
        return {
            "selector": selector,
            "container_factory": container_factory,
            "tracker": tracker,
        }

    # Test make() with default parameters
    result: Any = registry.make("test-decorator-make")
    assert result["container_factory"] == docker.DockerContainer
    assert result["tracker"] is None
    assert isinstance(result["selector"], registry.SliceSelector)

    # Test make() with custom tracker
    test_tracker = stat_tracker.NullStatTracker()
    result = registry.make("test-decorator-make", tracker=test_tracker)
    assert result["tracker"] == test_tracker

    # Clean up
    registry.unregister_preset("test-decorator-make")


def test_register_env_decorator_with_selector():
    """Test that decorator-registered presets work with selector syntax."""

    @registry.register_env(
        name="test-decorator-selector",
        description="Test selectors with decorator",
        num_tasks=100,
    )
    def create_test_env(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> dict[str, Any]:
        """Test environment factory."""
        return {
            "selector": selector,
            "container_factory": container_factory,
            "tracker": tracker,
        }

    # Test single index
    result: Any = registry.make("test-decorator-selector:5")
    assert isinstance(result["selector"], registry.IndexSelector)
    assert result["selector"].index == 5

    # Test slice
    result = registry.make("test-decorator-selector:0:10")
    assert isinstance(result["selector"], registry.SliceSelector)
    assert result["selector"].start == 0
    assert result["selector"].end == 10

    # Test shard
    result = registry.make("test-decorator-selector@3/8")
    assert isinstance(result["selector"], registry.ShardSelector)
    assert result["selector"].shard_index == 3
    assert result["selector"].total_shards == 8

    # Clean up
    registry.unregister_preset("test-decorator-selector")


def test_register_env_decorator_returns_function():
    """Test that decorator returns the original function unchanged."""

    @registry.register_env(
        name="test-decorator-return",
        description="Test return value",
        num_tasks=1,
    )
    def create_test_env(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> str:
        """Test environment factory that returns a simple string."""
        del selector, container_factory, tracker  # Unused in test
        return "test-result"

    # Verify we can call the decorated function directly
    selector = registry.SliceSelector(start=None, end=None)
    result = create_test_env(
        selector=selector,
        container_factory=docker.DockerContainer,
        tracker=None,
    )
    assert result == "test-result"

    # Clean up
    registry.unregister_preset("test-decorator-return")


def test_register_env_decorator_duplicate_name():
    """Test that decorator raises ValueError for duplicate names."""

    @registry.register_env(
        name="test-decorator-duplicate",
        description="First registration",
        num_tasks=1,
    )
    def create_first_env(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> str:
        """First environment factory."""
        del selector, container_factory, tracker  # Unused in test
        return "first"

    # Attempting to register a second preset with the same name should fail
    with pytest.raises(ValueError, match="already registered"):

        @registry.register_env(
            name="test-decorator-duplicate",
            description="Second registration",
            num_tasks=1,
        )
        def create_second_env(
            *,
            selector: registry.TaskSelector,
            container_factory: containers.ContainerFactory,
            tracker: stat_tracker.StatTracker | None = None,
        ) -> str:
            """Second environment factory."""
            del selector, container_factory, tracker  # Unused in test
            return "second"

    # Clean up
    registry.unregister_preset("test-decorator-duplicate")


def test_register_env_decorator_invalid_name():
    """Test that decorator raises ValueError for invalid names."""

    with pytest.raises(ValueError, match="contains invalid characters"):

        @registry.register_env(
            name="invalid:name",
            description="Invalid name test",
            num_tasks=1,
        )
        def create_invalid_env(
            *,
            selector: registry.TaskSelector,
            container_factory: containers.ContainerFactory,
            tracker: stat_tracker.StatTracker | None = None,
        ) -> str:
            """Environment with invalid name."""
            del selector, container_factory, tracker  # Unused in test
            return "invalid"


def test_register_env_decorator_defaults():
    """Test that decorator uses function name and docstring as defaults."""

    @registry.register_env(num_tasks=42)
    def my_test_environment(
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> dict[str, Any]:
        """A test environment for validating defaults."""
        return {
            "selector": selector,
            "container_factory": container_factory,
            "tracker": tracker,
        }

    # Verify preset is registered with function name
    assert "my_test_environment" in registry._list_presets()

    # Verify info uses function name and docstring
    spec = registry._REGISTRY["my_test_environment"]
    env_info = spec.get_info()

    assert env_info.name == "my_test_environment"
    assert env_info.description == "A test environment for validating defaults."
    assert env_info.num_tasks == 42

    # Clean up
    registry.unregister_preset("my_test_environment")
