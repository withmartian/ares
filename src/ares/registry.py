"""Registry mechanism for ARES environments.

This module provides a registry system for creating environments with preset configurations.
The registry allows users to:
1. Register environment presets with names and specs
2. Create environments using preset names via `make()`
3. Query available presets via `info()`
4. Query information about a specific preset via `info(preset_id)`

The registry itself is empty by default. Default presets are registered in the
`presets` module to avoid circular imports.
"""

from collections.abc import Sequence
import dataclasses
import logging
import re
from typing import Protocol, overload

from ares.containers import containers
from ares.containers import docker
from ares.environments import base
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)

# Valid preset name pattern: alphanumeric + hyphens, underscores, and forward slashes
_PRESET_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_/-]+$")


class TaskSelector(Protocol):
    """Protocol for task selection strategies."""

    def __call__[T](self, tasks: Sequence[T]) -> Sequence[T]:
        """Filter tasks based on selection strategy.

        Args:
            tasks: Sequence of tasks to filter.

        Returns:
            Filtered sequence of tasks.
        """
        ...


@dataclasses.dataclass(frozen=True)
class IndexSelector:
    """Select a single task by index.

    Attributes:
        index: Index of the task to select.
    """

    index: int

    def __call__[T](self, tasks: Sequence[T]) -> Sequence[T]:
        """Select single task at index."""
        return [tasks[self.index]]


@dataclasses.dataclass(frozen=True)
class SliceSelector:
    """Select tasks by slice (Python-style).

    Attributes:
        start: Starting index (inclusive). None means start from beginning.
        end: Ending index (exclusive). None means go to end.
    """

    start: int | None
    end: int | None

    def __call__[T](self, tasks: Sequence[T]) -> Sequence[T]:
        """Filter tasks based on slice specification."""
        return tasks[self.start : self.end]


@dataclasses.dataclass(frozen=True)
class ShardSelector:
    """Select tasks by shard index.

    Shards distribute tasks as evenly as possible across multiple partitions.
    Uses floating point division with rounding to ensure all shards differ by at most 1 task.

    Attributes:
        shard_index: Which shard to select (0-indexed).
        total_shards: Total number of shards.
    """

    shard_index: int
    total_shards: int

    def __call__[T](self, tasks: Sequence[T]) -> Sequence[T]:
        """Filter tasks based on shard specification.

        Uses floating point division to distribute tasks evenly across shards.
        This ensures that all shards have sizes within 1 of each other, avoiding
        the pattern where early shards get extra tasks and later ones are smaller.

        Example: 100 tasks / 3 shards = [33, 33, 34] instead of [34, 34, 32]
        """
        total = len(tasks)

        # Use floating point division to compute shard boundaries
        # This distributes tasks as evenly as possible across all shards
        start = round(self.shard_index * total / self.total_shards)
        end = round((self.shard_index + 1) * total / self.total_shards)

        return tasks[start:end]


def _parse_selector(selector_str: str) -> tuple[str, TaskSelector]:
    """Parse a task selector string into preset ID and selector.

    Supported syntaxes:
    - "dataset" - Select all tasks
    - "dataset:5" - Select single task at index 5
    - "dataset:0:10" - Select slice from index 0 to 9 (Python-style)
    - "dataset:5:" - Select from index 5 to end
    - "dataset::10" - Select from start to index 10
    - "dataset@2/8" - Select shard 2 out of 8 total shards

    Args:
        selector_str: The selector string to parse.

    Returns:
        A tuple of (preset_id, selector).

    Raises:
        ValueError: If the selector syntax is invalid.
    """
    # Check for shard syntax: dataset@shard/total
    if "@" in selector_str:
        parts = selector_str.split("@")
        if len(parts) != 2:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Expected 'dataset@shard/total'.")

        preset_id = parts[0]
        shard_spec = parts[1]

        if not shard_spec:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Shard specification cannot be empty.")

        if "/" not in shard_spec:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Expected 'dataset@shard/total'.")

        shard_parts = shard_spec.split("/")
        if len(shard_parts) != 2:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Expected 'dataset@shard/total'.")

        if not shard_parts[0] or not shard_parts[1]:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Shard and total cannot be empty.")

        try:
            shard_index = int(shard_parts[0])
            total_shards = int(shard_parts[1])
        except ValueError as e:
            raise ValueError(f"Invalid shard syntax: '{selector_str}'. Shard and total must be integers.") from e

        if shard_index < 0 or total_shards <= 0:
            raise ValueError(f"Invalid shard values: shard_index={shard_index}, total_shards={total_shards}.")
        if shard_index >= total_shards:
            raise ValueError(f"Shard index {shard_index} must be less than total shards {total_shards}.")

        return preset_id, ShardSelector(shard_index=shard_index, total_shards=total_shards)

    # Check for slice/single index syntax: dataset:start:end or dataset:idx
    if ":" in selector_str:
        preset_id, *parts = selector_str.split(":")

        if not preset_id:
            raise ValueError(f"Invalid selector syntax: '{selector_str}'. Preset ID cannot be empty.")

        if len(parts) == 1:
            # Single index: dataset:5
            idx_str = parts[0]

            if not idx_str:
                raise ValueError(f"Invalid index syntax: '{selector_str}'. Index cannot be empty.")
            try:
                index = int(idx_str)
            except ValueError as e:
                raise ValueError(f"Invalid index syntax: '{selector_str}'. Index must be an integer.") from e

            if index < 0:
                raise ValueError(f"Invalid index: {index}. Index must be non-negative.")

            return preset_id, IndexSelector(index=index)
        elif len(parts) == 2:
            # Slice: dataset:start:end
            raw_start, raw_end = parts

            try:
                # Parse start (can be empty for None)
                start = None if not raw_start else int(raw_start)
                # Parse end (can be empty for None)
                end = None if not raw_end else int(raw_end)
            except ValueError as e:
                raise ValueError(
                    f"Invalid slice syntax: '{selector_str}'. Start and end must be integers or empty."
                ) from e

            if start is not None and start < 0:
                raise ValueError(f"Invalid slice: start={start}. Start must be non-negative.")
            if end is not None and end < 0:
                raise ValueError(f"Invalid slice: end={end}. End must be non-negative.")
            if start is not None and end is not None and start >= end:
                raise ValueError(f"Invalid slice: start={start}, end={end}. Start must be less than end.")

            return preset_id, SliceSelector(start=start, end=end)
        else:
            raise ValueError(
                f"Invalid selector syntax: '{selector_str}'. Expected 'dataset:idx' or 'dataset:start:end'."
            )

    # No selector, just preset name - select all tasks
    return selector_str, SliceSelector(start=None, end=None)


@dataclasses.dataclass(frozen=True)
class EnvironmentInfo:
    """Metadata about an environment preset.

    Attributes:
        name: The unique identifier for the preset (e.g., "swebench-lite", "harbor-easy").
        description: A human-readable description of what the preset provides.
        num_tasks: The total number of tasks available in this environment dataset.
    """

    name: str
    description: str
    num_tasks: int

    def __str__(self) -> str:
        """Return formatted preset information."""
        return f"{self.name} ({self.num_tasks} tasks): {self.description}"


class EnvironmentSpec(Protocol):
    """Protocol for environment preset specifications.

    An EnvironmentSpec encapsulates both metadata about an environment preset and
    the logic to create environment instances. This separation allows querying
    information (like task count) without instantiating the expensive environment.

    The spec pattern is used throughout ARES for protocol-oriented design, enabling
    different implementations while maintaining a consistent interface.
    """

    def get_info(self) -> EnvironmentInfo:
        """Return metadata about this environment preset.

        Returns:
            Info object containing name, description, and number of tasks.
        """
        ...

    def get_env(
        self,
        *,
        selector: TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> base.Environment:
        """Create and return an environment instance.

        Args:
            selector: Task selector to filter which tasks to include.
            container_factory: Factory for creating containers. Required.
            tracker: Statistics tracker for monitoring. Optional.

        Returns:
            A configured environment instance ready for use in the RL loop.
        """
        ...


# Registry storage: maps preset names to environment specs
_REGISTRY: dict[str, EnvironmentSpec] = {}


def register_preset(
    name: str,
    spec: EnvironmentSpec,
) -> None:
    """Register an environment preset with the global registry.

    This function allows users to register custom environment specifications that can
    be instantiated later via `make()` or queried via `info()`. Presets provide a
    convenient way to share common environment configurations.

    Args:
        name: Unique identifier for the preset. Convention is "dataset-variant" (e.g.,
            "swebench-lite", "harbor-easy"). Must not already exist in the registry.
            Must contain only alphanumeric characters, hyphens, underscores, and forward slashes.
            Cannot contain colons or @ symbols as they are reserved for task selection syntax.
        spec: An EnvironmentSpec instance that provides both metadata (via get_info())
            and environment creation (via get_env()). The spec's get_env() method will
            receive any kwargs passed to `make()`, allowing users to override defaults.

    Raises:
        ValueError: If a preset with the given name is already registered, or if the
            name contains invalid characters.
    """
    if not _PRESET_NAME_PATTERN.match(name):
        raise ValueError(
            f"Preset name '{name}' contains invalid characters. "
            "Only alphanumeric characters, hyphens (-), underscores (_), and forward slashes (/) are allowed."
        )

    if name in _REGISTRY:
        raise ValueError(
            f"Preset '{name}' is already registered. Choose a different name or unregister the existing preset first."
        )

    _REGISTRY[name] = spec
    _LOGGER.debug("Registered preset '%s'", name)


def register_env(
    *,
    name: str | None = None,
    description: str | None = None,
    num_tasks: int,
):
    """Decorator for registering environment presets.

    This provides syntactic sugar for registering environment presets without
    manually creating spec classes. The decorated function should have the same
    signature as EnvironmentSpec.get_env() (minus self).

    Args:
        name: Unique identifier for the preset. Defaults to the function name.
            Same rules as register_preset().
        description: Human-readable description of what the preset provides.
            Defaults to the function's docstring.
        num_tasks: Total number of tasks available in this environment dataset.

    Returns:
        Decorator function that wraps the environment factory and registers it.

    Examples:
        Register with explicit name and description:

        >>> @register_env(
        ...     name="my-dataset",
        ...     description="My custom dataset with specific tasks",
        ...     num_tasks=100,
        ... )
        ... def create_my_env(
        ...     *,
        ...     selector: TaskSelector,
        ...     container_factory: containers.ContainerFactory,
        ...     tracker: stat_tracker.StatTracker | None = None,
        ... ) -> base.Environment:
        ...     # Load tasks, apply selector, create environment
        ...     return MyEnvironment(...)

        Register using function name and docstring:

        >>> @register_env(num_tasks=50)
        ... def my_custom_env(
        ...     *,
        ...     selector: TaskSelector,
        ...     container_factory: containers.ContainerFactory,
        ...     tracker: stat_tracker.StatTracker | None = None,
        ... ) -> base.Environment:
        ...     '''Custom environment for testing.'''
        ...     return MyEnvironment(...)
        # Registered as "my_custom_env" with description "Custom environment for testing."

        The decorated function can still be called directly:

        >>> env = create_my_env(
        ...     selector=SliceSelector(start=None, end=None),
        ...     container_factory=docker.DockerContainer,
        ... )
    """

    def decorator(func):
        """Inner decorator that creates and registers the spec."""
        # Use function name if name not provided
        preset_name = name if name is not None else func.__name__

        # Use function docstring if description not provided
        preset_description = description
        if preset_description is None:
            preset_description = func.__doc__ or ""
            # Clean up docstring (remove leading/trailing whitespace)
            preset_description = preset_description.strip()

        @dataclasses.dataclass(frozen=True)
        class _DecoratorGeneratedSpec:
            """Auto-generated EnvironmentSpec from @register_env decorator."""

            def get_info(self) -> EnvironmentInfo:
                """Return metadata provided to the decorator."""
                return EnvironmentInfo(
                    name=preset_name,
                    description=preset_description,
                    num_tasks=num_tasks,
                )

            def get_env(
                self,
                *,
                selector: TaskSelector,
                container_factory: containers.ContainerFactory,
                tracker: stat_tracker.StatTracker | None = None,
            ) -> base.Environment:
                """Delegate to the decorated function."""
                return func(
                    selector=selector,
                    container_factory=container_factory,
                    tracker=tracker,
                )

        # Register the auto-generated spec
        register_preset(preset_name, _DecoratorGeneratedSpec())

        # Return original function so it can still be called directly
        return func

    return decorator


def unregister_preset(name: str) -> None:
    """Remove a preset from the registry.

    Args:
        name: The name of the preset to unregister.

    Raises:
        KeyError: If no preset with the given name exists.

    Examples:
        >>> unregister_preset("custom:my-dataset")
    """
    if name not in _REGISTRY:
        raise KeyError(f"Preset '{name}' is not registered.")

    del _REGISTRY[name]
    _LOGGER.debug("Unregistered preset '%s'", name)


def _list_presets() -> Sequence[str]:
    """Return a sorted list of all registered preset names.

    Returns:
        A tuple of preset names in alphabetical order.

    Examples:
        >>> presets = _list_presets()
        >>> print(presets)
        ('harbor:easy', 'swebench:lite', 'swebench:verified')
    """
    return tuple(sorted(_REGISTRY.keys()))


@overload
def info(name: str) -> EnvironmentInfo: ...


@overload
def info(name: None = None) -> Sequence[EnvironmentInfo]: ...


def info(name: str | None = None) -> EnvironmentInfo | Sequence[EnvironmentInfo]:
    """Get information about registered presets.

    Args:
        name: Optional preset name to get info for. If None, returns info for all presets.

    Returns:
        If name is provided: An EnvironmentInfo object containing the preset's name,
            description, and number of tasks.
        If name is None: A sequence of EnvironmentInfo objects for all registered presets,
            or an empty sequence if no presets are registered.

    Raises:
        KeyError: If a specific name is provided but not found in the registry.

    Examples:
        Get info for all presets:

        >>> all_presets = info()
        >>> print(all_presets[0])
        sbv-mswea (500 tasks): SWE-bench Verified with mini-swe-agent

        Get info for a specific preset:

        >>> preset_info = info("sbv-mswea")
        >>> print(preset_info.num_tasks)
        500
        >>> print(preset_info)
        sbv-mswea (500 tasks): SWE-bench Verified with mini-swe-agent
    """
    if name is not None:
        if name not in _REGISTRY:
            raise KeyError(f"Preset '{name}' not found. Available presets: {', '.join(_list_presets())}")

        spec = _REGISTRY[name]
        return spec.get_info()

    # List all presets with summary information
    presets = _list_presets()
    if not presets:
        return []

    spec_infos: list[EnvironmentInfo] = []
    for preset_id in presets:
        spec = _REGISTRY[preset_id]
        spec_infos.append(spec.get_info())

    return spec_infos


def list_presets() -> str:
    """A utility function to easily list all presets.

    Returns:
        A nicely formatted string of all presets.
    """
    return "\n".join([str(x) for x in info()])


def make(
    preset_id: str,
    *,
    container_factory: containers.ContainerFactory = docker.DockerContainer,
    tracker: stat_tracker.StatTracker | None = None,
) -> base.Environment:
    """Create an environment instance from a registered preset.

    This is the primary way to instantiate environments in ARES. It looks up the
    preset by ID and creates the environment using the registered spec.

    Args:
        preset_id: The ID of the preset to instantiate, with optional task selector.
            Examples:
            - "sbv-mswea" - All tasks
            - "sbv-mswea:0" - Single task at index 0
            - "sbv-mswea:0:10" - Tasks 0-9 (slice)
            - "sbv-mswea:5:" - Tasks from index 5 to end
            - "sbv-mswea::10" - Tasks from start to index 10
            - "sbv-mswea@2/8" - Shard 2 out of 8 total shards
        container_factory: Factory for creating containers. Defaults to DockerContainer.
        tracker: Statistics tracker for monitoring. Optional.

    Returns:
        An environment instance configured according to the preset.

    Raises:
        KeyError: If the preset name is not found in the registry.
        ValueError: If the selector syntax is invalid.
        TypeError: If the spec's get_env() method doesn't accept the provided parameters.

    Examples:
        Create environment with default Docker containers:

        >>> env = make("sbv-mswea")

        Select single task:

        >>> env = make("sbv-mswea:0")

        Select slice of tasks:

        >>> env = make("sbv-mswea:0:10")

        Select shard:

        >>> env = make("sbv-mswea@2/8")

        Use Daytona containers instead:

        >>> from ares.containers import daytona
        >>> env = make("sbv-mswea", container_factory=daytona.DaytonaContainer)

        Add statistics tracking:

        >>> from ares.experiment_tracking import stat_tracker
        >>> tracker = stat_tracker.LoggingStatTracker()
        >>> env = make("sbv-mswea", tracker=tracker)
    """
    # Parse the selector syntax
    preset_id_clean, selector = _parse_selector(preset_id)

    if preset_id_clean not in _REGISTRY:
        available = ", ".join(_list_presets())
        raise KeyError(f"Preset '{preset_id_clean}' not found. Available presets: {available or '(none)'}")

    spec = _REGISTRY[preset_id_clean]
    _LOGGER.debug(
        "Creating environment from preset '%s' with selector=%s, container_factory=%s, tracker=%s",
        preset_id_clean,
        selector,
        container_factory,
        tracker,
    )

    env = spec.get_env(selector=selector, container_factory=container_factory, tracker=tracker)

    _LOGGER.debug("Successfully created environment from preset '%s'", preset_id_clean)
    return env


def clear_registry() -> None:
    """Clear all registered presets from the registry.

    This is primarily useful for testing. In production code, you typically want to
    keep presets registered throughout the program's lifetime.

    Examples:
        >>> clear_registry()
        >>> assert len(_list_presets()) == 0
    """
    _REGISTRY.clear()
    _LOGGER.debug("Cleared all presets from registry")
