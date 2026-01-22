"""Registry mechanism for ARES environments.

This module provides a registry system for creating environments with preset configurations.
The registry allows users to:
1. Register environment presets with names and specs
2. Create environments using preset names via `make()`
3. Query available presets via `info()`
4. Query information about a specific preset via `info(preset_name)`

The registry itself is empty by default. Default presets are registered in the
`presets` module to avoid circular imports.
"""

from collections.abc import Sequence
import dataclasses
import logging
from typing import Protocol

from ares.containers import containers
from ares.containers import docker
from ares.environments import base
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


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
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> base.Environment:
        """Create and return an environment instance.

        Args:
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
            Cannot contain colons as they are reserved for task selection syntax.
        spec: An EnvironmentSpec instance that provides both metadata (via get_info())
            and environment creation (via get_env()).

    Raises:
        ValueError: If a preset with the given name is already registered, or if the
            name contains a colon character.
    """
    if ":" in name:
        raise ValueError(
            f"Preset name '{name}' cannot contain colons. "
            "Colons are reserved for task selection syntax (e.g., 'preset:5')."
        )

    if name in _REGISTRY:
        raise ValueError(
            f"Preset '{name}' is already registered. Choose a different name or unregister the existing preset first."
        )

    _REGISTRY[name] = spec
    _LOGGER.debug("Registered preset '%s'", name)


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


def info(name: str | None = None) -> str:
    """Get information about registered presets.

    Args:
        name: Optional preset name to get info for. If None, returns info for all presets.

    Returns:
        A formatted string describing the preset(s). For a specific preset, includes
        the name, description, and number of tasks. For all presets, includes a summary
        table with key information.

    Raises:
        KeyError: If a specific name is provided but not found in the registry.

    Examples:
        Get info for all presets:

        >>> print(info())
        Available presets:
          - sbv-mswea (500 tasks): SWE-bench Verified with mini-swe-agent

        Get info for a specific preset:

        >>> print(info("sbv-mswea"))
        sbv-mswea (500 tasks): SWE-bench Verified with mini-swe-agent
    """
    if name is not None:
        if name not in _REGISTRY:
            raise KeyError(f"Preset '{name}' not found. Available presets: {', '.join(_list_presets())}")

        spec = _REGISTRY[name]
        return str(spec.get_info())

    # List all presets with summary information
    presets = _list_presets()
    if not presets:
        return "No presets registered."

    lines = ["Available presets:"]
    for preset_name in presets:
        spec = _REGISTRY[preset_name]
        lines.append(f"  - {spec.get_info()}")

    return "\n".join(lines)


def make(
    preset_name: str,
    *,
    container_factory: containers.ContainerFactory = docker.DockerContainer,
    tracker: stat_tracker.StatTracker | None = None,
) -> base.Environment:
    """Create an environment instance from a registered preset.

    This is the primary way to instantiate environments in ARES. It looks up the
    preset by name and creates the environment using the registered spec.

    Args:
        preset_name: The name of the preset to instantiate (e.g., "sbv-mswea").
        container_factory: Factory for creating containers. Defaults to DockerContainer.
        tracker: Statistics tracker for monitoring. Optional.

    Returns:
        An environment instance configured according to the preset.

    Raises:
        KeyError: If the preset name is not found in the registry.
        TypeError: If the spec's get_env() method doesn't accept the provided parameters.

    Examples:
        Create environment with default Docker containers:

        >>> env = make("sbv-mswea")

        Use Daytona containers instead:

        >>> from ares.containers import daytona
        >>> env = make("sbv-mswea", container_factory=daytona.DaytonaContainer)

        Add statistics tracking:

        >>> from ares.experiment_tracking import stat_tracker
        >>> tracker = stat_tracker.LoggingStatTracker()
        >>> env = make("sbv-mswea", tracker=tracker)
    """
    if preset_name not in _REGISTRY:
        available = ", ".join(_list_presets())
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {available or '(none)'}")

    spec = _REGISTRY[preset_name]
    _LOGGER.info(
        "Creating environment from preset '%s' with container_factory=%s, tracker=%s",
        preset_name,
        container_factory,
        tracker,
    )

    env = spec.get_env(container_factory=container_factory, tracker=tracker)

    _LOGGER.info("Successfully created environment from preset '%s'", preset_name)
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
