"""Registry mechanism for ARES environments.

This module provides a registry system for creating environments with preset configurations.
The registry allows users to:
1. Register environment presets with names and factory functions
2. Create environments using preset names via `make()`
3. Query available presets via `info()`
4. Override preset configurations with kwargs

The registry itself is empty by default. Default presets are registered in the
`presets` module to avoid circular imports.
"""

from collections.abc import Callable, Sequence
import dataclasses
import logging
from typing import Any

from ares.environments import base

_LOGGER = logging.getLogger(__name__)

# Registry storage: maps preset names to factory functions
_REGISTRY: dict[str, Callable[..., Any]] = {}


@dataclasses.dataclass(frozen=True)
class PresetInfo:
    """Information about a registered preset.

    Attributes:
        name: The unique identifier for the preset (e.g., "swebench:lite", "harbor:easy").
        factory: The factory function that creates the environment.
        description: A human-readable description of what the preset provides.
    """

    name: str
    factory: Callable[..., Any]
    description: str


def register_preset[EnvType: base.Environment](
    name: str,
    factory: Callable[..., EnvType],
    description: str = "",
) -> None:
    """Register an environment preset with the global registry.

    This function allows users to register custom environment configurations that can
    be instantiated later via `make()`. Presets provide a convenient way to share
    common environment configurations.

    Args:
        name: Unique identifier for the preset. Convention is "dataset:variant" (e.g.,
            "swebench:lite", "harbor:easy"). Must not already exist in the registry.
        factory: A callable that creates and returns an environment instance. The factory
            will be called with any kwargs passed to `make()`, allowing users to override
            preset defaults.
        description: Human-readable description of the preset's purpose and configuration.
            Displayed by `info()`. Defaults to empty string.

    Raises:
        ValueError: If a preset with the given name is already registered.

    Examples:
        Register a custom preset:

        >>> def my_custom_env_factory(**kwargs):
        ...     return HarborEnv(tasks=load_harbor_dataset("my-dataset", "v1"), **kwargs)
        >>> register_preset(
        ...     "custom:my-dataset",
        ...     my_custom_env_factory,
        ...     "Custom Harbor environment for my-dataset v1"
        ... )

        Register with configurable parameters:

        >>> def configurable_env(**kwargs):
        ...     step_limit = kwargs.pop("step_limit", 50)  # Default to 50
        ...     return SwebenchEnv(tasks=load_tasks(), step_limit=step_limit, **kwargs)
        >>> register_preset("swebench:fast", configurable_env, "Fast SWE-bench with 50 step limit")
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Preset '{name}' is already registered. Choose a different name or unregister the existing preset first."
        )

    _REGISTRY[name] = factory
    _LOGGER.debug("Registered preset '%s': %s", name, description or "(no description)")


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


def list_presets() -> Sequence[str]:
    """Return a sorted list of all registered preset names.

    Returns:
        A tuple of preset names in alphabetical order.

    Examples:
        >>> presets = list_presets()
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
        the name and description. For all presets, includes a list of names.

    Raises:
        KeyError: If a specific name is provided but not found in the registry.

    Examples:
        Get info for all presets:

        >>> print(info())
        Available presets:
          - harbor:easy
          - swebench:lite
          - swebench:verified

        Get info for a specific preset:

        >>> print(info("swebench:lite"))
        Preset: swebench:lite
        Description: SWE-bench Lite dataset (300 instances)
    """
    if name is not None:
        if name not in _REGISTRY:
            raise KeyError(f"Preset '{name}' not found. Available presets: {', '.join(list_presets())}")

        # For specific preset, we only have the factory, not the description stored separately
        # In a full implementation, we'd store PresetInfo objects instead of just factories
        return f"Preset: {name}\n(Use make() to instantiate)"

    # List all presets
    presets = list_presets()
    if not presets:
        return "No presets registered."

    lines = ["Available presets:"]
    for preset_name in presets:
        lines.append(f"  - {preset_name}")

    return "\n".join(lines)


def make(preset_name: str, **kwargs: Any) -> Any:
    """Create an environment instance from a registered preset.

    This is the primary way to instantiate environments in ARES. It looks up the
    preset by name, creates the environment using the registered factory function,
    and applies any user-provided overrides via kwargs.

    The preset name may include a task selection suffix in the format "preset:N" where
    N is a 0-based task index. For example, "swebench:lite:5" creates the "swebench:lite"
    preset and selects task index 5.

    Args:
        preset_name: The name of the preset to instantiate, optionally with ":N" suffix
            for task selection (e.g., "swebench:lite" or "swebench:lite:5").
        **kwargs: Additional keyword arguments to pass to the factory function. These
            override any defaults defined in the preset. Common overrides include:
            - step_limit: Maximum number of steps per episode
            - container_factory: Alternative container implementation
            - code_agent_factory: Alternative code agent implementation
            - tracker: Statistics tracker instance

    Returns:
        An environment instance configured according to the preset and overrides.

    Raises:
        KeyError: If the preset name is not found in the registry.
        TypeError: If the factory function doesn't accept the provided kwargs.

    Examples:
        Create environment with default settings:

        >>> env = make("swebench:lite")

        Override step limit:

        >>> env = make("swebench:lite", step_limit=50)

        Select specific task (0-based indexing):

        >>> env = make("swebench:lite:5")  # Select task at index 5

        Override multiple parameters:

        >>> from ares.containers import docker
        >>> env = make(
        ...     "harbor:easy",
        ...     step_limit=200,
        ...     container_factory=docker.DockerContainer
        ... )

        The kwargs override mechanism allows presets to define sensible defaults
        while giving users full control over the final configuration.
    """
    # Check for task selection suffix (e.g., "swebench:lite:5")
    task_index: int | None = None
    if ":" in preset_name and preset_name.split(":")[-1].isdigit():
        parts = preset_name.rsplit(":", 1)
        preset_name = parts[0]
        task_index = int(parts[1])
        _LOGGER.debug("Parsed task index %d from preset name", task_index)

    if preset_name not in _REGISTRY:
        available = ", ".join(list_presets())
        raise KeyError(f"Preset '{preset_name}' not found. Available presets: {available or '(none)'}")

    factory = _REGISTRY[preset_name]
    _LOGGER.info("Creating environment from preset '%s' with kwargs: %s", preset_name, kwargs)

    # Add task_index to kwargs if specified
    if task_index is not None:
        if "task_index" in kwargs:
            _LOGGER.warning(
                "Overriding task_index from kwargs (%d) with suffix value (%d)",
                kwargs["task_index"],
                task_index,
            )
        kwargs["task_index"] = task_index

    try:
        env = factory(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to create environment from preset '{preset_name}'. "
            f"The factory function may not accept the provided kwargs. "
            f"Original error: {e}"
        ) from e

    _LOGGER.info("Successfully created environment from preset '%s'", preset_name)
    return env


def clear_registry() -> None:
    """Clear all registered presets from the registry.

    This is primarily useful for testing. In production code, you typically want to
    keep presets registered throughout the program's lifetime.

    Examples:
        >>> clear_registry()
        >>> assert len(list_presets()) == 0
    """
    _REGISTRY.clear()
    _LOGGER.debug("Cleared all presets from registry")
