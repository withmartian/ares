"""Default preset registrations for ARES environments.

This module registers the built-in environment presets with the global registry.
It is imported automatically when ARES is imported, ensuring default presets are
always available.

Do not add the registry mechanism itself to this module - that belongs in registry.py.
This module only contains preset registrations to avoid circular imports.
"""

from collections.abc import Sequence
import functools
import logging
from typing import Any

from ares import registry
from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.containers import daytona as ares_daytona
from ares.environments import swebench_env
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache
def _swebench_lite_tasks() -> tuple[swebench_env.SwebenchTask, ...]:
    """Load SWE-bench Lite dataset.

    SWE-bench Lite is a curated subset of 300 instances from the full SWE-bench dataset,
    designed for faster iteration and testing.

    Returns:
        Tuple of SwebenchTask instances from the Lite dataset.
    """
    import datasets

    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    result: list[swebench_env.SwebenchTask] = []
    for task in ds:
        result.append(swebench_env.SwebenchTask.model_validate(task))
    return tuple(result)


def _make_swebench_verified(
    *,
    task_index: int | None = None,
    container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
    code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
    step_limit: int = 100,
    tracker: stat_tracker.StatTracker | None = None,
    **kwargs: Any,
) -> swebench_env.SweBenchEnv:
    """Factory function for SWE-bench Verified environment.

    The SWE-bench Verified dataset contains 500+ rigorously validated instances from
    the original SWE-bench dataset, with verified problem statements and test cases.

    Args:
        task_index: Optional 0-based index to select a specific task. If None, tasks
            are randomly sampled during each reset.
        container_factory: Factory for creating containers. Defaults to DaytonaContainer.
        code_agent_factory: Factory for creating code agents. Defaults to MiniSWECodeAgent.
        step_limit: Maximum number of steps per episode. Defaults to 100.
        tracker: Optional statistics tracker for monitoring performance.
        **kwargs: Additional arguments passed to SweBenchEnv constructor.

    Returns:
        Configured SweBenchEnv instance.

    Examples:
        >>> env = _make_swebench_verified()  # Random task sampling
        >>> env = _make_swebench_verified(task_index=5)  # Specific task
    """
    all_tasks = swebench_env.swebench_verified_tasks()

    # If task_index specified, use only that task
    tasks: Sequence[swebench_env.SwebenchTask]
    if task_index is not None:
        if task_index < 0 or task_index >= len(all_tasks):
            raise ValueError(
                f"task_index {task_index} out of range. "
                f"SWE-bench Verified has {len(all_tasks)} tasks (0-{len(all_tasks) - 1})."
            )
        tasks = [all_tasks[task_index]]
        _LOGGER.info("Selected task %d: %s", task_index, tasks[0].instance_id)
    else:
        tasks = all_tasks

    return swebench_env.SweBenchEnv(
        tasks=tasks,
        container_factory=container_factory,
        code_agent_factory=code_agent_factory,
        step_limit=step_limit,
        tracker=tracker,
        **kwargs,
    )


def _make_swebench_lite(
    *,
    task_index: int | None = None,
    container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
    code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
    step_limit: int = 100,
    tracker: stat_tracker.StatTracker | None = None,
    **kwargs: Any,
) -> swebench_env.SweBenchEnv:
    """Factory function for SWE-bench Lite environment.

    SWE-bench Lite is a curated subset of 300 instances, designed for faster
    experimentation and evaluation.

    Args:
        task_index: Optional 0-based index to select a specific task. If None, tasks
            are randomly sampled during each reset.
        container_factory: Factory for creating containers. Defaults to DaytonaContainer.
        code_agent_factory: Factory for creating code agents. Defaults to MiniSWECodeAgent.
        step_limit: Maximum number of steps per episode. Defaults to 100.
        tracker: Optional statistics tracker for monitoring performance.
        **kwargs: Additional arguments passed to SweBenchEnv constructor.

    Returns:
        Configured SweBenchEnv instance.

    Examples:
        >>> env = _make_swebench_lite()  # Random task sampling
        >>> env = _make_swebench_lite(task_index=42)  # Specific task
    """
    all_tasks = _swebench_lite_tasks()

    # If task_index specified, use only that task
    tasks: Sequence[swebench_env.SwebenchTask]
    if task_index is not None:
        if task_index < 0 or task_index >= len(all_tasks):
            raise ValueError(
                f"task_index {task_index} out of range. "
                f"SWE-bench Lite has {len(all_tasks)} tasks (0-{len(all_tasks) - 1})."
            )
        tasks = [all_tasks[task_index]]
        _LOGGER.info("Selected task %d: %s", task_index, tasks[0].instance_id)
    else:
        tasks = all_tasks

    return swebench_env.SweBenchEnv(
        tasks=tasks,
        container_factory=container_factory,
        code_agent_factory=code_agent_factory,
        step_limit=step_limit,
        tracker=tracker,
        **kwargs,
    )


def _make_harbor(
    dataset_name: str,
    dataset_version: str,
    *,
    task_index: int | None = None,
    container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
    code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
    step_limit: int = 100,
    tracker: stat_tracker.StatTracker | None = None,
    **kwargs: Any,
) -> Any:
    """Factory function for Harbor environments.

    Harbor is a framework for creating code agent benchmarks. This factory creates
    environments for any Harbor-compatible dataset.

    Args:
        dataset_name: Name of the Harbor dataset to load (e.g., "easy", "medium", "hard").
        dataset_version: Version of the dataset to load (e.g., "v1", "v2").
        task_index: Optional 0-based index to select a specific task. If None, tasks
            are randomly sampled during each reset.
        container_factory: Factory for creating containers. Defaults to DaytonaContainer.
        code_agent_factory: Factory for creating code agents. Defaults to MiniSWECodeAgent.
        step_limit: Maximum number of steps per episode. Defaults to 100.
        tracker: Optional statistics tracker for monitoring performance.
        **kwargs: Additional arguments passed to HarborEnv constructor.

    Returns:
        Configured HarborEnv instance.

    Examples:
        >>> env = _make_harbor("easy", "v1")  # Random task sampling
        >>> env = _make_harbor("medium", "v2", task_index=10)  # Specific task
    """
    # Lazy import to avoid loading harbor_env unless actually needed
    from ares.environments import harbor_env

    all_tasks = harbor_env.load_harbor_dataset(dataset_name, dataset_version)

    # If task_index specified, use only that task
    tasks: Sequence[Any]
    if task_index is not None:
        if task_index < 0 or task_index >= len(all_tasks):
            raise ValueError(
                f"task_index {task_index} out of range. "
                f"Harbor dataset '{dataset_name}:{dataset_version}' has {len(all_tasks)} tasks "
                f"(0-{len(all_tasks) - 1})."
            )
        tasks = [all_tasks[task_index]]
        _LOGGER.info("Selected task %d: %s", task_index, tasks[0].name)
    else:
        tasks = all_tasks

    return harbor_env.HarborEnv(
        tasks=tasks,
        container_factory=container_factory,
        code_agent_factory=code_agent_factory,
        step_limit=step_limit,
        tracker=tracker,
        **kwargs,
    )


def _register_default_presets() -> None:
    """Register all default ARES environment presets.

    This function is called automatically when the presets module is imported,
    ensuring built-in presets are always available.
    """
    # SWE-bench presets
    registry.register_preset(
        "swebench:verified",
        _make_swebench_verified,
        "SWE-bench Verified dataset (500+ validated instances)",
    )

    registry.register_preset(
        "swebench:lite",
        _make_swebench_lite,
        "SWE-bench Lite dataset (300 curated instances)",
    )

    # Harbor presets
    # Note: We create separate factory functions for each Harbor variant
    # to provide better defaults and documentation

    def _make_harbor_easy(**kwargs: Any) -> Any:
        """Factory for Harbor Easy dataset."""
        return _make_harbor("easy", "v1", **kwargs)

    def _make_harbor_medium(**kwargs: Any) -> Any:
        """Factory for Harbor Medium dataset."""
        return _make_harbor("medium", "v1", **kwargs)

    def _make_harbor_hard(**kwargs: Any) -> Any:
        """Factory for Harbor Hard dataset."""
        return _make_harbor("hard", "v1", **kwargs)

    registry.register_preset(
        "harbor:easy",
        _make_harbor_easy,
        "Harbor Easy dataset (beginner-level tasks)",
    )

    registry.register_preset(
        "harbor:medium",
        _make_harbor_medium,
        "Harbor Medium dataset (intermediate-level tasks)",
    )

    registry.register_preset(
        "harbor:hard",
        _make_harbor_hard,
        "Harbor Hard dataset (advanced-level tasks)",
    )

    _LOGGER.debug("Registered %d default presets", len(registry.list_presets()))


# Auto-register default presets when this module is imported
_register_default_presets()
