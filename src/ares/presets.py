"""Default preset registrations for ARES environments.

This module registers the built-in environment presets with the global registry.
It is imported automatically when ARES is imported, ensuring default presets are
always available.

Do not add the registry mechanism itself to this module - that belongs in registry.py.
This module only contains preset registrations to avoid circular imports.
"""

import dataclasses
import functools
import logging

from harbor.models import registry as harbor_registry
from harbor.models.task import task as harbor_task

from ares import registry
from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.code_agents.terminus2 import terminus2_agent
from ares.containers import containers
from ares.environments import base
from ares.environments import code_env
from ares.environments import twenty_questions
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


def _make_harbor_dataset_id(name: str, version: str) -> str:
    """Small helper to make a shorter Dataset ID from the Harbor name and version"""
    if name == "swe-lancer-diamond":
        name = f"swe-lancer-diamond-{version}"

    return name.replace("swebench-verified", "sbv").replace("terminal-bench", "tbench")


@dataclasses.dataclass(frozen=True)
class HarborSpec:
    """Environment spec for Harbor Verified with mini-swe-agent."""

    ds_spec: harbor_registry.DatasetSpec
    dataset_id: str
    code_agent_factory: code_agent_base.CodeAgentFactory
    code_agent_id: str

    @functools.cached_property
    def ds(self) -> list[harbor_task.Task]:
        return code_env.load_harbor_dataset(name=self.ds_spec.name, version=self.ds_spec.version)

    def get_info(self) -> registry.EnvironmentInfo:
        """Return metadata about Harbor Verified."""
        return registry.EnvironmentInfo(
            name=f"{self.dataset_id}-{self.code_agent_id}",
            description=(
                f"{self.ds_spec.name}@{self.ds_spec.version} (through Harbor registry) with {self.code_agent_id}"
            ),
            num_tasks=len(self.ds_spec.tasks),
        )

    def get_env(
        self,
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> base.Environment:
        """Create Harbor Verified environment with mini-swe-agent."""
        all_tasks = self.ds
        selected_tasks = selector(all_tasks)

        if not selected_tasks:
            raise ValueError("Task selector produced no tasks.")

        return code_env.CodeEnvironment(
            tasks=selected_tasks,
            container_factory=container_factory,
            code_agent_factory=self.code_agent_factory,
            step_limit=250,  # Same as mini-swe-agent default.
            tracker=tracker,
        )


@dataclasses.dataclass(frozen=True)
class TwentyQuestionsSpec:
    """Environment spec for Twenty Questions game."""

    objects: tuple[str, ...] | None = None
    oracle_model: str = "gpt-4o-mini"
    step_limit: int = 20

    def get_info(self) -> registry.EnvironmentInfo:
        """Return metadata about Twenty Questions environment."""
        num_objects = len(self.objects) if self.objects is not None else len(twenty_questions.DEFAULT_OBJECT_LIST)
        description = f"Twenty Questions game with {num_objects} objects using {self.oracle_model} oracle"
        return registry.EnvironmentInfo(
            name="20q" if self.objects is None else "20q-custom",
            description=description,
            num_tasks=num_objects,
        )

    def get_env(
        self,
        *,
        selector: registry.TaskSelector,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> base.Environment:
        """Create Twenty Questions environment.

        Note: Twenty Questions doesn't use containers or task selection.
        The selector and container_factory parameters are ignored.
        """
        # Warn if a non-trivial selector is provided
        if not isinstance(selector, registry.SliceSelector) or selector.start is not None or selector.end is not None:
            _LOGGER.warning(
                "Twenty Questions environment does not use task selection. "
                "Selector %s will be ignored. Objects are randomly selected each episode.",
                selector,
            )

        del selector, container_factory  # Unused - Twenty Questions doesn't need containers
        return twenty_questions.TwentyQuestionsEnvironment(
            objects=self.objects,
            oracle_model=self.oracle_model,
            step_limit=self.step_limit,
            tracker=tracker,
        )


def _register_default_presets() -> None:
    """Register all default ARES environment presets.

    This function is called automatically when the presets module is imported,
    ensuring built-in presets are always available.
    """
    for ds_spec in code_env.list_harbor_datasets():
        for code_agent_id, code_agent_factory in [
            ("mswea", mini_swe_agent.MiniSWECodeAgent),
            ("terminus2", terminus2_agent.Terminus2Agent),
        ]:
            ds_id = _make_harbor_dataset_id(ds_spec.name, ds_spec.version)
            registry.register_preset(
                f"{ds_id}-{code_agent_id}",
                HarborSpec(
                    ds_spec=ds_spec,
                    dataset_id=ds_id,
                    code_agent_factory=code_agent_factory,
                    code_agent_id=code_agent_id,
                ),
            )

    # Register Twenty Questions preset
    registry.register_preset(
        "20q",
        TwentyQuestionsSpec(
            objects=None,  # Use default full object list
            oracle_model="gpt-4o-mini",
            step_limit=20,
        ),
    )

    _LOGGER.debug("Registered %d default presets", len(registry._list_presets()))


# Auto-register default presets when this module is imported
_register_default_presets()
