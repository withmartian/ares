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
from ares.containers import containers
from ares.environments import base
from ares.environments import code_env
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

    dataset_name: str
    dataset_version: str
    dataset_id: str
    code_agent_factory: code_agent_base.CodeAgentFactory
    code_agent_id: str

    @functools.cached_property
    def ds(self) -> list[harbor_task.Task]:
        return code_env.load_harbor_dataset(name=self.dataset_name, version=self.dataset_version)

    @functools.cached_property
    def ds_spec(self) -> harbor_registry.DatasetSpec:
        ds_client = code_env.get_harbor_dataset_client()
        return ds_client.get_dataset_spec(name=self.dataset_name, version=self.dataset_version)

    def get_info(self) -> registry.EnvironmentInfo:
        """Return metadata about Harbor Verified."""
        return registry.EnvironmentInfo(
            name=f"{self.dataset_id}-{self.code_agent_id}",
            description=f"{self.dataset_name} (through Harbor registry) with {self.code_agent_id}",
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
            step_limit=250,  # Same as MiniSWEAgent default.
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
        ]:
            ds_id = _make_harbor_dataset_id(ds_spec.name, ds_spec.version)
            registry.register_preset(
                f"{ds_id}-{code_agent_id}",
                HarborSpec(
                    dataset_name=ds_spec.name,
                    dataset_version=ds_spec.version,
                    dataset_id=ds_id,
                    code_agent_factory=code_agent_factory,
                    code_agent_id=code_agent_id,
                ),
            )

    _LOGGER.debug("Registered %d default presets", len(registry._list_presets()))


# Auto-register default presets when this module is imported
_register_default_presets()
