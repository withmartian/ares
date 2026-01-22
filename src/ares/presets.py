"""Default preset registrations for ARES environments.

This module registers the built-in environment presets with the global registry.
It is imported automatically when ARES is imported, ensuring default presets are
always available.

Do not add the registry mechanism itself to this module - that belongs in registry.py.
This module only contains preset registrations to avoid circular imports.
"""

import dataclasses
import logging

from ares import registry
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.environments import base
from ares.environments import swebench_env
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SwebenchVerifiedMiniSWESpec:
    """Environment spec for SWE-bench Verified with mini-swe-agent."""

    def get_info(self) -> registry.EnvironmentInfo:
        """Return metadata about SWE-bench Verified."""
        return registry.EnvironmentInfo(
            name="sbv-mswea",
            description="SWE-bench Verified with mini-swe-agent",
            num_tasks=500,
        )

    def get_env(
        self,
        *,
        container_factory: containers.ContainerFactory,
        tracker: stat_tracker.StatTracker | None = None,
    ) -> base.Environment:
        """Create SWE-bench Verified environment with mini-swe-agent."""
        tasks = swebench_env.swebench_verified_tasks()
        return swebench_env.SweBenchEnv(
            tasks=tasks,
            container_factory=container_factory,
            code_agent_factory=mini_swe_agent.MiniSWECodeAgent,
            step_limit=100,
            tracker=tracker,
        )


def _register_default_presets() -> None:
    """Register all default ARES environment presets.

    This function is called automatically when the presets module is imported,
    ensuring built-in presets are always available.
    """
    registry.register_preset("sbv-mswea", SwebenchVerifiedMiniSWESpec())
    _LOGGER.debug("Registered %d default presets", len(registry._list_presets()))


# Auto-register default presets when this module is imported
_register_default_presets()
