"""ARES: Agentic Research and Evaluation Suite.

ARES is an RL-first framework for training and evaluating code agents. It implements
an async version of DeepMind's dm_env specification, treating LLM requests as
observations and LLM responses as actions within a standard RL loop.

The primary way to create environments is via the registry system:

    >>> import ares
    >>> env = ares.make("sbv-mswea")  # SWE-bench Verified with mini-swe-agent

Override container factory or add tracking:

    >>> from ares.containers import daytona
    >>> from ares.experiment_tracking import stat_tracker
    >>> tracker = stat_tracker.LoggingStatTracker()
    >>> env = ares.make("sbv-mswea", container_factory=daytona.DaytonaContainer, tracker=tracker)

To see available presets:

    >>> print(ares.info())  # List all presets
    >>> print(ares.info("sbv-mswea"))  # Get info about a specific preset

For advanced usage, register custom presets:

    >>> import ares.registry
    >>> class MyEnvSpec:
    ...     def get_info(self):
    ...         return ares.registry.EnvironmentInfo("my-env", "My custom environment", num_tasks=100)
    ...     def get_env(self, *, container_factory, tracker=None):
    ...         return MyEnvironment(...)
    >>> ares.registry.register_preset("my-env", MyEnvSpec())

All other functionality is available via submodules:
- ares.environments: Environment implementations
- ares.code_agents: Code agent implementations
- ares.containers: Container management
- ares.llms: LLM client implementations
- ares.registry: Registry functions for advanced use
"""

# Import presets first to register defaults
# This must come before we expose make and info to ensure presets are available
from ares import presets  # noqa: F401

# Import registry functions to expose at top level
from ares.registry import info
from ares.registry import make

# Define public API
__all__ = [
    "info",
    "make",
]
