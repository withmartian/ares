"""ARES: Agentic Research and Evaluation Suite.

ARES is an RL-first framework for training and evaluating code agents. It implements
an async version of DeepMind's dm_env specification, treating LLM requests as
observations and LLM responses as actions within a standard RL loop.

The primary way to create environments is via the registry system:

    >>> import ares
    >>> env = ares.make("swebench:lite")
    >>> env = ares.make("swebench:lite:5")  # Select task index 5
    >>> env = ares.make("harbor:easy", step_limit=50)  # Override defaults

To see available presets:

    >>> print(ares.info())

For advanced usage, register custom presets:

    >>> import ares.registry
    >>> ares.registry.register_preset("custom:my-env", my_factory, "My custom environment")

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
