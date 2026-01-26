"""Mechanistic interpretability utilities for ARES.

This module provides tools for analyzing agent behavior using mechanistic interpretability
techniques, with deep integration with TransformerLens for studying model internals across
long-horizon agent trajectories.
"""

from ares.contrib.mech_interp.activation_capture import ActivationCapture
from ares.contrib.mech_interp.activation_capture import TrajectoryActivations
from ares.contrib.mech_interp.hook_utils import InterventionManager
from ares.contrib.mech_interp.hook_utils import create_path_patching_hook
from ares.contrib.mech_interp.hook_utils import create_zero_ablation_hook
from ares.contrib.mech_interp.hooked_transformer_client import HookedTransformerLLMClient

__all__ = [
    "ActivationCapture",
    "HookedTransformerLLMClient",
    "InterventionManager",
    "TrajectoryActivations",
    "create_path_patching_hook",
    "create_zero_ablation_hook",
]
