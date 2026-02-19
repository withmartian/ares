"""Mechanistic interpretability utilities for ARES.

This module provides tools for analyzing agent behavior using mechanistic interpretability
techniques, with deep integration with TransformerLens for studying model internals across
long-horizon agent trajectories.
"""

from ares.contrib.mech_interp.activation_capture import ActivationCapture
from ares.contrib.mech_interp.activation_capture import TrajectoryActivations
from ares.contrib.mech_interp.hooked_transformer_client import HookedTransformerLLMClient
from ares.contrib.mech_interp.hooked_transformer_client import create_hooked_transformer_client_with_chat_template

__all__ = [
    "ActivationCapture",
    "HookedTransformerLLMClient",
    "TrajectoryActivations",
    "create_hooked_transformer_client_with_chat_template",
]
