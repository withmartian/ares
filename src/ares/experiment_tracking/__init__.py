"""Experiment tracking and statistics."""

from ares.experiment_tracking.stat_tracker import LoggingStatTracker
from ares.experiment_tracking.stat_tracker import NullStatTracker
from ares.experiment_tracking.stat_tracker import StatTracker

__all__ = [
    "LoggingStatTracker",
    "NullStatTracker",
    "StatTracker",
]

# TensorboardStatTracker is optionally available if torch is installed
try:
    from ares.experiment_tracking.tensorboard import TensorboardStatTracker  # noqa: F401

    __all__.append("TensorboardStatTracker")
except ImportError:
    pass
