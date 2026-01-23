"""Hook utilities for mechanistic interpretability interventions."""

import dataclasses

from ares.containers import containers
from ares.environments.base import TimeStep


@dataclasses.dataclass
class FullyObservableState:
    timestep: TimeStep | None
    container: containers.Container | None
    step_num: int
