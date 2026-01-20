"""TensorBoard stat tracking implementation."""

import asyncio
import collections
from collections.abc import Generator
import contextlib
import dataclasses
import time

import numpy as np
from torch.utils import tensorboard


@dataclasses.dataclass
class TensorboardStatTracker:
    metric_writer: tensorboard.SummaryWriter
    period_seconds: float = 60

    def __post_init__(self) -> None:
        self._stats: collections.defaultdict[str, list[float]] = collections.defaultdict(list)
        self._task = asyncio.create_task(self._track())

    async def _track(self) -> None:
        step = 0
        while True:
            step += 1
            await asyncio.sleep(self.period_seconds)

            # We don't need to sort the keys for this stat tracker.
            for k, v in self._stats.items():
                self.metric_writer.add_histogram(k, np.asarray(v), global_step=step)
            self._stats.clear()

    @contextlib.contextmanager
    def timeit(self, name: str) -> Generator:
        start_time = time.time()
        yield
        end_time = time.time()
        self._stats[name].append(end_time - start_time)

    def scalar(self, name: str, value: float) -> None:
        self._stats[name].append(value)
