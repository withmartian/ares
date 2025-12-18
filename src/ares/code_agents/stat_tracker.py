"""Statistic tracking objects."""

import asyncio
import collections
from collections.abc import Generator
import contextlib
import dataclasses
import logging
import time
from typing import Protocol

import numpy as np
from torch.utils import tensorboard

_LOGGER = logging.getLogger(__name__)


class StatTracker(Protocol):
    @contextlib.contextmanager
    def timeit(self, name: str) -> Generator: ...

    def scalar(self, name: str, value: float) -> None: ...


class NullStatTracker:
    @contextlib.contextmanager
    def timeit(self, name: str) -> Generator:
        del name  # Unused.
        yield

    def scalar(self, name: str, value: float) -> None:
        del name, value  # Unused.


class LoggingStatTracker(StatTracker):
    def __init__(self):
        self._period_seconds = 60
        self._stats: collections.defaultdict[str, list[float]] = collections.defaultdict(list)
        self._task = asyncio.create_task(self._track())

    async def _track(self) -> None:
        while True:
            await asyncio.sleep(self._period_seconds)

            # Sort the keys to ensure consistent ordering.
            sorted_keys = sorted(self._stats.keys())

            for k in sorted_keys:
                v = self._stats[k]
                if not v:
                    continue

                _LOGGER.info("%s: %s", k, np.percentile(np.array(v), [0, 25, 50, 75, 100]))
            self._stats.clear()

    @contextlib.contextmanager
    def timeit(self, name: str) -> Generator:
        start_time = time.time()
        yield
        end_time = time.time()
        self._stats[name].append(end_time - start_time)

    def scalar(self, name: str, value: float) -> None:
        self._stats[name].append(value)


@dataclasses.dataclass
class TensorboardStatTracker(StatTracker):
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
