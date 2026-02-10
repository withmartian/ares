"""Episode trajectory collection for ARES environments.

Provides data models and collectors for recording episode trajectories.
Trajectories capture the full sequence of (observation, action, reward, discount)
tuples that flow through the environment loop, enabling:

- Behavior cloning (learning from recorded expert episodes)
- Batch / offline RL (training on collected experience)
- Debugging and analysis (replaying what happened in an episode)

Usage:

    >>> from ares.environments.trajectory import JsonTrajectoryCollector
    >>> collector = JsonTrajectoryCollector(output_dir="./trajectories")
    >>> env = CodeEnvironment(tasks=..., trajectory_collector=collector)

Trajectories are stored as one JSON file per episode. Each file contains episode
metadata (task name, timing, reward, truncation status) and an ordered list of
step records.

Step record semantics follow the dm_env convention:

    - FIRST step (from reset): observation is set; action/reward/discount are None.
    - MID steps (from step): action is the LLMResponse provided, observation is
      the next LLMRequest, reward is the intermediate reward (usually 0.0).
    - LAST step (from step): action is the final LLMResponse, observation may be
      None (terminal), reward is the episode reward.
"""

import dataclasses
import json
import logging
import pathlib
import time
from typing import Any, Protocol, runtime_checkable
import uuid

from ares.environments.base import StepType
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def serialize_llm_request(req: request.LLMRequest) -> dict[str, Any]:
    """Serialize an LLMRequest to a JSON-compatible dict."""
    return dataclasses.asdict(req)


def serialize_llm_response(resp: response.LLMResponse) -> dict[str, Any]:
    """Serialize an LLMResponse to a JSON-compatible dict."""
    return dataclasses.asdict(resp)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """Records the data from a single environment step.

    For the FIRST step (from ``reset()``):
        - ``observation`` is the initial ``LLMRequest`` (serialized).
        - ``action``, ``reward``, and ``discount`` are ``None``.

    For MID steps (from ``step()``):
        - ``action`` is the ``LLMResponse`` that was provided to ``step()``.
        - ``observation`` is the resulting ``LLMRequest``.
        - ``reward`` and ``discount`` are from the returned ``TimeStep``.

    For the LAST step (from ``step()``):
        - ``action`` is the final ``LLMResponse`` provided to ``step()``.
        - ``observation`` may be ``None`` (terminal state).
        - ``reward`` contains the episode reward.
        - ``discount`` is ``0.0`` (terminal) or ``1.0`` (truncated).
    """

    step_index: int
    step_type: StepType
    observation: dict[str, Any] | None
    action: dict[str, Any] | None
    reward: float | None
    discount: float | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepRecord":
        """Reconstruct a StepRecord from a plain dict."""
        return cls(**data)


@dataclasses.dataclass
class EpisodeTrajectory:
    """A complete episode trajectory with metadata and step records.

    Attributes:
        episode_id: Unique identifier for this episode.
        task_name: The name of the task that was run.
        steps: Ordered list of step records.
        start_time: Wall-clock time when the episode started (``time.time()``).
        end_time: Wall-clock time when the episode ended.
        total_reward: The reward from the final step.
        num_steps: Total number of steps recorded.
        truncated: Whether the episode was truncated (step limit reached).
    """

    episode_id: str
    task_name: str
    steps: list[StepRecord]
    start_time: float
    end_time: float | None = None
    total_reward: float | None = None
    num_steps: int = 0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dict for JSON serialization."""
        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "truncated": self.truncated,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EpisodeTrajectory":
        """Reconstruct an EpisodeTrajectory from a plain dict."""
        steps = [StepRecord.from_dict(s) for s in data.get("steps", [])]
        return cls(
            episode_id=data["episode_id"],
            task_name=data["task_name"],
            steps=steps,
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            total_reward=data.get("total_reward"),
            num_steps=data.get("num_steps", len(steps)),
            truncated=data.get("truncated", False),
        )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "EpisodeTrajectory":
        """Load an EpisodeTrajectory from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Collector protocol and implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class TrajectoryCollector(Protocol):
    """Protocol for collecting episode trajectories.

    Implementations receive step-by-step data from the environment loop
    and persist or aggregate it however they choose.

    Lifecycle::

        collector.begin_episode(task_name="my-task")
        collector.record_step(first_step_record)
        collector.record_step(mid_step_record)
        ...
        trajectory = collector.end_episode(truncated=False)
    """

    def begin_episode(self, *, task_name: str) -> None:
        """Signal the start of a new episode."""
        ...

    def record_step(self, record: StepRecord) -> None:
        """Record a single step within the current episode."""
        ...

    def end_episode(self, *, truncated: bool = False) -> EpisodeTrajectory:
        """Finalize the current episode and return the completed trajectory."""
        ...


class JsonTrajectoryCollector:
    """Collects trajectories and persists each episode as a JSON file.

    Files are named ``{episode_id}.json`` and written to *output_dir*.

    Args:
        output_dir: Directory where episode JSON files will be saved.
            Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str | pathlib.Path):
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._current_episode: EpisodeTrajectory | None = None

    @property
    def output_dir(self) -> pathlib.Path:
        """The directory where trajectory files are written."""
        return self._output_dir

    def begin_episode(self, *, task_name: str) -> None:
        """Start recording a new episode.

        If a previous episode was not ended, it is discarded with a warning.
        """
        if self._current_episode is not None:
            _LOGGER.warning(
                "Previous episode %s was not ended — discarding %d steps.",
                self._current_episode.episode_id,
                len(self._current_episode.steps),
            )

        episode_id = str(uuid.uuid4())
        self._current_episode = EpisodeTrajectory(
            episode_id=episode_id,
            task_name=task_name,
            steps=[],
            start_time=time.time(),
        )
        _LOGGER.debug(
            "Started trajectory collection for episode %s (task: %s).",
            episode_id,
            task_name,
        )

    def record_step(self, record: StepRecord) -> None:
        """Append a step record to the current episode."""
        if self._current_episode is None:
            raise RuntimeError("No episode in progress. Call begin_episode() first.")
        self._current_episode.steps.append(record)

    def end_episode(self, *, truncated: bool = False) -> EpisodeTrajectory:
        """Finalize the current episode, write it to disk, and return it."""
        if self._current_episode is None:
            raise RuntimeError("No episode in progress. Call begin_episode() first.")

        episode = self._current_episode
        episode.end_time = time.time()
        episode.truncated = truncated
        episode.num_steps = len(episode.steps)

        # Extract total reward from the last step.
        if episode.steps:
            last_step = episode.steps[-1]
            episode.total_reward = last_step.reward

        # Persist to disk.
        filename = f"{episode.episode_id}.json"
        filepath = self._output_dir / filename
        with open(filepath, "w") as f:
            json.dump(episode.to_dict(), f, indent=2)

        _LOGGER.info(
            "Saved trajectory for episode %s: %d steps, reward=%s, truncated=%s → %s",
            episode.episode_id,
            episode.num_steps,
            episode.total_reward,
            truncated,
            filepath,
        )

        self._current_episode = None
        return episode
