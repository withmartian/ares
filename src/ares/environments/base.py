"""
dm_env Environment protocol and utilities for ARES.
"""

import atexit
from collections.abc import Awaitable, Callable
import functools
import logging
import os
import pathlib
import time
from types import TracebackType
from typing import Literal, NamedTuple, Protocol, Self
import uuid

from numpy.typing import NDArray

from ares.containers import containers

_LOGGER = logging.getLogger(__name__)

# Make sure using the correct docker socket
# NOTE: Don't override user configuration (e.g. Colima/Docker Desktop on macOS).
# Default to the standard Linux socket only if nothing is set.
os.environ.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")


StepType = Literal["FIRST", "MID", "LAST"]
NestedScalar = dict[str, "Scalar"] | list["Scalar"] | tuple["Scalar", ...]
Scalar = float | NDArray | NestedScalar


class TimeStep[ObservationType, RewardType: Scalar, DiscountType: Scalar](NamedTuple):
    """Returned with every call to `step` and `reset` on an environment.

    A `TimeStep` contains the data emitted by an environment at each step of
    interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
    NumPy array or a dict or list of arrays), and an associated `reward` and
    `discount`.

    The first `TimeStep` in a sequence will have `FIRST` step type. The final
    `TimeStep` will have `LAST` step type. All other `TimeStep`s in a sequence will
    have `MID` step type.

    Attributes:
      step_type: A `StepType` literal value, FIRST, MID, or LAST.
      reward:  A scalar, NumPy array, nested dict, list or tuple of rewards; or
        `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
        sequence.
      discount: A scalar, NumPy array, nested dict, list or tuple of discount
        values in the range `[0, 1]`, or `None` if `step_type` is
        `StepType.FIRST`, i.e. at the start of a sequence.
      observation: A NumPy array, or a nested dict, list or tuple of arrays.
        Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
        also valid in place of a scalar array.
    """

    step_type: StepType
    reward: RewardType | None
    discount: DiscountType | None
    observation: ObservationType

    def first(self) -> bool:
        return self.step_type == "FIRST"

    def mid(self) -> bool:
        return self.step_type == "MID"

    def last(self) -> bool:
        return self.step_type == "LAST"


class Environment[ActionType, ObservationType, RewardType: Scalar, DiscountType: Scalar](Protocol):
    async def reset(self) -> TimeStep[ObservationType, RewardType, DiscountType]:
        """Starts a new sequence and returns the first `TimeStep` of this sequence.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """
        ...

    async def step(self, action: ActionType) -> TimeStep[ObservationType, RewardType, DiscountType]:
        """Updates the environment according to the action and returns a `TimeStep`.

        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.

        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.

        Args:
          action: A NumPy array, or a nested dict, list or tuple of arrays
            corresponding to `action_spec()`.

        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` value.
            reward: Reward at this timestep, or None if step_type is
              `StepType.FIRST`. Must conform to the specification returned by
              `reward_spec()`.
            discount: A discount in the range [0, 1], or None if step_type is
              `StepType.FIRST`. Must conform to the specification returned by
              `discount_spec()`.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """
        ...

    async def close(self) -> None:
        """Frees any resources used by the environment.

        Implement this method for an environment backed by an external process.

        This method can be used directly

        ```python
        env = Env(...)
        # Use env.
        env.close()
        ```

        or via a context manager

        ```python
        with Env(...) as env:
          # Use env.
        ```
        """
        pass

    async def __aenter__(self) -> Self:
        """Allows the environment to be used in an async with-statement context."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ):
        """Allows the environment to be used in an async with-statement context."""
        del exc_type, exc_value, traceback  # Unused.
        await self.close()


class EnvironmentCheckpoint[EnvType]:
    """A restorable snapshot of environment state at a step boundary.

    Generic over the environment type so that restore() returns the concrete type.
    The checkpoint is self-contained: it holds closures for restore and release
    that capture whatever data and factories are needed. The environment that
    creates it decides how those closures work.

    restore() creates a brand new environment instance in the checkpointed state.
    The original environment is not affected. Multiple restores from the same
    checkpoint create independent environments.

    Usage::

        checkpoint = await env.checkpoint()
        env_restored, ts = await checkpoint.restore()
        # env_restored is a new, independent environment at the checkpointed state

    Resource management:
        Each checkpoint may hold expensive resources (e.g., a Docker image from
        ``docker commit``). Call release() when no longer needed, or rely on the
        global checkpoint janitor for atexit cleanup.
    """

    def __init__(
        self,
        *,
        restore_fn: Callable[[], Awaitable[EnvType]],
        release_fn: Callable[[], Awaitable[None]] | None = None,
        release_fn_sync: Callable[[], None] | None = None,
        step_count: int,
        timestep: TimeStep,
    ):
        self._restore_fn = restore_fn
        self._release_fn = release_fn
        self._release_fn_sync = release_fn_sync
        self.step_count = step_count
        self.timestep = timestep
        self._released = False

        _CHECKPOINT_JANITOR.register(self)

    async def restore(self) -> tuple[EnvType, TimeStep]:
        """Create a new environment at this checkpoint's state.

        Constructs a brand new environment instance initialised to the
        checkpointed state. The original environment is not touched.
        The caller owns the returned environment and must close() it.

        Can be called multiple times -- each call creates an independent
        environment.

        Returns:
            A tuple of (environment, timestep) where the environment is a new
            instance and the timestep is the observation at the checkpointed state.
        """
        if self._released:
            raise RuntimeError("Cannot restore from a released checkpoint.")
        env = await self._restore_fn()
        return env, self.timestep

    async def release(self) -> None:
        """Release resources held by this checkpoint.

        For environments backed by containers, this deletes the snapshot image.
        For simple environments, this is a no-op. Safe to call multiple times.
        After release(), restore() will raise RuntimeError.
        """
        if self._released:
            return
        self._released = True
        if self._release_fn is not None:
            await self._release_fn()
        _CHECKPOINT_JANITOR.unregister(self)

    @property
    def is_released(self) -> bool:
        """Whether this checkpoint's resources have been released."""
        return self._released


class CheckpointableEnvironment[ActionType, ObservationType, RewardType: Scalar, DiscountType: Scalar](
    Environment[ActionType, ObservationType, RewardType, DiscountType],
    Protocol,
):
    """An Environment that supports checkpointing for exploration algorithms.

    Extends the base Environment protocol with a single method: checkpoint().
    Everything else (restore, release, cleanup) lives on the returned
    EnvironmentCheckpoint object.

    The implementation constructs the checkpoint with closures that know how
    to create a brand new environment at the checkpointed state. No internal
    protocol methods are imposed -- the implementation is free.
    """

    async def checkpoint(self) -> EnvironmentCheckpoint[Self]:
        """Capture the current environment state.

        Must be called at a step boundary (after reset() or step() returned
        a non-LAST timestep, before the next step() call).

        Returns:
            An EnvironmentCheckpoint whose restore() creates a new instance
            of this concrete environment type.
        """
        ...


class _CheckpointJanitor:
    """Emergency cleanup for checkpoints that weren't explicitly released.

    Mirrors the _Janitor pattern in code_env.py for container cleanup.
    Registered checkpoints are cleaned up synchronously at interpreter exit.
    """

    def __init__(self):
        self._checkpoints: dict[int, EnvironmentCheckpoint] = {}
        atexit.register(self._cleanup)

    def register(self, checkpoint: EnvironmentCheckpoint) -> None:
        """Register a checkpoint for emergency cleanup."""
        self._checkpoints[id(checkpoint)] = checkpoint

    def unregister(self, checkpoint: EnvironmentCheckpoint) -> None:
        """Unregister a checkpoint from emergency cleanup."""
        self._checkpoints.pop(id(checkpoint), None)

    def _cleanup(self) -> None:
        """Clean up all registered checkpoints at exit."""
        if self._checkpoints:
            _LOGGER.info("Cleaning up %d unreleased checkpoints at exit.", len(self._checkpoints))
        for cp in list(self._checkpoints.values()):
            if cp._release_fn_sync is not None:
                try:
                    cp._release_fn_sync()
                except Exception:
                    _LOGGER.warning("Failed to clean up checkpoint.", exc_info=True)
        self._checkpoints.clear()


_CHECKPOINT_JANITOR = _CheckpointJanitor()


async def create_container(
    *,
    container_factory: containers.ContainerFactory,
    container_prefix: str,
    image_name: str | None = None,
    dockerfile_path: pathlib.Path | str | None = None,
    resources: containers.Resources | None = None,
) -> containers.Container:
    """Create a container from an image or Dockerfile.

    Args:
        container_factory: Factory for creating containers.
        container_prefix: Prefix for the container name.
        image_name: Optional image name to use.
        dockerfile_path: Optional path to Dockerfile.
        resources: Optional resource constraints.

    Returns:
        A created container (not yet started).

    Raises:
        ValueError: If neither image_name nor dockerfile_path is specified.
    """
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity

    if image_name is not None:
        image_name_short = image_name.split("/")[-1].removesuffix(":latest").split(".")[-1]
        create_fn = functools.partial(container_factory.from_image, image=image_name)

    elif dockerfile_path is not None:
        image_name_short = pathlib.Path(dockerfile_path).parent.name.split(".")[-1]
        create_fn = functools.partial(container_factory.from_dockerfile, dockerfile_path=dockerfile_path)

    else:
        raise ValueError("Must specify one of image_name or dockerfile_path")

    container_name = f"ares.{container_prefix}.{image_name_short}.{timestamp}.{unique_id}"
    container = create_fn(name=container_name, resources=resources)

    return container
