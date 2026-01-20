"""
SWE dm_env Environment implementation.
"""

import abc
import asyncio
import atexit
import functools
import logging
import os
import pathlib
import time
from types import TracebackType
from typing import Literal, NamedTuple, Protocol, Self
import uuid

from numpy.typing import NDArray

from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.containers import daytona as ares_daytona
from ares.environments import base
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.llms import queue_mediated_client

_LOGGER = logging.getLogger(__name__)

# Make sure using the correct docker socket
os.environ["DOCKER_HOST"] = "unix:///var/run/docker.sock"


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

    # TODO: adapt the dm_env specs for ARES.
    # def reward_spec(self):
    #     """Describes the reward returned by the environment.

    #     By default this is assumed to be a single float.

    #     Returns:
    #       An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    #     """
    #     return specs.Array(shape=(), dtype=float, name="reward")

    # def discount_spec(self) -> specs.Array:
    #     """Describes the discount returned by the environment.

    #     By default this is assumed to be a single float between 0 and 1.

    #     Returns:
    #       An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    #     """
    #     return specs.BoundedArray(shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount")

    # def observation_spec(self):
    #     """Defines the observations provided by the environment.

    #     May use a subclass of `specs.Array` that specifies additional properties
    #     such as min and max bounds on the values.

    #     Returns:
    #       An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    #     """

    # def action_spec(self):
    #     """Defines the actions that should be provided to `step`.

    #     May use a subclass of `specs.Array` that specifies additional properties
    #     such as min and max bounds on the values.

    #     Returns:
    #       An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    #     """

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


async def create_container(
    *,
    container_factory: containers.ContainerFactory,
    container_prefix: str,
    image_name: str | None = None,
    dockerfile_path: pathlib.Path | str | None = None,
    resources: containers.Resources | None = None,
) -> containers.Container:
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


class Janitor:
    def __init__(self):
        # We use the in-memory ID since the environment isn't hashable.
        self._environment_by_id: dict[int, CodeBaseEnv] = {}
        atexit.register(self._sync_cleanup)

    def register_for_cleanup(self, env: "CodeBaseEnv"):
        self._environment_by_id[id(env)] = env

    def unregister_for_cleanup(self, env: "CodeBaseEnv"):
        del self._environment_by_id[id(env)]

    def _cleanup_environment(self, env: "CodeBaseEnv") -> None:
        if env._container is not None:
            _LOGGER.info("Stopping and removing container %s.", env._container)
            env._container.stop_and_remove()

    def _sync_cleanup(self):
        _LOGGER.info("Cleaning up %d environments iteratively...", len(self._environment_by_id))
        # Copy keys so we can modify the dictionary during iteration.
        keys = list(self._environment_by_id.keys())
        for key in keys:
            env = self._environment_by_id[key]
            self._cleanup_environment(env)
            del self._environment_by_id[key]


# We don't need to do it this way, but it feels slightly more elegent to have a single
# function registered for cleanup than to register an atexit fn for every single environment.
_ENVIRONMENT_JANITOR = Janitor()


class CodeBaseEnv[TaskType](Environment[llm_clients.LLMResponse, llm_clients.LLMRequest | None, float, float]):
    """Base environment for code agents that computes reward at the end of an episode."""

    def __init__(
        self,
        *,
        container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
        code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
        step_limit: int = 100,
        prefix: str = "",
        tracker: stat_tracker.StatTracker | None = None,
    ):
        self._container_factory = container_factory
        self._code_agent_factory = code_agent_factory
        self._step_limit = step_limit
        self._prefix = prefix
        self._tracker = tracker if tracker is not None else stat_tracker.NullStatTracker()

        # We set the LLM client to a queue mediated client so that
        # we can return LLM requests in the reset and step methods.
        # We should never allow a user to pass a different LLM client.
        self._llm_client = queue_mediated_client.QueueMediatedLLMClient(q=asyncio.Queue())
        self._llm_req_future: asyncio.Future[llm_clients.LLMResponse] | None = None

        # State.
        self._is_active = False
        self._container: containers.Container | None = None
        self._current_task: TaskType | None = None
        self._code_agent_task: asyncio.Task[None] | None = None
        self._step_count = 0
        self._is_active = False

        # Register for cleanup on exit.
        _ENVIRONMENT_JANITOR.register_for_cleanup(self)

    async def reset(self) -> base.TimeStep[llm_clients.LLMRequest, float, float]:
        # Require the environment to be used as a context manager.
        reset_start_time = time.time()
        self._assert_active()

        # Ensure the environment is being used as a context manager.
        _LOGGER.debug("[%d] Resetting environment.", id(self))

        self._step_count = 0
        self._requires_reset = False

        if self._container is not None:
            _LOGGER.debug("[%d] Stopping container on reset.", id(self))
            # Stop the container to free resources.
            await self._container.stop()
            self._container = None

        await self._reset_task()
        await self._start_container()
        await self._start_code_agent()

        ts = await self._get_time_step()

        # If the timestep is supposed to be the last, then we have a problem.
        # This can't be both the first and last timestep, so we raise an exception.
        if ts.step_type == "LAST":
            raise RuntimeError("The code agent didn't make any LLM requests.")

        # get_time_step always returns a MID timestep, but we know it's actually a first timestep.
        assert ts.observation is not None
        result = TimeStep(step_type="FIRST", reward=ts.reward, discount=ts.discount, observation=ts.observation)

        reset_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/reset", reset_end_time - reset_start_time)
        return result

    async def step(self, action: llm_clients.LLMResponse) -> base.TimeStep[llm_clients.LLMRequest | None, float, float]:
        # Require the environment to be used as a context manager.
        step_start_time = time.time()
        self._assert_active()

        # Ensure the environment is being used as a context manager.
        _LOGGER.debug("[%d] Stepping environment.", id(self))

        if self._requires_reset:
            # TODO: Custom error.
            raise RuntimeError("Environment must be reset.")

        self._step_count += 1

        _LOGGER.debug("[%d] Setting LLM request future result.", id(self))
        assert self._llm_req_future is not None
        self._llm_req_future.set_result(action)
        _LOGGER.debug("[%d] LLM request future result set.", id(self))

        with self._tracker.timeit(f"{self._prefix}/get_time_step"):
            ts = await self._get_time_step()

        if self._step_count >= self._step_limit:
            _LOGGER.debug("[%d] Step limit reached. Returning LAST timestep.", id(self))
            assert self._code_agent_task is not None
            self._code_agent_task.cancel()
            # Truncation: step_type="LAST", discount=1.0, unless we're _also_ already in a terminal state.
            ts = TimeStep(step_type="LAST", reward=ts.reward, discount=ts.discount, observation=ts.observation)

        if ts.step_type == "LAST":
            self._requires_reset = True

        step_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/step", step_end_time - step_start_time)

        return ts

    async def _get_time_step(
        self,
    ) -> base.TimeStep[llm_clients.LLMRequest | None, float, float]:
        # Wait for the code agent to send another request or complete.
        _LOGGER.debug("[%d] Waiting for code agent or LLM request.", id(self))
        with self._tracker.timeit(f"{self._prefix}/get_from_queue"):
            get_from_queue_task = asyncio.create_task(self._llm_client.q.get())
            tasks = [self._code_agent_task, get_from_queue_task]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        _LOGGER.debug("[%d] Code agent or LLM request completed.", id(self))

        if self._code_agent_task in done:
            _LOGGER.debug("[%d] Code agent completed.", id(self))
            # We're done. Return the final reward.
            assert self._container is not None
            assert self._current_task is not None
            _LOGGER.debug("[%d] Running tests and evaluating.", id(self))
            with self._tracker.timeit(f"{self._prefix}/run_tests_and_evaluate"):
                reward = await self._compute_reward()
            _LOGGER.debug("[%d] Tests and evaluation completed. Reward: %f.", id(self), reward)

            # TODO: Make sure this is correct.
            # Cancel the queue get task.
            get_from_queue_task.cancel()

            return base.TimeStep(step_type="LAST", reward=reward, discount=0.0, observation=None)

        if get_from_queue_task in done:
            with self._tracker.timeit(f"{self._prefix}/get_and_make_timestep"):
                _LOGGER.debug("[%d] LLM request completed.", id(self))
                req_and_future = await get_from_queue_task
                self._llm_req_future = req_and_future.future
                _LOGGER.debug("[%d] LLM request received.", id(self))
                return base.TimeStep(step_type="MID", reward=0.0, discount=1.0, observation=req_and_future.value)

        raise RuntimeError("Code agent task or LLM request future did not complete.")

    async def close(self) -> None:
        # Shut down any resources used by the environment.
        if self._container is not None:
            _LOGGER.debug("[%d] Stopping container on exit.", id(self))
            await self._container.stop()
            self._container = None

    async def __aenter__(self) -> "CodeBaseEnv":
        self._is_active = True
        _ENVIRONMENT_JANITOR.register_for_cleanup(self)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        del exc_type, exc_value, traceback  # Unused.

        self._is_active = False
        await self.close()

        _ENVIRONMENT_JANITOR.unregister_for_cleanup(self)

    def _assert_active(self) -> None:
        if not self._is_active:
            raise RuntimeError("Environment is not active.")

    @abc.abstractmethod
    async def _reset_task(self) -> None:
        """Should set `self._current_task` with a TaskType"""
        pass

    @abc.abstractmethod
    async def _start_container(self) -> None:
        """Should set `self._container` with a running container"""
        pass

    @abc.abstractmethod
    async def _start_code_agent(self) -> None:
        """Should set `self._code_agent_task` with an Asyncio Task"""
        pass

    @abc.abstractmethod
    async def _compute_reward(self) -> float:
        """Runs when the episode has concluded - should return the reward for the episode"""
        pass
