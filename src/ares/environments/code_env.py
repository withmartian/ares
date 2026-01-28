"""Environment for any Harbor-compatible datasets.

This environment requires a different image for each instance.
This environment will use a new container for each instance at every reset.
"""

import asyncio
import atexit
from collections.abc import Sequence
import contextlib
import functools
import json
import logging
import pathlib
import random
import time
from types import TracebackType
from typing import Self

from harbor.models import registry as harbor_registry
from harbor.models.task import task as harbor_task
from harbor.models.trial import paths as harbor_paths
from harbor.registry import client as harbor_dataset_client

from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.containers import daytona as ares_daytona
from ares.environments import base
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.llms import queue_mediated_client
from ares.llms import request

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_harbor_dataset_client() -> harbor_dataset_client.BaseRegistryClient:
    return harbor_dataset_client.RegistryClientFactory.create()


@functools.lru_cache(maxsize=250)
def load_harbor_dataset(name: str, version: str) -> list[harbor_task.Task]:
    client = get_harbor_dataset_client()
    return [
        harbor_task.Task(task_dir=task_item.downloaded_path)
        for task_item in client.download_dataset(name=name, version=version)
    ]


@functools.lru_cache(maxsize=1)
def list_harbor_datasets() -> tuple[harbor_registry.DatasetSpec, ...]:
    client = get_harbor_dataset_client()
    return tuple(client.get_datasets())


class CodeEnvironment(base.Environment[llm_clients.LLMResponse, request.LLMRequest | None, float, float]):
    """Environment for code agent datasets that computes reward at the end of an episode."""

    def __init__(
        self,
        tasks: Sequence[harbor_task.Task],
        *,
        container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
        code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
        step_limit: int = 250,  # Same as MiniSWEAgent default.
        prefix: str = "harbor_env",
        tracker: stat_tracker.StatTracker | None = None,
    ):
        self._tasks = tasks
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
        self._current_task: harbor_task.Task | None = None
        self._code_agent_task: asyncio.Task[None] | None = None
        self._step_count = 0
        self._requires_reset = False

        # Register for cleanup on exit.
        _ENVIRONMENT_JANITOR.register_for_cleanup(self)

    async def reset(self) -> base.TimeStep[request.LLMRequest, float, float]:
        reset_start_time = time.time()
        self._assert_active()

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
        result = base.TimeStep(step_type="FIRST", reward=ts.reward, discount=ts.discount, observation=ts.observation)

        reset_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/reset", reset_end_time - reset_start_time)
        return result

    async def step(self, action: llm_clients.LLMResponse) -> base.TimeStep[request.LLMRequest | None, float, float]:
        step_start_time = time.time()
        self._assert_active()

        _LOGGER.debug("[%d] Stepping environment.", id(self))

        if self._requires_reset:
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
            ts = base.TimeStep(step_type="LAST", reward=ts.reward, discount=ts.discount, observation=ts.observation)

        if ts.step_type == "LAST":
            self._requires_reset = True

        step_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/step", step_end_time - step_start_time)

        return ts

    async def _get_time_step(
        self,
    ) -> base.TimeStep[request.LLMRequest | None, float, float]:
        # Wait for the code agent to send another request or complete.
        _LOGGER.debug("[%d] Waiting for code agent or LLM request.", id(self))
        with self._tracker.timeit(f"{self._prefix}/get_from_queue"):
            get_from_queue_task = asyncio.create_task(self._llm_client.q.get())
            tasks = [self._code_agent_task, get_from_queue_task]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        _LOGGER.debug("[%d] Code agent or LLM request completed.", id(self))

        if self._code_agent_task in done:
            _LOGGER.debug("[%d] Code agent completed.", id(self))

            assert self._code_agent_task is not None
            exc = self._code_agent_task.exception()
            if exc is not None:
                raise RuntimeError("Code agent task failed") from exc

            # We're done. Return the final reward.
            assert self._container is not None
            assert self._current_task is not None
            _LOGGER.debug("[%d] Running tests and evaluating.", id(self))
            with self._tracker.timeit(f"{self._prefix}/run_tests_and_evaluate"):
                reward = await self._compute_reward()
            _LOGGER.debug("[%d] Tests and evaluation completed. Reward: %f.", id(self), reward)

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
        """Shut down any resources used by the environment."""
        if self._code_agent_task is not None and not self._code_agent_task.done():
            self._code_agent_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._code_agent_task
            self._code_agent_task = None

        if self._container is not None:
            _LOGGER.debug("[%d] Stopping container on exit.", id(self))
            await self._container.stop()
            self._container = None

    async def __aenter__(self) -> Self:
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

    async def _reset_task(self) -> None:
        """Randomly select a task from the task list."""
        self._current_task = random.choice(self._tasks)
        _LOGGER.debug("[%d] Selected task %s.", id(self), self._current_task.name)

    async def _start_container(self) -> None:
        """Create and start a container for the current task."""
        if self._current_task is None:
            raise RuntimeError("Task has not been selected before starting the container.")

        _LOGGER.info("[%d] Setting up container for task %s.", id(self), self._current_task.name)
        with self._tracker.timeit("harbor_env/create_container"):
            self._container = await base.create_container(
                container_factory=self._container_factory,
                container_prefix=self._prefix,
                image_name=self._current_task.config.environment.docker_image,  # NOTE: May be None
                # TODO: This is pulled from current Harbor implementation for all
                #       supported environments, track changes in future
                dockerfile_path=self._current_task.paths.environment_dir / "Dockerfile",
                resources=containers.Resources(
                    cpu=self._current_task.config.environment.cpus,
                    memory=self._current_task.config.environment.memory_mb // 1024,
                    disk=self._current_task.config.environment.storage_mb // 1024,
                ),
            )
            await self._container.start()
        _LOGGER.debug("[%d] Container setup complete.", id(self))

    async def _start_code_agent(self) -> None:
        """Start the code agent with the current task's instruction."""
        if self._container is None:
            raise RuntimeError("Container has not been created before starting the code agent.")
        if self._current_task is None:
            raise RuntimeError("Task has not been selected before starting the code agent.")

        _LOGGER.debug("[%d] Starting code agent.", id(self))
        self._code_agent = self._code_agent_factory(container=self._container, llm_client=self._llm_client)
        self._code_agent_task = asyncio.create_task(self._code_agent.run(self._current_task.instruction))
        _LOGGER.debug("[%d] Code agent started.", id(self))

    async def _compute_reward(self) -> float:
        """Run tests and compute the reward for the current episode."""
        if self._container is None:
            raise RuntimeError("Container has not been created before computing reward.")
        if self._current_task is None:
            raise RuntimeError("Task has not been selected before computing reward.")

        _LOGGER.debug("[%d] Uploading tests to container.", id(self))
        await self._container.upload_dir(
            local_path=self._current_task.paths.tests_dir,
            remote_path="/tests",
        )

        _LOGGER.debug("[%d] Creating verifier directory", id(self))
        verifier_dir = str(harbor_paths.EnvironmentPaths.verifier_dir)
        await self._container.exec_run(command=f"mkdir -p {verifier_dir}")

        _LOGGER.debug("[%d] Running tests and evaluating.", id(self))
        test_path = str(
            pathlib.Path("/tests") / self._current_task.paths.test_path.relative_to(self._current_task.paths.tests_dir)
        )
        # TODO: Log the output of the test execution somewhere that makes sense
        test_result = await self._container.exec_run(command=f"bash {test_path}")
        _LOGGER.debug("[%d] Test result: %s.", id(self), test_result.output)

        # Try to read reward from both
        for reward_path in [
            harbor_paths.EnvironmentPaths.reward_text_path,
            harbor_paths.EnvironmentPaths.reward_json_path,
        ]:
            try:
                curr_reward = await self._parse_reward_file(reward_path)
                if curr_reward is not None:
                    _LOGGER.debug("[%d] Reward found: %f.", id(self), curr_reward)
                    return curr_reward
            except ValueError as e:
                # Warn, but still try the other reward path
                _LOGGER.warning("Error parsing reward file %s: %s", reward_path, e)
                pass

        raise ValueError(f"[{id(self)}] No reward found for task {self._current_task.name}")

    async def _parse_reward_file(self, remote_path: pathlib.Path | str) -> float | None:
        """Helper to parse a reward from a text or json file in the container."""
        if self._container is None:
            raise RuntimeError("Container has not been created before parsing reward file.")

        remote_path = str(remote_path)
        cat_result = await self._container.exec_run(command=f"cat {remote_path}")
        if cat_result.exit_code != 0:
            # File doesn't exist
            return None

        file_contents = cat_result.output.strip()

        if remote_path.endswith(".txt"):
            return float(file_contents)

        elif remote_path.endswith(".json"):
            rewards_dict = json.loads(file_contents)

            # TODO: Possibly support multiple rewards?
            reward_keys = list(rewards_dict.keys())
            if len(reward_keys) != 1:
                raise ValueError(f"Expected 1 reward key, got {len(reward_keys)}")
            return float(rewards_dict[reward_keys[0]])

        else:
            raise ValueError(f"Unsupported reward file type: {remote_path}")


class _Janitor:
    """Emergency cleanup handler for environments.

    Ensures containers are cleaned up even on abnormal termination.
    """

    def __init__(self):
        # We use the in-memory ID since the environment isn't hashable.
        self._environment_by_id: dict[int, CodeEnvironment] = {}
        atexit.register(self._sync_cleanup)

    def register_for_cleanup(self, env: CodeEnvironment):
        """Register an environment for emergency cleanup."""
        self._environment_by_id[id(env)] = env

    def unregister_for_cleanup(self, env: CodeEnvironment):
        """Unregister an environment from emergency cleanup."""
        del self._environment_by_id[id(env)]

    def _cleanup_environment(self, env: CodeEnvironment) -> None:
        """Clean up a single environment's container."""
        # Access the _container attribute if it exists
        container = getattr(env, "_container", None)
        if container is not None:
            _LOGGER.info("Stopping and removing container %s.", container)
            container.stop_and_remove()

    def _sync_cleanup(self):
        """Cleanup all registered environments at exit."""
        if self._environment_by_id:
            _LOGGER.info("Cleaning up %d environments iteratively...", len(self._environment_by_id))
        # Copy keys so we can modify the dictionary during iteration.
        keys = list(self._environment_by_id.keys())
        for key in keys:
            env = self._environment_by_id[key]
            self._cleanup_environment(env)
            del self._environment_by_id[key]


# We don't need to do it this way, but it feels slightly more elegant to have a single
# function registered for cleanup than to register an atexit fn for every single environment.
_ENVIRONMENT_JANITOR = _Janitor()
