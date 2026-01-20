"""Environment for any Harbor-compatible datasets.

This environment requires a different image for each instance.
This environment will use a new container for each instance at every reset.
"""

import asyncio
from collections.abc import Sequence
import functools
import json
import logging
import pathlib
import random

from harbor.models.task import task as harbor_task
from harbor.models.trial import paths as harbor_paths
from harbor.registry import client as harbor_dataset_client

from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.containers import daytona as ares_daytona
from ares.environments import base
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_harbor_dataset_client() -> harbor_dataset_client.RegistryClient:
    return harbor_dataset_client.RegistryClient()


def load_harbor_dataset(name: str, version: str) -> list[harbor_task.Task]:
    client = _get_harbor_dataset_client()
    return [
        harbor_task.Task(task_dir=task_item.downloaded_path)
        for task_item in client.download_dataset(name=name, version=version)
    ]


class HarborEnv(base.CodeBaseEnv[harbor_task.Task]):
    def __init__(
        self,
        tasks: Sequence[harbor_task.Task],
        *,
        container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
        code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
        step_limit: int = 100,
        prefix: str = "harbor_env",
        tracker: stat_tracker.StatTracker | None = None,
    ):
        super().__init__(
            container_factory=container_factory,
            code_agent_factory=code_agent_factory,
            step_limit=step_limit,
            prefix=prefix,
            tracker=tracker,
        )
        self._tasks = tasks

    async def _reset_task(self) -> None:
        self._current_task = random.choice(self._tasks)
        _LOGGER.debug("[%d] Selected task %s.", id(self), self._current_task.name)

    async def _start_container(self) -> None:
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
        _LOGGER.debug("[%d] Starting code agent.", id(self))
        self._code_agent = self._code_agent_factory(container=self._container, llm_client=self._llm_client)
        self._code_agent_task = asyncio.create_task(self._code_agent.run(self._current_task.instruction))
        _LOGGER.debug("[%d] Code agent started.", id(self))

    async def _compute_reward(self) -> float:
        _LOGGER.debug("[%d] Uploading tests to container.", id(self))
        await self._container.upload_dir(
            local_path=self._current_task.paths.tests_dir,
            remote_path="/tests",
        )

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
