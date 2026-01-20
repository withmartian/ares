"""
Deepmind Environment for the SWE-Bench dataset.

The SWE-Bench dataset requires a different image for each instance unlike the SWE-Smith dataset.
This environment will use a new container for each instance at every reset.
"""

import asyncio
from collections.abc import Sequence
import functools
import json
import logging
import random
import time
from typing import Any, cast

import datasets
import pydantic
import swebench.harness.constants
from swebench.harness.constants import FAIL_TO_PASS
from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.constants import PASS_TO_PASS
from swebench.harness.constants import ResolvedStatus
from swebench.harness.grading import get_eval_tests_report
from swebench.harness.grading import get_resolution_status
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.test_spec import test_spec
import tqdm

from ares.code_agents import code_agent_base
from ares.code_agents import mini_swe_agent
from ares.containers import containers
from ares.containers import daytona as ares_daytona
from ares.environments import base
from ares.environments import utils
from ares.experiment_tracking import stat_tracker

_LOGGER = logging.getLogger(__name__)


# TODO: Make consistent throughout SweBench or Swebench or SWEBench.
class SwebenchTask(pydantic.BaseModel):
    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str  # TODO: Convert to datetime.
    version: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    environment_setup_commit: str

    @pydantic.field_validator("FAIL_TO_PASS", mode="before")
    @classmethod
    def _validate_fail_to_pass(cls, v: str) -> list[str]:
        return json.loads(v)

    @pydantic.field_validator("PASS_TO_PASS", mode="before")
    @classmethod
    def _validate_pass_to_pass(cls, v: str) -> list[str]:
        return json.loads(v)

    def to_swebench_typeddict(self) -> swebench.harness.constants.SWEbenchInstance:
        return cast(swebench.harness.constants.SWEbenchInstance, self.model_dump())


@functools.lru_cache
def swebench_verified_tasks() -> tuple[SwebenchTask, ...]:
    ds = datasets.load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    result: list[SwebenchTask] = []
    for task in tqdm.tqdm(ds, desc="Loading SWE-bench tasks"):
        result.append(SwebenchTask.model_validate(task))
    return tuple(result)


async def _reset_test_files(container: containers.Container, ts: test_spec.TestSpec) -> None:
    """Reset the test files to the original state.

    Args:
        container: The container to reset the test files in.
        ts: The test specification.
    """
    # TODO: Implement this.
    # test_files = ts.PASS_TO_PASS + ts.FAIL_TO_PASS
    # test_files_str = " ".join(test_files)
    # test_reset_script = f"git checkout {ts.instance_id} -- {test_files_str}"
    # result = await container.exec_run(test_reset_script, workdir=DOCKER_WORKDIR)
    # assert result.exit_code == 0


async def _execute_tests(
    container: containers.Container,
    ts: test_spec.TestSpec,
) -> tuple[str, float]:
    """Execute tests using the eval script from test_spec.

    Returns:
        Tuple[str, float]: Test output string and runtime in seconds.
    """

    eval_script_content = ts.eval_script

    await utils.write_content_to_file_in_container(container, eval_script_content, "/eval.sh")
    assert (await container.exec_run("chmod +x /eval.sh")).exit_code == 0

    start_time = time.time()
    _LOGGER.debug("[%d] Executing tests.", id(container))
    test_output = await container.exec_run("/eval.sh")
    runtime = time.time() - start_time

    _LOGGER.info("[%d] Test execution completed in %.2fs", id(container), runtime)

    return test_output.output, runtime


def _parse_test_results(task: SwebenchTask, test_output: str) -> tuple[bool, str]:
    """Parse test output and determine if the task is resolved.

    Args:
        test_output: Raw output from test execution.

    Returns:
        Tuple[bool, str]: (resolved, resolution_status) where resolved is True if all
            required tests pass, and resolution_status is a string label.
    """
    # Get the log parser for this repo

    repo = task.repo
    version = task.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    test_status_map = log_parser(test_output, specs)

    # Prepare instance with parsed JSON fields for get_eval_tests_report
    eval_instance = task.model_dump()
    # Parse JSON strings to lists if needed
    if isinstance(eval_instance.get(FAIL_TO_PASS), str):
        eval_instance[FAIL_TO_PASS] = json.loads(eval_instance[FAIL_TO_PASS])
    if isinstance(eval_instance.get(PASS_TO_PASS), str):
        eval_instance[PASS_TO_PASS] = json.loads(eval_instance[PASS_TO_PASS])

    report = get_eval_tests_report(test_status_map, eval_instance)
    resolved = get_resolution_status(report) == ResolvedStatus.FULL.value
    resolution_status = ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value

    return resolved, resolution_status


async def _run_tests_and_evaluate(
    container: containers.Container, task: SwebenchTask, ts: test_spec.TestSpec
) -> dict[str, Any]:
    """Run the evaluation for the current instance and container.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - resolved (bool): Whether all required tests are satisfied.
            - resolution_status (str): Status label.
            - test_runtime (float): Seconds taken to run tests.
            - test_output (str): Raw test execution output.
            - error (str, optional): Error string on failure.
    """
    await _reset_test_files(container, ts)
    test_output, total_runtime = await _execute_tests(container, ts)
    resolved, status = _parse_test_results(task, test_output)

    out = {
        "resolved": resolved,
        "resolution_status": status,
        "test_runtime": total_runtime,
        "test_output": test_output,
    }
    return out


class SweBenchEnv(base.CodeBaseEnv[SwebenchTask]):
    def __init__(
        self,
        # TODO: Decide if we want to include the task sampling logic in the base class
        tasks: Sequence[SwebenchTask],
        *,
        container_factory: containers.ContainerFactory = ares_daytona.DaytonaContainer,
        code_agent_factory: code_agent_base.CodeAgentFactory = mini_swe_agent.MiniSWECodeAgent,
        step_limit: int = 100,
        prefix: str = "swebench_env",
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
        # In SWEBench, we always recreate the container for each instance.
        # TODO: Use an RNG object.
        self._current_task = random.choice(self._tasks)
        _LOGGER.debug("[%d] Selected task %s.", id(self), self._current_task.instance_id)

        self._test_spec = test_spec.make_test_spec(
            self._current_task.to_swebench_typeddict(),
            namespace="swebench",
            instance_image_tag="latest",
            env_image_tag="latest",
        )

        if not self._test_spec.is_remote_image:
            raise NotImplementedError("Need to implement local image support.")

        # We shouldn't need anything extra when creating the container.
        assert not self._test_spec.docker_specs, f"Expected no docker specs, got {self._test_spec.docker_specs}"

    async def _start_container(self) -> None:
        _LOGGER.info("[%d] Setting up container for instance %s.", id(self), self._test_spec.instance_id)
        with self._tracker.timeit(f"{self._prefix}/create_container"):
            self._container = await base.create_container(
                container_factory=self._container_factory,
                container_prefix=self._prefix,
                image_name=self._test_spec.instance_image_key,
            )
            await self._container.start()
        _LOGGER.debug("[%d] Container setup complete.", id(self))

    async def _start_code_agent(self) -> None:
        # Start the code agent, and await its response.
        # TODO: These duplicate checks can be made into a helper function.
        if self._container is None:
            raise RuntimeError("Container has not been created before starting the code agent.")
        if self._current_task is None:
            raise RuntimeError("Task has not been selected before starting the code agent.")

        _LOGGER.debug("[%d] Starting code agent.", id(self))
        self._code_agent = self._code_agent_factory(container=self._container, llm_client=self._llm_client)
        self._code_agent_task = asyncio.create_task(self._code_agent.run(self._current_task.problem_statement))
        _LOGGER.debug("[%d] Code agent started.", id(self))

    async def _compute_reward(self) -> float:
        if self._container is None:
            raise RuntimeError("Container has not been created before computing reward.")
        if self._current_task is None:
            raise RuntimeError("Task has not been selected before computing reward.")

        test_result = await _run_tests_and_evaluate(self._container, self._current_task, self._test_spec)
        return 1.0 if test_result["resolved"] else 0.0
