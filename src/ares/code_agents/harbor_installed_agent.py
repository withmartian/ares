import pathlib

from harbor.environments import base as harbor_env_base
from harbor.agents.installed import base as harbor_ia_base

from ares.code_agents import code_agent_base
from ares.containers import containers
from ares.containers import docker as ares_docker
from ares.containers import daytona as ares_daytona
from ares.llms import llm_clients


class ContainerToHarborEnvWrapper(harbor_env_base.BaseEnvironment):
    """Wrapper around harbor's BaseEnvironment for ARES containers"""
    def __init__(self, container: containers.Container):
        self._container = container

    def type(self) -> harbor_env_base.EnvironmentType:
        match type(self._container):
            case ares_docker.DockerContainer:
                return harbor_env_base.EnvironmentType.DOCKER
            case ares_daytona.DaytonaContainer:
                return harbor_env_base.EnvironmentType.DAYTONA
            case _:
                raise ValueError(f"Unsupported container type: {type(self._container)}")

    @property
    def is_mounted(self) -> bool:
        return False

    @property
    def supports_gpus(self) -> bool:
        return False
    
    @property
    def can_disable_internet(self) -> bool:
        return False
    
    def _validate_definition(self) -> None:
        # TODO: Should we do anything here?
        pass

    async def start(self, force_build: bool) -> None:
        await self._container.start()
    
    async def stop(self, delete: bool) -> None:
        await self._container.stop()
        if delete:
            self._container.stop_and_remove()

    async def upload_file(self, source_path: pathlib.Path | str, target_path: str) -> None:
        await self._container.upload_file(pathlib.Path(source_path), target_path)

    async def upload_dir(self, source_dir: pathlib.Path | str, target_dir: str) -> None:
        await self._container.upload_dir(pathlib.Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: pathlib.Path | str) -> None:
        await self._container.download_file(source_path, pathlib.Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: pathlib.Path | str) -> None:
        await self._container.download_dir(source_dir, pathlib.Path(target_dir))

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> harbor_env_base.ExecResult:
        ares_result = await self._container.exec_run(command, workdir=cwd, env=env, timeout_s=timeout_sec)
        return harbor_env_base.ExecResult(
            # TODO: Split stdout/stderr
            stdout=ares_result.output,
            stderr="",
            return_code=ares_result.exit_code,
        )


class HarborInstalledAgent(code_agent_base.CodeAgent):
    def __init__(self, container: containers.Container, llm_client: llm_clients.LLMClient, harbor_ia_cls: type[harbor_ia_base.BaseInstalledAgent]):
        self._container = container
        self._llm_client = llm_client
        self._harbor_env = ContainerToHarborEnvWrapper(container)
        self._harbor_installed_agent = harbor_ia_cls(
            logs_dir=# TODO,
            prompt_template_path=# TODO,
        )

    async def run(self, task: str) -> None:
        agent_context = harbor_ia_base.AgentContext()
        await self._harbor_installed_agent.setup(self._harbor_env)
        await self._harbor_installed_agent.run(
            # TODO: FIGURE OUT INSTRUCTION
            instruction="TODO",
            environment=self._harbor_env,
            context=agent_context,
        )
