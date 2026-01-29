"""TODO: LLM Docs"""

import inspect
from typing import Any, Callable, Coroutine, Union

from skyrl_train.generators import sky_rl_gym_generator
from skyrl_gym.envs import base_text_env


class SkyRLAsyncGymGenerator(sky_rl_gym_generator.SkyRLGymGenerator):
    """Wrapper around SkyRLGymGenerator to support async Gym Environments"""

    async def _run_in_executor_if_available(self, func: Union[Callable, Coroutine], *args: Any, **kwargs: Any):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await super()._run_in_executor_if_available(func, *args, **kwargs)


class ARESSkyRLGymEnv(base_text_env.BaseTextEnv):
    async def step(self, action: str) -> base_text_env.BaseTextEnvStepOutput:
        """
        Runs one environment step.

        Return:
        - new_obs: [{"role": "user", "content": observation}]
        - reward: float
        - done: bool
        - postprocessed_action: Optional[str]
        - Dict[str, Any]: any metadata
        """
        pass

    async def init(self, prompt: base_text_env.ConversationType) -> tuple[base_text_env.ConversationType, dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        """
        return prompt, {}

    async def close(self) -> None:
        """
        Closes the environment, override if needed by subclasses.
        """
        pass
