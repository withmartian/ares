"""TODO: LLM Docs"""

import inspect
from typing import Any, Callable, Coroutine, Union

from skyrl_train.generators import sky_rl_gym_generator


class SkyRLAsyncGymGenerator(sky_rl_gym_generator.SkyRLGymGenerator):
    """Wrapper around SkyRLGymGenerator to support async Gym Environments"""

    async def _run_in_executor_if_available(self, func: Union[Callable, Coroutine], *args: Any, **kwargs: Any):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await super()._run_in_executor_if_available(func, *args, **kwargs)
