"""RL training with SkyRL

Example heavily inspired by:
https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/terminal_bench/entrypoints/main_tbench.py

TODO: Add usage docs
"""

import functools
import inspect
from typing import Any, Callable, Coroutine, Union

import ray
import hydra
from omegaconf import DictConfig
from skyrl_gym.envs import base_text_env
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl_train.generators import sky_rl_gym_generator
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray

import ares
from ares import llms


class SkyRLAsyncGymGenerator(sky_rl_gym_generator.SkyRLGymGenerator):
    """Wrapper around SkyRLGymGenerator to support async Gym Environments"""

    async def _run_in_executor_if_available(self, func: Union[Callable, Coroutine], *args: Any, **kwargs: Any):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await super()._run_in_executor_if_available(func, *args, **kwargs)


class ARESSkyRLGymEnv(base_text_env.BaseTextEnv):
    """Wrapper around ares.CodeEnvironment for SkyRL Gym
    
    NOTE: Sandbox cleanup is not great here, and we are mostly relying on the Janitor in
          src/ares/environments/code_env.py to clean up extra sandboxes in the case of errors.
    """

    def __init__(self, preset_name: str):
        self.preset_name = preset_name
        self.env: ares.Environment[llms.LLMResponse, llms.LLMRequest, float, float] | None = None

    async def init(self, prompt: base_text_env.ConversationType) -> tuple[base_text_env.ConversationType, dict[str, Any]]:
        """
        Return the first prompt to be given to the model and optional metadata.
        """
        task_idx = int(prompt[0]["content"])
        self.env = ares.make(f"{self.preset_name}:{task_idx}")

        # Setting up ARES environment
        await self.env.__aenter__()
        ts = await self.env.reset()

        return ts.observation.messages, {}  # type: ignore

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
        assert self.env is not None

        llm_resp = llms.LLMResponse(
            data=[llms.TextData(content=action)],
            cost=0.0,
            usage=llms.Usage(prompt_tokens=-1, generated_tokens=-1),
        )
        ts = await self.env.step(llm_resp)

        if ts.last():
            # Hack to approximate a context manager
            await self.env.__aexit__(None, None, None)

        return base_text_env.BaseTextEnvStepOutput(
            observations=ts.observation.messages,  # type: ignore
            reward=ts.reward or 0.0,
            done=ts.last(),
            metadata={},
        )

    async def close(self) -> None:
        """
        Closes the environment, override if needed by subclasses.
        """
        if self.env is not None:
            await self.env.__aexit__(None, None, None)
            self.env = None


class ARESCodeEnvDataset:
    def __init__(self, preset_name: str):
        self.preset_name = preset_name

    def __getitem__(self, index: int) -> dict:
        return {
            "prompt": [{"role": "user", "content": str(index)}],
            "env_class": "ARESSkyRLGymEnv",
            "env_extras": {"preset_name": self.preset_name},
            "uid": str(index),
        }

    @functools.lru_cache(maxsize=1)
    def __len__(self) -> int:
        return ares.info(self.preset_name).num_tasks

    def collate_fn(self, batch: list[ARESSkyRLGymEnv]) -> list[ARESSkyRLGymEnv]:
        return batch


class ARESExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the SkyRLAsyncGymGenerator.
        """
        return SkyRLAsyncGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            ARESCodeEnvDataset: The training dataset.
        """
        dataset = ARESCodeEnvDataset(
            preset_name=self.cfg.data.ares_preset_name_train,
        )
        return dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            ARESCodeEnvDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.ares_preset_name_val:
            dataset = ARESCodeEnvDataset(
                preset_name=self.cfg.data.ares_preset_name_val,
            )
            return dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = ARESExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

