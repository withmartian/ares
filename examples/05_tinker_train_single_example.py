"""TODO: Description
Code heavily inspired from
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/code_rl/code_env.py
and
https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/code_rl/train.py
"""

import asyncio
from collections.abc import Sequence
from datetime import datetime
import functools
import frozendict
import logging
from typing import Any, Literal

import chz
import numpy as np
import tinker
from tinker_cookbook import cli_utils, model_info, renderers, tokenizer_utils
from tinker_cookbook.rl import train as tinker_train
from tinker_cookbook.rl import types as tinker_types

import ares
from ares.containers import daytona
from ares.environments import base
from ares.llms import llm_clients
from ares import registry

_ALL_ENVS: list[base.Environment] = []

_LOGGER = logging.getLogger(__name__)


def _close_all_envs() -> None:
    """Tinker Env does not have a close pattern, so we have to make sure everything closed gracefully after running"""
    asyncio.gather(*[env.close() for env in _ALL_ENVS])


def _get_text_content(message: renderers.Message) -> str:
    """Extract text content from message, stripping thinking parts.

    Use this after parse_response when you only need the text output,
    ignoring any thinking/reasoning content.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")  # type: ignore


def _middle_truncate(model_input: tinker.ModelInput, max_tokens: int) -> tinker.ModelInput:
    num_tokens_to_truncate = model_input.length - max_tokens + 10
    if num_tokens_to_truncate <= 0:
        return model_input

    center_idx = model_input.length // 2
    truncate_start_idx = center_idx - num_tokens_to_truncate // 2
    truncate_end_idx = center_idx + num_tokens_to_truncate // 2

    curr_ints = model_input.to_ints()
    # TODO: Put something in between (like an ellipsis)
    new_ints = curr_ints[:truncate_start_idx] + curr_ints[truncate_end_idx:]
    return tinker.ModelInput.from_ints(new_ints)


class TinkerCompatibleEnv(tinker_types.Env):
    def __init__(
        self,
        env: base.Environment[llm_clients.LLMResponse, llm_clients.LLMRequest, float, float],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
    ):
        self.env = env
        self.renderer = renderer
        self.convo_prefix = convo_prefix

    def _get_tinker_observation(self, ts: base.TimeStep[llm_clients.LLMRequest | None, float, float]) -> tinker_types.Observation:
        if ts.observation is None:
            return tinker.ModelInput.empty()

        messages = [
            renderers.Message(role=message["role"], content=message["content"])  # type: ignore
            for message in ts.observation.messages
        ]
        model_input = self.renderer.build_generation_prompt(messages)
        # TODO: How do we access model max context length and max_tokens?
        if model_input.length + 2048 > 32768 - 10:
            model_input = _middle_truncate(model_input, 32768 - 2048)
        
        return model_input

    def _get_ares_action(self, action: tinker_types.Action) -> llm_clients.LLMResponse:
        message, parse_success = self.renderer.parse_response(action)
        if not parse_success:
            _LOGGER.warning("Failed to parse response: %s", message)

        return llm_clients.build_openai_compatible_llm_response(
            output_text=_get_text_content(message),
            num_input_tokens=-1,
            num_output_tokens=-1,
            model="tinker",
            cost=0.0,
        )

    @property
    def stop_condition(self) -> tinker_types.StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker_types.Observation, tinker_types.StopCondition]:
        # Hack to set is_active without context manager
        self.env._is_active = True  # type: ignore
        ts = await self.env.reset()
        return self._get_tinker_observation(ts), self.stop_condition

    async def step(self, action: tinker_types.Action) -> tinker_types.StepResult:
        ares_action = self._get_ares_action(action)
        ts = await self.env.step(ares_action)

        if ts.last():
            await self.env.close()
            _ALL_ENVS.remove(self.env)

        return tinker_types.StepResult(
            reward=ts.reward or 0.0,
            episode_done=ts.last(),
            next_observation=self._get_tinker_observation(ts),
            next_stop_condition=self.stop_condition,
            metrics={
                "reward": ts.reward or 0.0,
                "step_count": self.env._step_count,  # type: ignore
            },
        )


@chz.chz
class TinkerEnvGroupBuilder(tinker_types.EnvGroupBuilder):
    env_preset_name: str
    env_preset_idx: int
    env_make_kwargs: frozendict.frozendict[str, Any]
    group_size: int
    renderer: renderers.Renderer
    convo_prefix: list[renderers.Message] | None = None

    async def make_envs(self) -> Sequence[TinkerCompatibleEnv]:
        envs: list[TinkerCompatibleEnv] = []
        for _ in range(self.group_size):
            env = ares.make(
                f"{self.env_preset_name}:{self.env_preset_idx}",
                **self.env_make_kwargs,
            )
            _ALL_ENVS.append(env)
            envs.append(
                TinkerCompatibleEnv(
                    env,
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                )
            )
        return envs


@chz.chz
class TinkerDataset(tinker_types.RLDataset):
    # ARES params
    env_preset_name: str
    env_make_kwargs: frozendict.frozendict[str, Any]
    max_num_tasks: int | None = None
    # Tinker params
    batch_size: int
    renderer: renderers.Renderer
    group_size: int
    convo_prefix: list[renderers.Message] | None = None
    split: Literal["train", "test"]
    seed: int = 0

    @property
    def num_tasks(self) -> int:
        return self.max_num_tasks or self.env_info.num_tasks

    @functools.cached_property
    def idx_map(self) -> dict[int, int]:
        """Indexing map for shuffling the dataset before indexing via ares.make"""
        np.random.seed(self.seed)
        # Need to use self.env_info.num_tasks here to ensure random shuffling
        shuffled_indices = list(range(self.env_info.num_tasks))
        np.random.shuffle(shuffled_indices)

        return {idx: shuffled_idx for idx, shuffled_idx in enumerate(shuffled_indices[:self.num_tasks])}

    @functools.cached_property
    def env_info(self) -> registry.EnvironmentInfo:
        return registry.info(self.env_preset_name)

    def get_batch(self, index: int) -> Sequence[TinkerEnvGroupBuilder]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.num_tasks)
        if start >= end:
            raise IndexError("Incorrect batch index for TinkerDataset")

        builders: list[TinkerEnvGroupBuilder] = []
        for idx in range(start, end):
            selected_idx = self.idx_map[idx]
            builders.append(
                TinkerEnvGroupBuilder(
                    env_preset_name=self.env_preset_name,
                    env_preset_idx=selected_idx,
                    env_make_kwargs=self.env_make_kwargs,
                    group_size=self.group_size,
                    renderer=self.renderer,
                    convo_prefix=self.convo_prefix,
                )
            )

        return builders

    def __len__(self) -> int:
        return (self.num_tasks + self.batch_size - 1) // self.batch_size


@chz.chz
class TinkerDatasetBuilder(tinker_types.RLDatasetBuilder):
    # ARES params
    env_preset_name: str
    env_make_kwargs: frozendict.frozendict[str, Any]
    num_tasks: int | None = None
    # Tinker params
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int  # NOTE: This is how many times to rerun the same env
    convo_prefix: list[renderers.Message] | None = None
    seed: int = 0

    async def __call__(self) -> tuple[TinkerDataset, TinkerDataset]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        train_ds = TinkerDataset(
            env_preset_name=self.env_preset_name,
            env_make_kwargs=self.env_make_kwargs,
            max_num_tasks=self.num_tasks,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            split="train",
            seed=self.seed,
        )
        test_ds = TinkerDataset(
            env_preset_name=self.env_preset_name,
            env_make_kwargs=self.env_make_kwargs,
            max_num_tasks=self.num_tasks,
            batch_size=self.batch_size,
            group_size=1,
            renderer=renderer,
            convo_prefix=self.convo_prefix,
            split="test",
            seed=self.seed,
        )
        return train_ds, test_ds


# Config

@chz.chz
class CLIConfig:
    """Command-line configuration for DeepCoder RL training."""

    # Model configuration
    # model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    # Data / environment configuration
    seed: int = 0

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 20
    learning_rate: float = 1e-6
    max_tokens: int = 2048
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Logging / eval / checkpoints
    log_dir: str | None = None
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 0
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Async rollout configuration
    max_steps_off_policy: int | None = 10

    # ARES params
    env_preset_name: str = "sbv-mswea"
    env_make_kwargs: frozendict.frozendict[str, Any] = frozendict.frozendict(
        {"container_factory": daytona.DaytonaContainer}
    )
    num_tasks: int | None = 40


async def main():
    cli_config = CLIConfig()

    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"ares-tinker-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    # Set log path
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/code_rl/{run_name}"

    wandb_name = cli_config.wandb_name or run_name

    dataset_builder = TinkerDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
        env_preset_name=cli_config.env_preset_name,
        env_make_kwargs=cli_config.env_make_kwargs,
    )

    config = tinker_train.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=tinker_train.AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    try:
        await tinker_train.main(config)
    finally:
        _close_all_envs()


if __name__ == "__main__":
    asyncio.run(main())
