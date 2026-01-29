"""ARES + Tinker RL Training Script

Train code agents using reinforcement learning with ARES environments and Tinker's training infrastructure.
This script combines ARES's Harbor-compatible environments (including SWE-bench) with Tinker's RL training
loop, enabling async rollouts, LoRA fine-tuning, and WandB logging.

The integration works by wrapping ARES environments in Tinker-compatible adapters that:
- Convert LLM requests/responses between ARES and Tinker formats
- Handle tokenization and prompt rendering
- Manage episode lifecycle and reward computation
- Support async parallel rollouts across multiple environments

Prerequisites:
    Set your TINKER_API_KEY environment variable.

Usage:
    # Train with defaults (Qwen3-4B on sbv-mswea)
    uv run -m examples.05_tinker_train

    # Train with custom model and hyperparameters
    uv run -m examples.05_tinker_train \
        model_name=meta-llama/Llama-3.1-8B-Instruct \
        learning_rate=5e-7 \
        lora_rank=64 \
        num_tasks=100

    # Enable WandB logging
    uv run -m examples.05_tinker_train \
        wandb_project=ares-rl \
        wandb_name=my-experiment

    # Continue from checkpoint
    uv run -m examples.05_tinker_train \
        load_checkpoint_path=/path/to/checkpoint

Implementation heavily inspired by:
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/code_rl/code_env.py
- https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/code_rl/train.py
"""

import asyncio
from collections.abc import Sequence
import datetime as dt
import functools
import logging
import os
from typing import Any, Literal

import ares
from ares import containers
from ares import llms
import chz
import frozendict
import numpy as np
import tinker
from tinker_cookbook import cli_utils
from tinker_cookbook import model_info
from tinker_cookbook import renderers
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.rl import train as tinker_train
from tinker_cookbook.rl import types as tinker_types

_LOGGER = logging.getLogger(__name__)

CONTEXT_LEN_BUFFER = 10
# I can't find how to programmatically access this anywhere, def varies by model
DEFAULT_MAX_CONTEXT_LEN = 32768


# === Utility Functions ===


def _get_text_content(message: renderers.Message) -> str:
    """Extract text content from message, stripping thinking parts.

    Use this after parse_response when you only need the text output,
    ignoring any thinking/reasoning content.
    """
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")  # type: ignore


def _middle_truncate(model_input: tinker.ModelInput, max_context_len: int) -> tinker.ModelInput:
    """Truncate model input from the middle when exceeding max context length.

    This preserves both the beginning (task context) and end (recent history)
    of the conversation while removing middle content.
    """
    num_tokens_to_truncate = model_input.length - max_context_len + CONTEXT_LEN_BUFFER
    if num_tokens_to_truncate <= 0:
        return model_input

    center_idx = model_input.length // 2
    truncate_start_idx = center_idx - num_tokens_to_truncate // 2
    truncate_end_idx = center_idx + num_tokens_to_truncate // 2

    curr_ints = model_input.to_ints()
    # TODO: Put something in between (like an ellipsis)
    new_ints = curr_ints[:truncate_start_idx] + curr_ints[truncate_end_idx:]
    return tinker.ModelInput.from_ints(new_ints)


# === Environment Adapters ===


class TinkerCompatibleEnv(tinker_types.Env):
    """Adapter wrapping ARES environments to work with Tinker's RL training loop.

    Handles bidirectional conversion:
    - ARES LLMRequest -> Tinker ModelInput (tokenized prompts)
    - Tinker Action (text) -> ARES LLMResponse
    - ARES TimeStep -> Tinker StepResult

    This enables using any ARES environment with Tinker's training infrastructure.

    NOTE: Sandbox cleanup is not great here, and we are mostly relying on the Janitor in
          src/ares/environments/code_env.py to clean up extra sandboxes in the case of errors.
    """

    def __init__(
        self,
        env: ares.Environment[llms.LLMResponse, llms.LLMRequest, float, float],
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None,
        max_tokens: int,
    ):
        self.env = env
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.max_tokens = max_tokens

    def _get_tinker_observation(
        self, ts: ares.TimeStep[llms.LLMRequest | None, float, float]
    ) -> tinker_types.Observation:
        if ts.observation is None:
            return tinker.ModelInput.empty()

        messages = self.convo_prefix + [
            renderers.Message(role=message["role"], content=message["content"])  # type: ignore
            for message in ts.observation.messages
        ]
        model_input = self.renderer.build_generation_prompt(messages)

        # May need to truncate context len to prevent errors
        if model_input.length > DEFAULT_MAX_CONTEXT_LEN - self.max_tokens + CONTEXT_LEN_BUFFER:
            model_input = _middle_truncate(model_input, DEFAULT_MAX_CONTEXT_LEN - self.max_tokens)

        return model_input

    def _get_ares_action(self, action: tinker_types.Action) -> llms.LLMResponse:
        message, parse_success = self.renderer.parse_response(action)
        if not parse_success:
            _LOGGER.warning("Failed to parse response: %s", message)

        return llms.LLMResponse(
            data=[llms.TextData(content=_get_text_content(message))],
            cost=0.0,
            usage=llms.Usage(prompt_tokens=-1, generated_tokens=-1),
        )

    @property
    def stop_condition(self) -> tinker_types.StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker_types.Observation, tinker_types.StopCondition]:
        # Hack to approximate a context manager
        await self.env.__aenter__()
        ts = await self.env.reset()
        return self._get_tinker_observation(ts), self.stop_condition

    async def step(self, action: tinker_types.Action) -> tinker_types.StepResult:
        ares_action = self._get_ares_action(action)
        ts = await self.env.step(ares_action)

        if ts.last():
            # Hack to approximate a context manager
            await self.env.__aexit__(None, None, None)

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
    """Factory for creating groups of identical ARES environments for parallel rollouts.

    Creates `group_size` copies of the same environment instance (same task) to enable
    multiple rollout attempts per task, improving gradient estimates in RL training.
    """

    env_preset_name: str
    env_preset_idx: int
    env_make_kwargs: frozendict.frozendict[str, Any]
    group_size: int
    renderer: renderers.Renderer
    convo_prefix: list[renderers.Message] | None = None
    max_tokens: int

    async def make_envs(self) -> Sequence[TinkerCompatibleEnv]:
        envs: list[TinkerCompatibleEnv] = []
        for _ in range(self.group_size):
            env = ares.make(
                f"{self.env_preset_name}:{self.env_preset_idx}",
                **self.env_make_kwargs,
            )
            envs.append(
                TinkerCompatibleEnv(
                    env,
                    self.renderer,
                    convo_prefix=self.convo_prefix,
                    max_tokens=self.max_tokens,
                )
            )
        return envs


@chz.chz
class TinkerDataset(tinker_types.RLDataset):
    """RL dataset wrapping ARES environment presets with batching and shuffling.

    Provides batched access to ARES tasks (e.g., SWE-bench instances) with:
    - Random shuffling based on seed
    - Batch iteration for parallel training
    - Train/test split support
    - Optional task limit for quick experiments
    """

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
    max_tokens: int

    @property
    def num_tasks(self) -> int:
        return (
            self.env_info.num_tasks if self.max_num_tasks is None else min(self.max_num_tasks, self.env_info.num_tasks)
        )

    @functools.cached_property
    def idx_map(self) -> dict[int, int]:
        """Indexing map for shuffling the dataset before indexing via ares.make"""
        np.random.seed(self.seed)
        # Need to use self.env_info.num_tasks here to ensure random shuffling
        shuffled_indices = list(range(self.env_info.num_tasks))
        np.random.shuffle(shuffled_indices)

        return dict(enumerate(shuffled_indices[: self.num_tasks]))

    @functools.cached_property
    def env_info(self) -> ares.EnvironmentInfo:
        return ares.info(self.env_preset_name)

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
                    max_tokens=self.max_tokens,
                )
            )

        return builders

    def __len__(self) -> int:
        return (self.num_tasks + self.batch_size - 1) // self.batch_size


@chz.chz
class TinkerDatasetBuilder(tinker_types.RLDatasetBuilder):
    """Builder for constructing train and test datasets from ARES environments.

    Handles tokenizer initialization, renderer setup, and dataset configuration.
    Returns separate train and test datasets with appropriate group sizes:
    - Train: group_size rollouts per task (for better gradients)
    - Test: 1 rollout per task (for efficient evaluation)
    """

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
    max_tokens: int

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
            max_tokens=self.max_tokens,
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
            max_tokens=self.max_tokens,
        )
        return train_ds, test_ds


# === CLI Configuration ===


@chz.chz
class CLIConfig:
    """Command-line configuration for ARES + Tinker RL training.

    All fields can be overridden via command-line arguments.
    Example: model_name=meta-llama/Llama-3.1-8B-Instruct learning_rate=5e-7
    """

    # === Model Configuration ===
    # HuggingFace model identifier - supported models:
    # https://tinker-docs.thinkingmachines.ai/model-lineup#full-listing
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    # LoRA rank for parameter-efficient fine-tuning (higher = more capacity, slower)
    lora_rank: int = 32
    # Renderer for prompt formatting (auto-detected from model if None)
    renderer_name: str | None = None
    # Path to resume training from a checkpoint
    load_checkpoint_path: str | None = None

    # === ARES Environment Configuration ===
    # ARES environment preset (e.g., "sbv-mswea", "tbench-mswea")
    env_preset_name: str = "sbv-mswea"
    # Number of tasks to train on (None = use all available)
    num_tasks: int | None = 20
    # Additional kwargs passed to ares.make() (e.g., container factory, resources)
    env_make_kwargs: frozendict.frozendict[str, Any] = frozendict.frozendict(
        {"container_factory": containers.DaytonaContainer}
    )

    # === Training Hyperparameters ===
    # Number of rollouts per environment (higher = better gradient estimates)
    group_size: int = 4
    # Number of environment groups per training batch
    groups_per_batch: int = 20
    # Learning rate for LoRA parameter updates
    learning_rate: float = 1e-6
    # Maximum tokens to generate per LLM response
    max_tokens: int = 2048
    # KL penalty coefficient for policy regularization (0.0 = no regularization)
    kl_penalty_coef: float = 0.0
    # Number of gradient accumulation substeps per batch
    num_substeps: int = 1
    # Random seed for reproducibility
    seed: int = 0

    # === Async Rollout Configuration ===
    # Max steps an environment can lag behind current policy (None = synchronous, 10 = async)
    # NOTE: This is required to enable async rollouts.
    max_steps_off_policy: int | None = 10

    # === Logging, Evaluation, and Checkpointing ===
    # Directory for logs (deprecated, use log_path instead)
    log_dir: str | None = None
    # Full path for training logs and checkpoints (auto-generated if None)
    log_path: str | None = None
    # WandB project name (enables WandB logging if set)
    wandb_project: str | None = None
    # WandB run name (auto-generated if None)
    wandb_name: str | None = None
    # Compute KL divergence after each update (adds overhead)
    compute_post_kl: bool = False
    # Run evaluation every N training steps (0 = no evaluation)
    eval_every: int = 0
    # Save checkpoint every N training steps
    save_every: int = 20
    # Behavior when log directory already exists ("ask", "overwrite", "fail")
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # === Service Configuration ===
    # Tinker service base URL (None = use default local service)
    base_url: str | None = None


# === Main Training Script ===


async def main(cli_config: CLIConfig):
    """Main training loop with ARES environments and Tinker RL infrastructure.

    Args:
        cli_config: Command-line configuration parsed by chz.entrypoint()
    """
    # Fail fast if env vars aren't set.
    if "TINKER_API_KEY" not in os.environ:
        raise ValueError("TINKER_API_KEY is not set")
    if (
        cli_config.env_make_kwargs.get("container_factory") == containers.DaytonaContainer
        and "DAYTONA_API_KEY" not in os.environ
    ):
        raise ValueError("DAYTONA_API_KEY is not set")

    # Auto-detect renderer if not specified
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(cli_config.model_name)

    # Generate run name for logging and checkpointing
    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"ares-tinker-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-seed{cli_config.seed}-"
        f"{dt.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    # Set log path (use provided or auto-generate)
    log_path = f"/tmp/tinker-examples/code_rl/{run_name}" if cli_config.log_path is None else cli_config.log_path

    # Set WandB run name (use provided or auto-generate)
    wandb_name = cli_config.wandb_name or run_name

    _LOGGER.info("Starting training run: %s", run_name)
    _LOGGER.info("Model: %s (rank=%d)", cli_config.model_name, cli_config.lora_rank)
    _LOGGER.info("Environment: %s (%d tasks)", cli_config.env_preset_name, cli_config.num_tasks or 0)
    _LOGGER.info("Log path: %s", log_path)

    # Build ARES dataset with Tinker compatibility layer
    dataset_builder = TinkerDatasetBuilder(
        batch_size=cli_config.groups_per_batch,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        group_size=cli_config.group_size,
        seed=cli_config.seed,
        env_preset_name=cli_config.env_preset_name,
        env_make_kwargs=cli_config.env_make_kwargs,
        num_tasks=cli_config.num_tasks,
        max_tokens=cli_config.max_tokens,
    )

    # Configure Tinker training
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

    # Verify log directory behavior
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await tinker_train.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig, allow_hyphens=True)
    asyncio.run(main(cli_config))
