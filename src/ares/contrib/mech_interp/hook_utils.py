"""Hook utilities for mechanistic interpretability interventions."""

from collections.abc import Callable
import dataclasses

import torch
from transformer_lens import hook_points

from ares.containers import containers
from ares.environments import base


@dataclasses.dataclass
class FullyObservableState:
    timestep: base.TimeStep | None
    container: containers.Container | None
    step_num: int


def create_zero_ablation_hook(
    positions: list[int] | None = None,
    heads: list[int] | None = None,
) -> Callable[[torch.Tensor, hook_points.HookPoint], torch.Tensor]:
    """Create a hook that zeros out specific positions or attention heads.

    Args:
        positions: Optional list of token positions to ablate. If None, ablates all.
        heads: Optional list of attention head indices to ablate (for attention patterns).

    Returns:
        Hook function that performs zero ablation.

    Example:
        ```python
        # Ablate positions 5-10 in layer 0 residual stream
        hook = create_zero_ablation_hook(positions=list(range(5, 11)))
        model.run_with_hooks(
            tokens,
            fwd_hooks=[("blocks.0.hook_resid_post", hook)]
        )
        ```
    """

    def zero_ablation_hook(activation: torch.Tensor, hook: hook_points.HookPoint) -> torch.Tensor:  # noqa: ARG001
        ablated = activation.clone()

        if heads is not None:
            # For attention patterns: [batch, head, query_pos, key_pos]
            if len(ablated.shape) == 4:
                ablated[:, heads, :, :] = 0.0
            # For attention outputs: [batch, pos, head_index, d_head]
            elif len(ablated.shape) == 4:
                ablated[:, :, heads, :] = 0.0
        elif positions is not None:
            # For residual stream or other positional activations
            ablated[:, positions, :] = 0.0
        else:
            # Ablate everything
            ablated = torch.zeros_like(ablated)

        return ablated

    return zero_ablation_hook


def create_path_patching_hook(
    clean_activation: torch.Tensor,
    positions: list[int] | None = None,
) -> Callable[[torch.Tensor, hook_points.HookPoint], torch.Tensor]:
    """Create a hook for activation patching (path patching).

    This replaces activations from a corrupted run with those from a clean run,
    enabling causal analysis of information flow.

    Args:
        clean_activation: Activation tensor from the clean run to patch in.
        positions: Optional list of positions to patch. If None, patches all positions.

    Returns:
        Hook function that performs activation patching.

    Example:
        ```python
        # First, run on clean input and cache activations
        clean_cache, _ = model.run_with_cache(clean_tokens)
        clean_resid = clean_cache["blocks.0.hook_resid_post"]

        # Then run on corrupted input with patching
        hook = create_path_patching_hook(clean_resid, positions=[5, 6, 7])
        corrupted_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[("blocks.0.hook_resid_post", hook)]
        )
        ```
    """

    def path_patching_hook(activation: torch.Tensor, hook: hook_points.HookPoint) -> torch.Tensor:  # noqa: ARG001
        patched = activation.clone()

        if positions is not None:
            patched[:, positions, :] = clean_activation[:, positions, :]
        else:
            patched = clean_activation.clone()

        return patched

    return path_patching_hook


def create_mean_ablation_hook(
    mean_activation: torch.Tensor | None = None,
    positions: list[int] | None = None,
) -> Callable[[torch.Tensor, hook_points.HookPoint], torch.Tensor]:
    """Create a hook that replaces activations with their mean.

    Mean ablation is often more realistic than zero ablation as it preserves
    the scale of activations.

    Args:
        mean_activation: Pre-computed mean activation. If None, computes mean on-the-fly.
        positions: Optional list of positions to ablate.

    Returns:
        Hook function that performs mean ablation.
    """

    def mean_ablation_hook(activation: torch.Tensor, hook: hook_points.HookPoint) -> torch.Tensor:  # noqa: ARG001
        ablated = activation.clone()

        # Compute mean across batch and position dimensions if not provided
        mean = activation.mean(dim=(0, 1), keepdim=True) if mean_activation is None else mean_activation

        if positions is not None:
            ablated[:, positions, :] = mean.expand_as(ablated[:, positions, :])
        else:
            ablated = mean.expand_as(ablated)

        return ablated

    return mean_ablation_hook
