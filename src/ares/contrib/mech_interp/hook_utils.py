"""Hook utilities for mechanistic interpretability interventions."""

from collections.abc import Callable
import dataclasses
from typing import Any

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from ares.containers import containers
from ares.environments.base import TimeStep


@dataclasses.dataclass
class FullyObservableState:
    timestep: TimeStep | None
    container: containers.Container | None
    step_num: int


def create_zero_ablation_hook(
    positions: list[int] | None = None,
    heads: list[int] | None = None,
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
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

    def zero_ablation_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
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
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
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

    def path_patching_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
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
) -> Callable[[torch.Tensor, HookPoint], torch.Tensor]:
    """Create a hook that replaces activations with their mean.

    Mean ablation is often more realistic than zero ablation as it preserves
    the scale of activations.

    Args:
        mean_activation: Pre-computed mean activation. If None, computes mean on-the-fly.
        positions: Optional list of positions to ablate.

    Returns:
        Hook function that performs mean ablation.
    """

    def mean_ablation_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
        ablated = activation.clone()

        # Compute mean across batch and position dimensions if not provided
        mean = activation.mean(dim=(0, 1), keepdim=True) if mean_activation is None else mean_activation

        if positions is not None:
            ablated[:, positions, :] = mean.expand_as(ablated[:, positions, :])
        else:
            ablated = mean.expand_as(ablated)

        return ablated

    return mean_ablation_hook


class InterventionManager:
    """Manager for applying multiple interventions during agent execution.

    This class helps coordinate multiple hook-based interventions across an agent
    trajectory, making it easy to study causal effects of different model components.

    Example:
        ```python
        model = HookedTransformer.from_pretrained("gpt2-small")
        client = HookedTransformerLLMClient(model=model, model_name="gpt2-small")

        # Create intervention manager
        manager = InterventionManager(model)

        # Add interventions
        manager.add_intervention(
            hook_name="blocks.0.attn.hook_pattern",
            hook_fn=create_zero_ablation_hook(heads=[0, 1]),
            description="Ablate attention heads 0-1 in layer 0"
        )

        # Run with interventions active
        with manager:
            async with env:
                ts = await env.reset()
                while not ts.last():
                    response = await client(ts.observation)
                    ts = await env.step(response)
        ```
    """

    def __init__(self, model: HookedTransformer):
        """Initialize intervention manager.

        Args:
            model: HookedTransformer to apply interventions to.
        """
        self.model = model
        self.interventions: list[dict[str, Any]] = []
        self._active_handles: list[Any] = []

    def add_intervention(
        self,
        hook_name: str,
        hook_fn: Callable[[torch.Tensor, HookPoint], torch.Tensor],
        description: str = "",
        apply_at_steps: list[int] | None = None,
    ) -> None:
        """Add an intervention to apply during execution.

        Args:
            hook_name: Name of the hook point (e.g., "blocks.0.hook_resid_post").
            hook_fn: Hook function to apply.
            description: Human-readable description of the intervention.
            apply_at_steps: Optional list of step indices when to apply this intervention.
                If None, applies at all steps.
        """
        self.interventions.append(
            {
                "hook_name": hook_name,
                "hook_fn": hook_fn,
                "description": description,
                "apply_at_steps": apply_at_steps,
                "step_count": 0,
            }
        )

    def clear_interventions(self) -> None:
        """Remove all interventions."""
        self.interventions.clear()

    def __enter__(self) -> "InterventionManager":
        """Enter context manager and activate interventions."""
        for intervention in self.interventions:
            hook_point = self.model.hook_dict[intervention["hook_name"]]

            # Wrap hook_fn to track step count and properly capture loop variables
            def make_wrapped_hook(interv):
                # Capture these from the intervention dict to avoid loop variable binding issues
                hook_fn = interv["hook_fn"]
                steps = interv["apply_at_steps"]

                def wrapped_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:
                    # Check if we should apply this intervention at current step
                    if steps is None or interv["step_count"] in steps:
                        return hook_fn(activation, hook)
                    return activation

                return wrapped_hook

            handle = hook_point.add_hook(make_wrapped_hook(intervention))
            self._active_handles.append(handle)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and deactivate interventions."""
        for handle in self._active_handles:
            if handle is not None:
                handle.remove()
        self._active_handles.clear()

    def increment_step(self) -> None:
        """Increment the step counter for all interventions.

        Call this between agent steps to track which step you're on.
        """
        for intervention in self.interventions:
            intervention["step_count"] += 1

    def get_intervention_summary(self) -> str:
        """Get a summary of all active interventions.

        Returns:
            Human-readable summary string.
        """
        if not self.interventions:
            return "No interventions active"

        lines = ["Active interventions:"]
        for i, interv in enumerate(self.interventions, 1):
            desc = interv["description"] or "No description"
            hook = interv["hook_name"]
            steps = interv["apply_at_steps"]
            step_str = f"steps {steps}" if steps else "all steps"
            lines.append(f"  {i}. {desc} ({hook}, {step_str})")

        return "\n".join(lines)
