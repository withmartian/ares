"""Utilities for capturing and analyzing activations across agent trajectories."""

from collections.abc import Callable
import dataclasses
import pathlib
from typing import Any

import torch
from transformer_lens import ActivationCache
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint


@dataclasses.dataclass
class TrajectoryActivations:
    """Container for activations captured across an agent trajectory.

    Attributes:
        step_activations: List of ActivationCache objects, one per agent step.
        step_metadata: Optional metadata for each step (e.g., observation, action).
        model_name: Name of the model used.
    """

    step_activations: list[ActivationCache]
    step_metadata: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    model_name: str = "unknown"

    def __len__(self) -> int:
        """Return number of steps in the trajectory."""
        return len(self.step_activations)

    def save(self, path: str | pathlib.Path) -> None:
        """Save trajectory activations to disk.

        Args:
            path: Directory path to save activations. Will be created if it doesn't exist.
        """
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each step's activations as plain dicts (ActivationCache can't be pickled)
        for i, cache in enumerate(self.step_activations):
            # Convert ActivationCache to dict for serialization
            cache_dict = dict(cache.cache_dict.items())
            torch.save(cache_dict, path / f"step_{i:04d}.pt")

        # Save metadata
        import json

        metadata = {
            "model_name": self.model_name,
            "num_steps": len(self),
            "step_metadata": self.step_metadata,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "TrajectoryActivations":
        """Load trajectory activations from disk.

        Args:
            path: Directory path containing saved activations.

        Returns:
            TrajectoryActivations instance.
        """
        import json

        path = pathlib.Path(path)

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        # Load activation dicts (we saved as plain dicts, not ActivationCache objects)
        step_files = sorted(path.glob("step_*.pt"))
        step_activations_dicts = [torch.load(f, weights_only=False) for f in step_files]

        # Convert back to ActivationCache objects if needed, or just use dicts
        # For now, we'll keep them as ActivationCache objects for API compatibility
        step_activations = [ActivationCache(cache_dict, model=None) for cache_dict in step_activations_dicts]

        return cls(
            step_activations=step_activations,
            step_metadata=metadata.get("step_metadata", []),
            model_name=metadata.get("model_name", "unknown"),
        )

    def get_activation(self, step: int, hook_name: str) -> torch.Tensor:
        """Get activation tensor for a specific step and hook.

        Args:
            step: Step index in trajectory.
            hook_name: Name of the hook point (e.g., "blocks.0.attn.hook_pattern").

        Returns:
            Activation tensor for the specified step and hook.
        """
        return self.step_activations[step][hook_name]

    def get_activation_across_trajectory(self, hook_name: str) -> list[torch.Tensor]:
        """Get activations for a specific hook across all trajectory steps.

        Args:
            hook_name: Name of the hook point.

        Returns:
            List of activation tensors, one per step.
        """
        return [cache[hook_name] for cache in self.step_activations]


class ActivationCapture:
    """Context manager for capturing activations during agent execution.

    This class provides a convenient way to capture activations from a HookedTransformer
    during an ARES agent episode, enabling trajectory-level mechanistic interpretability
    analysis.

    Example:
        ```python
        from transformer_lens import HookedTransformer
        from ares.contrib.mech_interp import ActivationCapture, HookedTransformerLLMClient

        model = HookedTransformer.from_pretrained("gpt2-small")
        client = HookedTransformerLLMClient(model=model)

        # Capture activations during episode
        with ActivationCapture(model) as capture:
            async with env:
                ts = await env.reset()
                while not ts.last():
                    response = await client(ts.observation)
                    capture.record_step_metadata({"action": response})
                    ts = await env.step(response)

        # Analyze captured activations
        trajectory = capture.get_trajectory()
        print(f"Captured {len(trajectory)} steps")

        # Save for later analysis
        trajectory.save("./activations/episode_001")
        ```
    """

    def __init__(
        self,
        model: HookedTransformer,
        hook_filter: Callable[[str], bool] | None = None,
    ):
        """Initialize activation capture.

        Args:
            model: HookedTransformer to capture activations from.
            hook_filter: Optional function to filter which hooks to capture.
                If None, captures all hooks. Example: lambda name: "attn" in name
        """
        # TODO: Should we store logits and loss as well? By default or via flag?
        self.model = model
        self.hook_filter = hook_filter
        self.step_activations: list[ActivationCache] = []
        self.step_metadata: list[dict[str, Any]] = []
        self._hook_handles: list[Any] = []

    def __enter__(self) -> "ActivationCapture":
        """Enter context manager and start capturing activations."""
        # Register hooks to capture activations
        for name, hp in self.model.hook_dict.items():
            if self.hook_filter is None or self.hook_filter(name):
                handle = hp.add_hook(self._make_capture_hook(name))
                self._hook_handles.append(handle)

        self._current_step_cache: dict[str, torch.Tensor] = {}
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up hooks."""
        # Remove all hooks
        for handle in self._hook_handles:
            if handle is not None:
                handle.remove()
        self._hook_handles.clear()

    def _make_capture_hook(self, hook_name: str) -> Callable:
        """Create a hook function that captures activations."""

        def capture_hook(activation: torch.Tensor, hook: HookPoint) -> torch.Tensor:  # noqa: ARG001
            # Store a copy of the activation (detached from computation graph)
            self._current_step_cache[hook_name] = activation.detach().cpu()
            return activation

        return capture_hook

    def start_step(self) -> None:
        """Mark the start of a new agent step."""
        self._current_step_cache = {}

    def end_step(self) -> None:
        """Mark the end of an agent step and save captured activations."""
        if self._current_step_cache:
            # Convert dict to ActivationCache
            cache = ActivationCache(
                self._current_step_cache.copy(),
                self.model,
            )
            self.step_activations.append(cache)
            self._current_step_cache = {}

    def record_step_metadata(self, metadata: dict[str, Any]) -> None:
        """Record metadata for the current/last step.

        Args:
            metadata: Dictionary of metadata to associate with this step.
        """
        self.step_metadata.append(metadata)

    def get_trajectory(self) -> TrajectoryActivations:
        """Get the complete trajectory of captured activations.

        Returns:
            TrajectoryActivations containing all captured steps.
        """
        return TrajectoryActivations(
            step_activations=self.step_activations.copy(),
            step_metadata=self.step_metadata.copy(),
            model_name=self.model.cfg.model_name,
        )

    def clear(self) -> None:
        """Clear all captured activations and metadata."""
        self.step_activations.clear()
        self.step_metadata.clear()
        self._current_step_cache = {}


def automatic_activation_capture(model: HookedTransformer) -> ActivationCapture:
    """Create an ActivationCapture that automatically records steps during generation.

    This wraps the model's generate method to automatically call start_step() and
    end_step() around each generation, making it seamless to use with ARES environments.

    Args:
        model: HookedTransformer to capture activations from.

    Returns:
        ActivationCapture instance with automatic step tracking.

    Example:
        ```python
        model = HookedTransformer.from_pretrained("gpt2-small")

        with automatic_activation_capture(model) as capture:
            client = HookedTransformerLLMClient(model=model)
            # Now activations are captured automatically during client calls
            async with env:
                ts = await env.reset()
                while not ts.last():
                    response = await client(ts.observation)
                    ts = await env.step(response)

        trajectory = capture.get_trajectory()
        ```
    """
    capture = ActivationCapture(model)

    # Wrap model.generate to auto-capture
    original_generate = model.generate

    def wrapped_generate(*args, **kwargs):
        capture.start_step()
        result = original_generate(*args, **kwargs)
        capture.end_step()
        return result

    model.generate = wrapped_generate

    return capture
