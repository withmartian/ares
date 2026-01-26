"""Environment state snapshotting for RL research and mechanistic interpretability.

This module provides functionality to snapshot and restore environment state,
enabling use cases like:
- RL trajectory replay and analysis
- Debugging failed episodes
- Mechanistic interpretability of agent behavior
- Checkpointing long-running experiments

## Usage Example

```python
import pathlib
from ares.environments import swebench_env, snapshot

# Create and run environment
async with swebench_env.SweBenchEnv(tasks=[task]) as env:
    ts = await env.reset()

    # Take some steps
    for _ in range(10):
        action = await agent(ts.observation)
        ts = await env.step(action)
        if ts.last():
            break

    # Wait for episode to complete (required for snapshotting)
    if env._code_agent_task and not env._code_agent_task.done():
        env._code_agent_task.cancel()
        await env._code_agent_task

    # Export state at episode boundary
    snap = await env.export_state(pathlib.Path("./snapshots"))

# Later: restore from snapshot
loaded_snap = snapshot.EnvironmentSnapshot.load_from_file(
    pathlib.Path("./snapshots/abc-123/snapshot.json")
)

restored_env = await swebench_env.SweBenchEnv.load_from_state(loaded_snap)
async with restored_env:
    # Continue from saved state
    ts = await restored_env.reset()
    ...
```

## Limitations

- **Episode boundaries only**: Snapshots can only be created when no code agent
  task is running (after reset() or after final step() with done=True)
- **No mid-execution state**: Agent message history is saved, but not mid-execution
  state like running asyncio tasks or futures
- **Large snapshots**: Container filesystems are saved as tarballs (100MB-2GB typical)
- **Container restoration**: Containers are recreated from original images and
  filesystem is restored from tarball, not from running container state

## What Gets Snapshotted

Serializable state:
- Step count and step limit
- Task metadata (serialized Pydantic models or paths)
- Container metadata (image, dockerfile, resources)
- Agent message history

Non-serializable state (cannot snapshot):
- Running asyncio tasks and futures
- Active LLM request queues
- Live container connections
"""

import dataclasses
import json
import pathlib
from typing import Literal


@dataclasses.dataclass(frozen=True)
class EnvironmentSnapshot:
    """Complete environment state snapshot.

    Can only be created at episode boundaries:
    - After env.reset() completes (FIRST timestep)
    - After env.step() returns LAST timestep (done=True)
    - When no code agent task is running

    Limitations:
    - Snapshots only at episode boundaries (after reset or final step)
    - Cannot snapshot mid-episode (running async tasks/futures)
    - Agent message history preserved, but not mid-execution state
    - Large filesystem snapshots (100MB-2GB tarballs)
    """

    # Unique identifier and metadata
    snapshot_id: str
    created_at: str  # ISO timestamp
    snapshot_dir: pathlib.Path

    # Episode state
    step_count: int
    step_limit: int
    requires_reset: bool

    # Task metadata (for reconstruction)
    task_type: Literal["swebench", "harbor"]
    task_data: dict  # Serialized task (Pydantic model_dump or Harbor path)

    # Container metadata
    container_type: Literal["daytona", "docker"]
    container_image: str | None
    container_dockerfile_path: str | None
    container_resources: dict | None  # Serialized Resources

    # Code agent state
    agent_messages: list[dict]  # Chat history from MiniSWECodeAgent._messages

    def save_to_file(self, path: pathlib.Path) -> None:
        """Save snapshot metadata to JSON file.

        Args:
            path: Path to save JSON file (typically snapshot_dir/snapshot.json)
        """
        # Convert pathlib.Path to string for JSON serialization
        data = dataclasses.asdict(self)
        data["snapshot_dir"] = str(data["snapshot_dir"])

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: pathlib.Path) -> "EnvironmentSnapshot":
        """Load snapshot metadata from JSON file.

        Args:
            path: Path to JSON file (typically snapshot_dir/snapshot.json)

        Returns:
            EnvironmentSnapshot instance
        """
        with open(path) as f:
            data = json.load(f)

        # Convert string back to pathlib.Path
        data["snapshot_dir"] = pathlib.Path(data["snapshot_dir"])

        return cls(**data)
