"""Example demonstrating environment state snapshotting and restoration.

This example shows how to:
1. Create a snapshot after reset (at episode boundary)
2. Save the snapshot to disk
3. Restore an environment from a saved snapshot
4. Continue execution from the restored state

Example usage:

    1. Make sure you have examples dependencies installed
       `uv sync --group examples`
    2. Run the example
       `uv run -m examples.03_state_snapshotting`
"""

import asyncio
import pathlib
import tempfile

from ares.code_agents import mini_swe_agent
from ares.containers import docker
from ares.environments import snapshot
from ares.environments import swebench_env
from ares.llms import chat_completions_compatible


async def main():
    # Create an LLM client
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="openai/gpt-4o-mini")

    # Load SWE-bench tasks
    all_tasks = swebench_env.swebench_verified_tasks()
    tasks = [all_tasks[0]]

    print(f"Running on task: {tasks[0].instance_id}")
    print(f"Repository: {tasks[0].repo}")
    print("-" * 80)

    # Create a temporary directory for snapshots
    with tempfile.TemporaryDirectory() as snapshot_dir:
        snapshot_path = pathlib.Path(snapshot_dir)

        # === PART 1: Create and save a snapshot ===
        print("\n[PART 1] Creating initial environment and snapshot...")

        async with swebench_env.SweBenchEnv(
            tasks=tasks,
            code_agent_factory=mini_swe_agent.MiniSWECodeAgent,
            container_factory=docker.DockerContainer,
        ) as env:
            # Reset the environment to get the first timestep
            ts = await env.reset()
            print(f"Environment reset complete. Step count: {env._step_count}")

            # Take a few steps before snapshotting
            for i in range(3):
                action = await agent(ts.observation)
                print(f"  Step {i}: Taking action...")
                ts = await env.step(action)

                if ts.last():
                    print("  Episode completed early")
                    break

            print(f"Current step count: {env._step_count}")

            # Wait for agent to finish current operation (reach episode boundary)
            # In practice, you'd snapshot after step() returns with done=True
            # or after reset() completes. For this example, we'll simulate
            # waiting for agent to finish.
            if not ts.last():
                print("\n  Note: For snapshotting, we need to be at episode boundary.")
                print("  Cancelling agent task to reach boundary...")
                if env._code_agent_task and not env._code_agent_task.done():
                    env._code_agent_task.cancel()
                    import contextlib

                    with contextlib.suppress(asyncio.CancelledError):
                        await env._code_agent_task

            # Now we can export state (at episode boundary)
            print("\n  Exporting state snapshot...")
            snap = await env.export_state(snapshot_path, snapshot_id="example-snapshot")

            print(f"  ✓ Snapshot created: {snap.snapshot_id}")
            print(f"  ✓ Snapshot saved to: {snap.snapshot_dir}")
            print(f"  ✓ Step count in snapshot: {snap.step_count}")
            print(f"  ✓ Task type: {snap.task_type}")
            print(f"  ✓ Container type: {snap.container_type}")

        # === PART 2: Restore from snapshot ===
        print("\n[PART 2] Restoring environment from snapshot...")

        # Load snapshot metadata
        snapshot_file = snapshot_path / "example-snapshot" / "snapshot.json"
        loaded_snap = snapshot.EnvironmentSnapshot.load_from_file(snapshot_file)

        print(f"  ✓ Loaded snapshot: {loaded_snap.snapshot_id}")
        print(f"  ✓ Original step count: {loaded_snap.step_count}")

        # Restore environment from snapshot
        # Note: This creates a new environment instance with the saved state
        restored_env = await swebench_env.SweBenchEnv.load_from_state(
            loaded_snap,
            container_factory=docker.DockerContainer,
            code_agent_factory=mini_swe_agent.MiniSWECodeAgent,
        )

        print("  ✓ Environment restored")
        print(f"  ✓ Restored step count: {restored_env._step_count}")
        print(f"  ✓ Task: {restored_env._current_task.instance_id}")

        # Use the restored environment in async context
        async with restored_env:
            print("\n[PART 3] Continuing from restored state...")

            # The environment is now at the same state as when we snapshotted
            # We can continue taking steps from here
            ts = await restored_env.reset()  # Reset to start a new episode
            step_count = 0

            # Take a few more steps to demonstrate it works
            while not ts.last() and step_count < 3:
                action = await agent(ts.observation)
                print(f"  Step {step_count}: Taking action from restored env...")
                ts = await restored_env.step(action)
                step_count += 1

            print(f"\n  ✓ Completed {step_count} additional steps from restored state")

        print("\n" + "=" * 80)
        print("Snapshot example completed successfully!")
        print("=" * 80)
        print("\nKey takeaways:")
        print("  1. Snapshots can only be taken at episode boundaries")
        print("  2. Snapshots save: task state, container filesystem, agent messages")
        print("  3. Restored environments can continue execution normally")
        print("  4. Use cases: debugging, RL replay, mechanistic interpretability")


if __name__ == "__main__":
    asyncio.run(main())
