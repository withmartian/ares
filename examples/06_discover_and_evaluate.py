"""Discover available tasks and run an interactive evaluation.

This example shows ARES's task discovery capabilities and lets you
interactively explore and evaluate tasks from different datasets.

Prerequisites:

    - `CHAT_COMPLETION_API_KEY` set in `.env` with a Martian API key.
       To get a Martian API key:
       1) sign up at https://app.withmartian.com.
       2) on the `Billing` tab, add a payment method + top up some credits.
       3) on the `API Keys` tab create an API key.
    - Install dependencies: `uv sync --group examples`

Example usage:

    # Interactive mode - prompts you to choose a preset
    uv run -m examples.06_discover_and_evaluate

    # Non-interactive mode - specify preset directly
    uv run -m examples.06_discover_and_evaluate --preset vmax-tasks-mswea --task-idx 0

    # Quick mode - run immediately without prompts
    uv run -m examples.06_discover_and_evaluate --preset vmax-tasks-mswea --quick
"""

import argparse
import asyncio
import sys

import ares
from ares import config
from ares.llms import chat_completions_compatible

from . import utils


def discover_presets() -> None:
    """Discover and display available env presets."""
    print("🔍 Discovering available environment presets in ARES...\n")

    # Get all available presets
    all_presets = ares.info()

    # Group presets by base dataset (remove agent suffix)
    datasets = {}
    for preset in all_presets:
        # Extract base dataset name (e.g., "vmax-tasks-mswea" -> "vmax-tasks")
        base_name = preset.name.rsplit("-", 1)[0]
        if base_name not in datasets:
            datasets[base_name] = []
        datasets[base_name].append(preset)

    print(f"Found {len(all_presets)} total presets across {len(datasets)} datasets\n")
    print("=" * 80)


def show_featured_presets() -> list[str]:
    """Show a curated list of featured/interesting presets.

    Returns:
        List of featured preset names.
    """
    # Curate a list of interesting presets to highlight
    featured_names = [
        "sbv-mswea",  # SWE-bench Verified (500 tasks)
        "tbench-mswea",  # Terminal-bench
        "vmax-tasks-mswea",  # Colaunched with ARES - 1000 tasks with more on the way
    ]

    print("\n📌 Featured Presets:\n")
    featured_presets = []

    for i, name in enumerate(featured_names, 1):
        info = ares.info(name)
        featured_presets.append(name)
        marker = "⭐" if name == "vmax-tasks-mswea" else "  "
        print(f"{marker} {i}. {info.name}")
        if name == "vmax-tasks-mswea":
            print("     └─ 🚀 Colaunched with ARES: 1000 tasks, with more on the way!")
        else:
            print(f"     └─ {info.num_tasks} tasks | {info.description}")

    print("\n💡 To list all presets in your own code, simply run: \"import ares; ares.list_presets()\"")
    return featured_presets


def list_all_presets() -> None:
    """Display all available presets grouped by dataset."""
    print(ares.list_presets())

    print("\n" + "=" * 80)
    print("💡 Type the full preset name (e.g., 'vmax-tasks-mswea') to select it")


def prompt_preset_selection(featured_presets: list[str]) -> str:
    """Prompt user to select a preset interactively.

    Args:
        featured_presets: List of featured preset names to choose from.

    Returns:
        Selected preset name.
    """
    print("\n" + "=" * 80)
    print("\n🎯 Select a preset to evaluate:\n")
    print(f"   • 1-{len(featured_presets)}: Choose a featured preset")
    print("   • 'list': See all available presets")
    print("   • [Enter]: Use default (vmax-tasks-mswea)")
    print("   • Or type any preset name directly")

    while True:
        choice = input("\n> ").strip()

        # Default to vmax-tasks-mswea
        if not choice:
            return "vmax-tasks-mswea"

        # Handle 'list' command to show all presets
        if choice.lower() == "list":
            list_all_presets()
            continue

        # Check if it's a number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(featured_presets):
                return featured_presets[idx]
            print(f"❌ Please enter a number between 1 and {len(featured_presets)}")
            continue

        # Check if it's a valid preset name
        try:
            ares.info(choice)
            return choice
        except KeyError:
            print(f"❌ Preset '{choice}' not found. Try again or press Enter for default.")


def prompt_task_selection(preset_name: str) -> int:
    """Prompt user to select a task index.

    Args:
        preset_name: The preset to get task info from.

    Returns:
        Selected task index.
    """
    info = ares.info(preset_name)
    print(f"\n📋 {preset_name} has {info.num_tasks} tasks available")
    print("   Enter a task index (0-based) or press Enter for task 0")

    while True:
        choice = input("\nTask index: ").strip()

        # Default to task 0
        if not choice:
            return 0

        # Validate the index
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < info.num_tasks:
                return idx
            print(f"❌ Please enter an index between 0 and {info.num_tasks - 1}")
        else:
            print("❌ Please enter a valid number")


async def run_evaluation(preset_name: str, task_idx: int) -> None:
    """Run evaluation on the selected preset and task.

    Args:
        preset_name: The preset to evaluate.
        task_idx: The task index to run.
    """
    # Create an LLM client
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="openai/gpt-5-mini")

    print("\n" + "=" * 80)
    print(f"\n🚀 Starting evaluation: {preset_name}:{task_idx}\n")
    print("=" * 80)

    # Create environment with the selected task
    async with ares.make(f"{preset_name}:{task_idx}") as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0
        total_reward = 0.0

        # Continue until the episode is done
        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)

            # Print the observation and action
            utils.print_step(step_count, ts.observation, action)

            # Step the environment with the action
            ts = await env.step(action)

            assert ts.reward is not None
            total_reward += ts.reward
            step_count += 1

        # Display final results
        print(f"\n{'=' * 80}")
        print(f"✅ Episode completed after {step_count} steps")
        print(f"🎯 Total reward: {total_reward}")
        if total_reward > 0:
            print("🎉 Task solved successfully!")
        else:
            print("❌ Task not solved - the agent didn't pass all tests")
        print(f"{'=' * 80}")


async def main():
    """Main entry point for the discovery and evaluation example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Discover and evaluate ARES tasks interactively")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Preset name to evaluate (skips interactive selection)",
    )
    parser.add_argument(
        "--task-idx",
        type=int,
        default=None,
        help="Task index to evaluate (skips interactive selection)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip all prompts and run immediately with defaults",
    )
    args = parser.parse_args()

    # Fail fast if env vars aren't set
    if not config.CONFIG.chat_completion_api_key:
        raise ValueError("CHAT_COMPLETION_API_KEY is not set")

    # If --quick mode, run with defaults immediately
    if args.quick:
        preset_name = args.preset or "vmax-tasks-mswea"
        task_idx = args.task_idx if args.task_idx is not None else 0
        print(f"🚀 Quick mode: Running {preset_name}:{task_idx}")
        await run_evaluation(preset_name, task_idx)
        return

    # Interactive mode
    discover_presets()
    featured_presets = show_featured_presets()

    # Determine preset (from args or interactive prompt)
    if args.preset:
        preset_name = args.preset
        print(f"\n✓ Using preset from command line: {preset_name}")
    else:
        preset_name = prompt_preset_selection(featured_presets)

    print(f"\n✓ Selected preset: {preset_name}")

    # Determine task index (from args or interactive prompt)
    if args.task_idx is not None:
        task_idx = args.task_idx
        print(f"✓ Using task index from command line: {task_idx}")
    else:
        task_idx = prompt_task_selection(preset_name)

    print(f"✓ Selected task index: {task_idx}")

    # Confirmation before running
    print("\n" + "=" * 80)
    print(f"\n📝 Ready to evaluate: {preset_name}:{task_idx}")
    response = input("Press Enter to start evaluation (or 'q' to quit): ").strip().lower()

    if response == "q":
        print("👋 Evaluation cancelled")
        sys.exit(0)

    # Run the evaluation
    await run_evaluation(preset_name, task_idx)

    # Next steps
    print("\n💡 Next steps:")
    print(f"   - Try another task: uv run -m examples.06_discover_and_evaluate --preset {preset_name}")
    print("   - Run parallel evaluation: uv run -m examples.03_parallel_eval_with_api")
    print("   - Explore all presets: python -c \"import ares; print(ares.list_presets())\"")


if __name__ == "__main__":
    asyncio.run(main())
