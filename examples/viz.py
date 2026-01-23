"""Textual TUI visualization for parallel task evaluation.

This module provides a terminal-based UI for monitoring parallel task execution
with real-time statistics, logs, and progress tracking.
"""

import asyncio
from collections import defaultdict
import contextlib
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from io import StringIO
import logging
import sys
import time

from textual import app
from textual import containers
from textual import widgets

from ares.environments import base
from ares.llms import llm_clients


class TaskStatus(Enum):
    """Status of a task."""

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TaskInfo:
    """Information about a single task."""

    task_id: int
    status: TaskStatus = TaskStatus.WAITING
    current_step: int = 0
    reward: base.Scalar | None = None
    cost: float = 0.0
    start_time: float | None = None
    end_time: float | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float | None:
        """Get the duration of the task in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time


class TrackedEnvironment[RewardType: base.Scalar, DiscountType: base.Scalar]:
    """Wrapper around an ARES LLM environment that automatically tracks state and reports to dashboard.

    This wrapper is specifically designed for environments that use LLMResponse as actions
    and LLMRequest as observations. It automatically tracks LLM costs and step progress.

    This wrapper intercepts reset() and step() calls to automatically update the dashboard
    with progress information, eliminating the need for manual instrumentation.

    Type parameters: [RewardType, DiscountType] (ActionType and ObservationType are fixed to LLM types)

    Usage:
        async with dashboard.wrap(task_id, ares.make(...)) as env:
            ts = await env.reset()
            while not ts.last():
                action = agent(ts.observation)
                ts = await env.step(action)
    """

    def __init__(
        self,
        env: base.Environment[llm_clients.LLMResponse, llm_clients.LLMRequest, RewardType, DiscountType],
        task_id: int,
        dashboard: "EvaluationDashboard",
    ):
        """Initialize the tracked environment wrapper.

        Args:
            env: The environment to wrap (must use LLMRequest/LLMResponse).
            task_id: The task ID for dashboard tracking.
            dashboard: The dashboard to report to.
        """
        self._env = env
        self._task_id = task_id
        self._dashboard = dashboard
        self._step_count = 0
        self._total_cost = 0.0

    async def reset(self) -> base.TimeStep[llm_clients.LLMRequest, RewardType, DiscountType]:
        """Reset the environment and update dashboard."""
        self._dashboard.update_task(self._task_id, status=TaskStatus.RUNNING, log="Resetting environment")
        ts = await self._env.reset()
        self._dashboard.update_task(self._task_id, current_step=0, log="Environment ready, starting episode")
        self._step_count = 0
        self._total_cost = 0.0
        return ts

    async def step(
        self, action: llm_clients.LLMResponse
    ) -> base.TimeStep[llm_clients.LLMRequest, RewardType, DiscountType]:
        """Step the environment and update dashboard."""
        self._step_count += 1

        # Track LLM cost from the action
        self._total_cost += action.cost

        ts = await self._env.step(action)

        # Update dashboard with current progress
        self._dashboard.update_task(
            self._task_id,
            current_step=self._step_count,
            cost=self._total_cost,
            log=f"Step {self._step_count} completed",
        )

        # If episode is done, mark as completed
        if ts.last():
            self._dashboard.update_task(
                self._task_id,
                status=TaskStatus.COMPLETED,
                reward=ts.reward,
                log=f"Completed with reward {ts.reward}",
            )

        return ts

    async def __aenter__(self):
        """Enter the environment context."""
        await self._env.__aenter__()
        self._dashboard.update_task(self._task_id, log="Environment setup complete")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the environment context."""
        # If there was an exception, mark the task as error
        if exc_type is not None:
            error_msg = f"{exc_type.__name__}: {str(exc_val)[:100]}"
            self._dashboard.update_task(
                self._task_id,
                status=TaskStatus.ERROR,
                error=error_msg,
                log=f"Failed: {error_msg}",
            )

        return await self._env.__aexit__(exc_type, exc_val, exc_tb)


class EvaluationDashboard(app.App):
    """Real-time dashboard for monitoring parallel task evaluation.

    This dashboard uses Textual to display:
    - Summary statistics (running, completed, errors, success rate, avg return, total cost)
    - Scrollable task status table (showing all tasks with details)
    - Histogram of current agent steps
    - Scrollable logs for captured output

    Keyboard shortcuts:
    - Arrow keys: Scroll through tasks and logs
    - q: Quit the dashboard
    """

    CSS = """
    #summary {
        height: 10;
        border: solid $primary;
    }

    #tasks-container {
        height: 1fr;
        border: solid $primary;
    }

    #histogram {
        height: 18;
        border: solid $primary;
    }

    #logs {
        height: 1fr;
        border: solid $primary;
    }

    .left-panel {
        width: 2fr;
    }

    .right-panel {
        width: 1fr;
    }
    """

    def __init__(
        self,
        total_tasks: int,
        preset_name: str | None = None,
        max_parallel: int | None = None,
    ):
        """Initialize the dashboard.

        Args:
            total_tasks: Total number of tasks to evaluate.
            preset_name: Optional preset name to display in header.
            max_parallel: Optional max parallel workers to display in header.
        """
        super().__init__()
        self.total_tasks = total_tasks
        self.preset_name = preset_name
        self.max_parallel = max_parallel
        self.tasks: dict[int, TaskInfo] = {i: TaskInfo(task_id=i) for i in range(total_tasks)}
        self.start_time = time.time()

        # For capturing logs
        self._original_log_handlers: list[logging.Handler] = []
        self._log_capture = StringIO()
        self._log_lines: list[str] = []

        # For stdout/stderr redirection
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_capture = StringIO()
        self._stderr_capture = StringIO()

        # App task for background execution
        self._app_task: asyncio.Task[None] | None = None

    def compose(self) -> app.ComposeResult:
        """Compose the dashboard layout."""
        yield widgets.Header()
        with containers.Horizontal():
            with containers.Vertical(classes="left-panel"):
                yield widgets.Static(id="summary")
                with containers.Container(id="tasks-container"):
                    yield widgets.DataTable(id="tasks")
            with containers.Vertical(classes="right-panel"):
                yield widgets.Static(id="histogram")
                yield containers.ScrollableContainer(widgets.Static(id="logs-content"), id="logs")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard when mounted."""
        # Set up the task table columns
        table = self.query_one("#tasks", widgets.DataTable)
        table.add_column("ID", key="id")
        table.add_column("Status", key="status")
        table.add_column("Step", key="step")
        table.add_column("Reward", key="reward")
        table.add_column("Cost", key="cost")
        table.add_column("Duration", key="duration")
        table.cursor_type = "row"

        # Set up periodic refresh (4 times per second)
        self.set_interval(0.25, self._refresh_display)

        # Update header with preset info
        header = self.query_one(widgets.Header)
        title_parts = ["ARES Parallel Evaluation"]
        if self.preset_name:
            title_parts.append(f"Preset: {self.preset_name}")
        if self.max_parallel:
            title_parts.append(f"Max Parallel: {self.max_parallel}")
        header.tall = False
        self.title = " | ".join(title_parts)
        self.sub_title = "Press q to quit"

        # Now that Textual is mounted, redirect stdout/stderr to prevent interference
        self._redirect_output()

    def _refresh_display(self) -> None:
        """Refresh all dashboard widgets."""
        self._update_summary()
        self._update_task_table()
        self._update_histogram()
        self._update_logs()

    def _update_summary(self) -> None:
        """Update the summary statistics widget."""
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        errors = sum(1 for t in self.tasks.values() if t.status == TaskStatus.ERROR)
        waiting = sum(1 for t in self.tasks.values() if t.status == TaskStatus.WAITING)

        # Calculate success metrics
        finished_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED and t.reward is not None]
        # Convert rewards to float for comparison (assumes scalar rewards are float-like)
        success_count = sum(1 for t in finished_tasks if t.reward is not None and float(t.reward) > 0)  # type: ignore
        success_rate = (success_count / completed * 100) if completed > 0 else 0.0

        # Calculate average return (only for completed non-error tasks)
        rewards = [float(t.reward) for t in finished_tasks if t.reward is not None]  # type: ignore
        avg_return = sum(rewards) / len(rewards) if rewards else 0.0

        # Calculate total cost
        total_cost = sum(t.cost for t in self.tasks.values())

        # Calculate average time per task
        durations = [
            t.duration for t in self.tasks.values() if t.duration is not None and t.status == TaskStatus.COMPLETED
        ]
        avg_time = sum(durations) / len(durations) if durations else 0.0

        # Estimate time remaining
        tasks_remaining = waiting + running
        has_eta = avg_time > 0 and tasks_remaining > 0
        eta = avg_time * tasks_remaining if has_eta else None

        # Calculate elapsed time
        elapsed = time.time() - self.start_time

        # Build summary text
        summary_lines = [
            f"[bold cyan]Summary[/bold cyan] (Elapsed: {elapsed:.1f}s)",
            "",
            f"Running: [yellow]{running}[/yellow]",
            f"Completed: [green]{completed}[/green] / {self.total_tasks}",
            f"Errors: [red]{errors}[/red]",
            f"Success Rate: [cyan]{success_rate:.1f}%[/cyan] ({success_count}/{completed})",
            f"Avg Return: [cyan]{avg_return:.4f}[/cyan]",
            f"Total Cost: [magenta]${total_cost:.4f}[/magenta]",
        ]
        if eta is not None:
            summary_lines.append(f"ETA: [dim]{eta:.1f}s[/dim]")

        summary_widget = self.query_one("#summary", widgets.Static)
        summary_widget.update("\n".join(summary_lines))

    def _update_task_table(self) -> None:
        """Update the task status table widget."""
        table = self.query_one("#tasks", widgets.DataTable)

        # Sort tasks by ID
        sorted_tasks = sorted(self.tasks.values(), key=lambda t: t.task_id)

        # Clear and rebuild table rows
        table.clear()

        for task in sorted_tasks:
            # Status with color
            status_map = {
                TaskStatus.WAITING: ("⏳", "dim"),
                TaskStatus.RUNNING: ("🔄", "yellow"),
                TaskStatus.COMPLETED: ("✓", "green"),
                TaskStatus.ERROR: ("✗", "red"),
            }
            status_icon, status_style = status_map[task.status]
            status_str = f"[{status_style}]{status_icon} {task.status.value}[/{status_style}]"

            # Step
            step_str = str(task.current_step) if task.current_step > 0 else "-"

            # Reward
            if task.reward is not None:
                reward_float = float(task.reward)  # type: ignore
                reward_style = "green" if reward_float > 0 else "red"
                reward_str = f"[{reward_style}]{reward_float:.2f}[/{reward_style}]"
            else:
                reward_str = "-"

            # Cost
            cost_str = f"${task.cost:.3f}" if task.cost > 0 else "-"

            # Duration
            duration_str = f"{task.duration:.1f}s" if task.duration is not None else "-"

            table.add_row(
                str(task.task_id),
                status_str,
                step_str,
                reward_str,
                cost_str,
                duration_str,
            )

    def _update_histogram(self) -> None:
        """Update the histogram widget."""
        # Get steps for all running tasks
        running_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING and t.current_step > 0]

        histogram_widget = self.query_one("#histogram", widgets.Static)

        if not running_tasks:
            histogram_widget.update("[bold cyan]Step Distribution[/bold cyan]\n\n[dim]No running tasks[/dim]")
            return

        # Create histogram buckets
        steps = [t.current_step for t in running_tasks]
        max_step = max(steps) if steps else 1
        num_buckets = 10
        bucket_size = max(1, (max_step + num_buckets - 1) // num_buckets)

        # Count tasks in each bucket
        buckets: dict[int, int] = defaultdict(int)
        for step in steps:
            bucket_idx = min(step // bucket_size, num_buckets - 1)
            buckets[bucket_idx] += 1

        max_count = max(buckets.values()) if buckets else 1

        # Build histogram text
        histogram_lines = [f"[bold cyan]Step Distribution[/bold cyan] ({len(running_tasks)} running)", ""]

        for i in range(num_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size - 1
            count = buckets.get(i, 0)

            if count == 0 and i > 0 and all(buckets.get(j, 0) == 0 for j in range(i, num_buckets)):
                # Skip empty trailing buckets
                break

            # Create bar
            bar_width = 20
            filled = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)

            range_str = f"{start:3d}-{end:3d}"
            histogram_lines.append(f"[cyan]{range_str}[/cyan] {bar} [bold]{count}[/bold]")

        histogram_widget.update("\n".join(histogram_lines))

    def _update_logs(self) -> None:
        """Update the logs widget."""
        # Get captured logging output and parse it
        log_content = self._log_capture.getvalue()
        if log_content and log_content not in self._log_lines:
            # Parse new log lines
            for line in log_content.strip().split("\n"):
                if line.strip() and line not in self._log_lines:
                    # Color code by log level
                    if "ERROR" in line:
                        self._log_lines.append(f"[red]{line}[/red]")
                    elif "WARNING" in line:
                        self._log_lines.append(f"[yellow]{line}[/yellow]")
                    else:
                        self._log_lines.append(f"[dim]{line}[/dim]")

        # Add task error logs
        for task in self.tasks.values():
            if task.status == TaskStatus.ERROR and task.error:
                error_line = f"[red bold][Task {task.task_id} ERROR][/red bold] {task.error}"
                if error_line not in self._log_lines:
                    self._log_lines.append(error_line)

        # Keep only the last 1000 lines to avoid memory issues
        if len(self._log_lines) > 1000:
            self._log_lines = self._log_lines[-1000:]

        log_text = "[dim]No logs yet[/dim]" if not self._log_lines else "\n".join(self._log_lines)

        logs_widget = self.query_one("#logs-content", widgets.Static)
        logs_widget.update(log_text)

    def _redirect_output(self) -> None:
        """Redirect stdout, stderr, and logging to prevent interference with Textual.

        This should be called AFTER Textual has mounted, not before.
        """
        # Redirect stdout and stderr
        sys.stdout = self._stdout_capture
        sys.stderr = self._stderr_capture

        # Redirect logging to our capture buffer
        root_logger = logging.getLogger()
        self._original_log_handlers = root_logger.handlers.copy()

        # Remove all console handlers to prevent output to terminal
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add our own handler that writes to the capture buffer
        log_handler = logging.StreamHandler(self._log_capture)
        log_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.INFO)

    def _restore_output(self) -> None:
        """Restore stdout, stderr, and logging."""
        # Restore stdout and stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        # Restore logging handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        for handler in self._original_log_handlers:
            root_logger.addHandler(handler)

    def update_task(
        self,
        task_id: int,
        status: TaskStatus | None = None,
        current_step: int | None = None,
        reward: base.Scalar | None = None,
        cost: float | None = None,
        error: str | None = None,
        log: str | None = None,
    ) -> None:
        """Update information for a specific task.

        Args:
            task_id: The task ID to update.
            status: New status for the task.
            current_step: Current step number.
            reward: Final reward.
            cost: Accumulated cost.
            error: Error message if task failed.
            log: Log message to append.
        """
        task = self.tasks[task_id]

        if status is not None:
            task.status = status
            if status == TaskStatus.RUNNING and task.start_time is None:
                task.start_time = time.time()
            elif status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
                task.end_time = time.time()

        if current_step is not None:
            task.current_step = current_step

        if reward is not None:
            task.reward = reward

        if cost is not None:
            task.cost = cost

        if error is not None:
            task.error = error

        if log is not None:
            task.logs.append(log)

        # The display will be refreshed automatically by the interval timer

    def wrap[RewardType: base.Scalar, DiscountType: base.Scalar](
        self,
        task_id: int,
        env: base.Environment[llm_clients.LLMResponse, llm_clients.LLMRequest, RewardType, DiscountType],
    ) -> TrackedEnvironment[RewardType, DiscountType]:
        """Wrap an ARES LLM environment with automatic dashboard tracking.

        This method is specifically for environments that use LLMRequest as observations
        and LLMResponse as actions (all ARES code agent environments).

        Args:
            task_id: The task ID for this environment.
            env: The environment to wrap (must use LLMRequest/LLMResponse).

        Returns:
            A tracked environment that automatically updates the dashboard with
            step progress and LLM cost tracking.

        Example:
            async with dashboard.wrap(task_id, ares.make(...)) as env:
                ts = await env.reset()
                while not ts.last():
                    action = agent(ts.observation)
                    ts = await env.step(action)
        """
        return TrackedEnvironment(env, task_id, self)

    async def __aenter__(self):
        """Async context manager entry - starts the dashboard."""
        # Start the app in the background
        # We use create_task so it runs concurrently with the evaluation
        self._app_task = asyncio.create_task(self.run_async())

        # Give the app a moment to mount and redirect output
        await asyncio.sleep(0.2)

        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        """Async context manager exit - stops the dashboard."""
        # Request the app to exit
        self.exit()

        # Wait for the app task to complete
        if self._app_task:
            try:
                await asyncio.wait_for(self._app_task, timeout=2.0)
            except TimeoutError:
                # Force cancel if it doesn't exit cleanly
                self._app_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._app_task

        # Restore output to allow normal printing after dashboard
        self._restore_output()

        return None
