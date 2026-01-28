"""Textual TUI visualization for parallel task evaluation.

EXPERIMENTAL: This module is mostly vibe-coded and experimental, but included in case
it's useful for other eval scripts. The API may change without notice.

This module provides a terminal-based UI for monitoring parallel task execution
with real-time statistics, logs, and progress tracking.

Dependencies:
    This module requires optional dependencies from the 'eval-viz' group:
    Working from source:
        uv sync --group eval-viz
    or externally:
        uv add martian-ares[eval-viz]
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
from typing import ClassVar

import rich.markup
import rich.text
from textual import app
from textual import containers
from textual import widgets

from ares.environments import base
from ares.llms import llm_clients
from ares.llms import request


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
        env: base.Environment[llm_clients.LLMResponse, request.LLMRequest, RewardType, DiscountType],
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

    async def reset(self) -> base.TimeStep[request.LLMRequest, RewardType, DiscountType]:
        """Reset the environment and update dashboard."""
        self._dashboard.update_task(self._task_id, status=TaskStatus.RUNNING, log="Resetting environment")
        ts = await self._env.reset()
        self._dashboard.update_task(self._task_id, current_step=0, log="Environment ready, starting episode")
        self._step_count = 0
        self._total_cost = 0.0
        return ts

    async def step(
        self, action: llm_clients.LLMResponse
    ) -> base.TimeStep[request.LLMRequest, RewardType, DiscountType]:
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
    - Task detail pane (shows error messages when you select an errored task)
    - Histogram of current agent steps
    - Scrollable logs for captured output

    Keyboard shortcuts:
    - Arrow keys: Scroll through tasks and logs, select task rows
    - p: Pause/resume rendering
    - q: Quit the dashboard
    """

    BINDINGS: ClassVar = [("p", "toggle_pause", "Pause/Resume")]

    CSS = """
    #summary {
        height: 10;
        border: solid #4a9eff;
        background: #0a1628;
    }

    #tasks-container {
        height: 2fr;
        border: solid #4a9eff;
        background: #0a1628;
    }

    #task-detail {
        height: 8;
        border: solid #e76f51;
        background: #0a1628;
    }

    #histogram {
        height: 18;
        border: solid #4a9eff;
        background: #0a1628;
    }

    #logs {
        height: 1fr;
        border: solid #4a9eff;
        background: #0a1628;
    }

    .left-panel {
        width: 2fr;
    }

    .right-panel {
        width: 1fr;
    }

    DataTable {
        background: #0a1628;
    }

    DataTable > .datatable--header {
        background: #1a2844;
        color: #4a9eff;
    }

    DataTable > .datatable--cursor {
        background: #1a3a5f;
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
        self._log_lines: list[rich.text.Text] = []
        self._log_position = 0  # Track how much of log_capture we've processed
        self._seen_task_errors: set[int] = set()  # Track which task errors we've logged

        # For stdout/stderr redirection
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._stdout_capture = StringIO()
        self._stderr_capture = StringIO()

        # App task for background execution
        self._app_task: asyncio.Task[None] | None = None

        # Track if table has been initialized
        self._table_initialized = False

        # Track if rendering is paused
        self._paused = False

        # Track selected task for detail view
        self._selected_task_id: int | None = None

    def compose(self) -> app.ComposeResult:
        """Compose the dashboard layout."""
        yield widgets.Header()
        with containers.Horizontal():
            with containers.Vertical(classes="left-panel"):
                yield widgets.Static(id="summary")
                with containers.Container(id="tasks-container"):
                    yield widgets.DataTable(id="tasks")
                yield widgets.Static(id="task-detail")
            with containers.Vertical(classes="right-panel"):
                yield widgets.Static(id="histogram")
                yield containers.ScrollableContainer(widgets.Static(id="logs-content", markup=False), id="logs")
        yield widgets.Footer()

    def on_mount(self) -> None:
        """Initialize the dashboard when mounted."""
        # Set up the task table columns
        table = self.query_one("#tasks", widgets.DataTable)
        table.add_column("ID", key="id", width=6)
        table.add_column("Status", key="status", width=15)
        table.add_column("Step", key="step", width=6)
        table.add_column("Reward", key="reward", width=8)
        table.add_column("Cost", key="cost", width=8)
        table.add_column("Duration", key="duration", width=10)
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
        self.sub_title = "Press q to quit | p to pause/resume"

        # Initialize task detail widget
        self._update_task_detail()

        # Now that Textual is mounted, redirect stdout/stderr to prevent interference
        self._redirect_output()

    def action_toggle_pause(self) -> None:
        """Toggle pause state for rendering."""
        self._paused = not self._paused
        if self._paused:
            self.sub_title = "PAUSED - Press p to resume | q to quit"
        else:
            self.sub_title = "Press q to quit | p to pause/resume"
        # Trigger immediate refresh to update the pause indicator
        self._update_summary()

    def on_data_table_row_selected(self, event: widgets.DataTable.RowSelected) -> None:
        """Handle task row selection."""
        # Extract task ID from the row key
        if event.row_key.value is not None:
            task_id = int(event.row_key.value)
            self._selected_task_id = task_id
            self._update_task_detail()

    def _update_task_detail(self) -> None:
        """Update the task detail widget."""
        detail_widget = self.query_one("#task-detail", widgets.Static)

        if self._selected_task_id is None:
            detail_widget.update("[dim]Select a task to view details[/dim]")
            return

        task = self.tasks.get(self._selected_task_id)
        if task is None:
            detail_widget.update("[dim]Task not found[/dim]")
            return

        # Build detail text using Text objects to avoid markup parsing issues
        detail_text = rich.text.Text()
        detail_text.append(f"Task {task.task_id} Details", style="bold #4a9eff")
        detail_text.append("\n")

        if task.status == TaskStatus.ERROR and task.error:
            detail_text.append("Error: ", style="bold #e76f51")
            detail_text.append(str(task.error), style="#e76f51")
        elif task.status == TaskStatus.COMPLETED:
            detail_text.append("Completed successfully", style="#2a9d8f")
            detail_text.append(f" - Reward: {task.reward}")
        elif task.status == TaskStatus.RUNNING:
            detail_text.append("Running", style="#f4a261")
            detail_text.append(f" - Step: {task.current_step}")
        else:
            detail_text.append("Waiting to start", style="dim")

        detail_widget.update(detail_text)

    def _refresh_display(self) -> None:
        """Refresh all dashboard widgets."""
        # Always update summary to show pause status
        self._update_summary()

        if self._paused:
            return

        self._update_task_table()
        self._update_histogram()
        self._update_logs()
        self._update_task_detail()

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
        pause_indicator = " [bold #f4a261]⏸ PAUSED[/bold #f4a261]" if self._paused else ""
        summary_lines = [
            f"[bold #4a9eff]Summary[/bold #4a9eff] [dim](Elapsed: {elapsed:.1f}s)[/dim]{pause_indicator}",
            "",
            f"Running: [#f4a261]{running}[/#f4a261]",
            f"Completed: [#2a9d8f]{completed}[/#2a9d8f] / {self.total_tasks}",
            f"Errors: [#e76f51]{errors}[/#e76f51]",
            f"Success Rate: [#4a9eff]{success_rate:.1f}%[/#4a9eff] [dim]({success_count}/{completed})[/dim]",
            f"Avg Return: [#4a9eff]{avg_return:.4f}[/#4a9eff]",
            f"Total Cost: [#bc6ff1]${total_cost:.4f}[/#bc6ff1]",
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

        # If first time, add all rows with keys
        if not self._table_initialized:
            for task in sorted_tasks:
                table.add_row(
                    f"[dim]{task.task_id}[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    "[dim]-[/dim]",
                    key=str(task.task_id),
                )
            self._table_initialized = True

        # Update existing rows
        for task in sorted_tasks:
            row_key = str(task.task_id)

            # Status with color
            status_map = {
                TaskStatus.WAITING: ("⏳", "#6c757d"),
                TaskStatus.RUNNING: ("🔄", "#f4a261"),
                TaskStatus.COMPLETED: ("✓", "#2a9d8f"),
                TaskStatus.ERROR: ("✗", "#e76f51"),
            }
            status_icon, status_color = status_map[task.status]
            status_str = f"[{status_color}]{status_icon} {task.status.value}[/{status_color}]"

            # Step
            step_str = str(task.current_step) if task.current_step > 0 else "[dim]-[/dim]"

            # Reward
            if task.reward is not None:
                reward_float = float(task.reward)  # type: ignore
                reward_color = "#2a9d8f" if reward_float > 0 else "#e76f51"
                reward_str = f"[{reward_color}]{reward_float:.2f}[/{reward_color}]"
            else:
                reward_str = "[dim]-[/dim]"

            # Cost
            cost_str = f"[#bc6ff1]${task.cost:.2f}[/#bc6ff1]" if task.cost > 0 else "[dim]-[/dim]"

            # Duration
            duration_str = f"[dim]{task.duration:.1f}s[/dim]" if task.duration is not None else "[dim]-[/dim]"

            # Update all cells in this row
            table.update_cell(row_key, "status", status_str)
            table.update_cell(row_key, "step", step_str)
            table.update_cell(row_key, "reward", reward_str)
            table.update_cell(row_key, "cost", cost_str)
            table.update_cell(row_key, "duration", duration_str)

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
        histogram_lines = [
            f"[bold #4a9eff]Step Distribution[/bold #4a9eff] [dim]({len(running_tasks)} running)[/dim]",
            "",
        ]

        for i in range(num_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size - 1
            count = buckets.get(i, 0)

            if count == 0 and i > 0 and all(buckets.get(j, 0) == 0 for j in range(i, num_buckets)):
                # Skip empty trailing buckets
                break

            # Create bar with gradient effect
            bar_width = 20
            filled = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = f"[#4a9eff]{'█' * filled}[/#4a9eff][dim]{'░' * (bar_width - filled)}[/dim]"

            range_str = f"{start:3d}-{end:3d}"
            histogram_lines.append(f"[dim]{range_str}[/dim] {bar} [bold #f4a261]{count}[/bold #f4a261]")

        histogram_widget.update("\n".join(histogram_lines))

    def _update_logs(self) -> None:
        """Update the logs widget."""
        # Get only new log content since last position
        log_content = self._log_capture.getvalue()
        new_content = log_content[self._log_position :]
        self._log_position = len(log_content)

        # Parse new log lines - only WARNING and worse
        if new_content:
            for line in new_content.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Only include WARNING, ERROR, CRITICAL logs
                # Use Text objects to safely add color without markup parsing issues
                if "ERROR" in line or "CRITICAL" in line:
                    text = rich.text.Text(line, style="#e76f51")
                    self._log_lines.append(text)
                elif "WARNING" in line:
                    text = rich.text.Text(line, style="#f4a261")
                    self._log_lines.append(text)
                # Skip INFO and DEBUG logs

        # Add task error logs (only new ones)
        for task in self.tasks.values():
            if task.status == TaskStatus.ERROR and task.error and task.task_id not in self._seen_task_errors:
                # Ensure task.error is a string
                error_msg = str(task.error) if task.error else "Unknown error"
                # Create styled Text object
                text = rich.text.Text()
                text.append(f"Task {task.task_id} ERROR: ", style="bold #e76f51")
                text.append(error_msg, style="#e76f51")
                self._log_lines.append(text)
                self._seen_task_errors.add(task.task_id)

        # Keep only the last 1000 lines to avoid memory issues
        if len(self._log_lines) > 1000:
            self._log_lines = self._log_lines[-1000:]

        if not self._log_lines:
            log_text = rich.text.Text("No warnings or errors yet", style="dim")
        else:
            # Assemble Text objects with newlines between them
            log_text = rich.text.Text()
            for i, line in enumerate(self._log_lines):
                if i > 0:
                    log_text.append("\n")
                log_text.append_text(line)

        logs_widget = self.query_one("#logs-content", widgets.Static)
        logs_widget.update(log_text)

        # Auto-scroll to bottom to show latest logs
        logs_container = self.query_one("#logs", containers.ScrollableContainer)
        logs_container.scroll_end(animate=False)

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

        # Suppress noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

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
        env: base.Environment[llm_clients.LLMResponse, request.LLMRequest, RewardType, DiscountType],
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

    async def __aexit__(self, *_args):
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
