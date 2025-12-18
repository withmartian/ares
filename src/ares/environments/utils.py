"""
Utility functions for SWE-smith environments.

Includes formatters for observations and actions that control how command
execution results are presented to agents.
"""

from collections.abc import Callable
import shlex

from ares.containers import containers

# Type aliases for formatter and parser functions
ObservationFormatter = Callable[[str, str, int], str]
ObservationParser = Callable[[str], str]


async def write_content_to_file_in_container(container: containers.Container, content: str, filename: str) -> None:
    """Write content to a file in the container.

    Args:
        container: The container to write to.
        content: The content to write.
        filename: The filename to write to.
    """
    escaped_content = shlex.quote(content)
    write_result = await container.exec_run(f"printf %s {escaped_content} > {filename}")
    if write_result.exit_code != 0:
        raise RuntimeError(f"Failed to write content to file in container. Exit code: {write_result.exit_code}")
