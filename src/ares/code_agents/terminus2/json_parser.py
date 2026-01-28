"""JSON parser for Terminus 2 agent responses."""

import dataclasses
import json
import logging
import re

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Command:
    """A command to execute in the terminal."""

    keystrokes: str
    duration: float = 1.0  # Default duration in seconds (matches official)


@dataclasses.dataclass(frozen=True)
class ParsedResponse:
    """Parsed response from the agent."""

    commands: list[Command]
    task_complete: bool
    thoughts: str | None = None


class Terminus2JSONParser:
    """Parser for JSON-formatted Terminus 2 responses."""

    def parse(self, response_text: str) -> tuple[ParsedResponse, str | None]:
        """Parse the agent's response.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A tuple of (ParsedResponse, feedback) where feedback is None if parsing
            succeeded, or an error message if parsing failed.
        """
        # Try to extract JSON from code blocks or raw text
        json_match = re.search(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in the text
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return (
                    ParsedResponse(commands=[], task_complete=False),
                    "WARNINGS: Could not find JSON in response. Please provide a valid JSON response.",
                )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Fallback: Use broad regex to extract fields
            _LOGGER.warning("JSON parsing failed: %s. Using regex fallback", e)
            try:
                fallback_parsed = self._parse_with_regex(json_str)
                if fallback_parsed:
                    _LOGGER.info("Successfully recovered data using regex fallback")
                    return fallback_parsed, None
            except Exception as regex_error:
                _LOGGER.warning("Regex fallback also failed: %s", regex_error)

            return (
                ParsedResponse(commands=[], task_complete=False),
                f"WARNINGS: Invalid JSON: {e}. Please provide valid JSON.",
            )

        # Parse commands
        commands = []
        raw_commands = data.get("commands", [])
        if not isinstance(raw_commands, list):
            return (
                ParsedResponse(commands=[], task_complete=False),
                "WARNINGS: 'commands' field must be an array.",
            )

        for i, cmd in enumerate(raw_commands):
            if not isinstance(cmd, dict):
                return (
                    ParsedResponse(commands=[], task_complete=False),
                    f"WARNINGS: Command at index {i} must be an object.",
                )

            keystrokes = cmd.get("keystrokes")
            if not keystrokes or not isinstance(keystrokes, str):
                return (
                    ParsedResponse(commands=[], task_complete=False),
                    f"WARNINGS: Command at index {i} must have a 'keystrokes' string field.",
                )

            duration = cmd.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                return (
                    ParsedResponse(commands=[], task_complete=False),
                    f"WARNINGS: Command at index {i} 'duration' must be a number.",
                )

            # Don't strip keystrokes - preserve exact whitespace as in official implementation
            commands.append(Command(keystrokes=keystrokes, duration=float(duration)))

        # Parse task_complete
        task_complete = data.get("task_complete", False)
        if not isinstance(task_complete, bool):
            return (
                ParsedResponse(commands=commands, task_complete=False),
                "WARNINGS: 'task_complete' field must be a boolean.",
            )

        # Parse thoughts (optional)
        thoughts = data.get("thoughts")
        if thoughts is not None and not isinstance(thoughts, str):
            thoughts = str(thoughts)

        return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts), None

    def _parse_with_regex(self, json_str: str) -> ParsedResponse | None:
        """Fallback parser using very broad regex to extract fields from malformed JSON.

        Args:
            json_str: The malformed JSON string.

        Returns:
            ParsedResponse if successful, None if extraction failed.
        """
        commands = []

        # Very broad pattern: find all "keystrokes": "..." patterns
        # Don't try to match the full JSON structure, just grab keystrokes and durations
        keystroke_pattern = r'"keystrokes"\s*:\s*"([^"]*(?:\\"[^"]*)*)"'
        duration_pattern = r'"duration"\s*:\s*(\d+(?:\.\d+)?)'

        # Find all keystrokes
        keystroke_matches = list(re.finditer(keystroke_pattern, json_str, re.DOTALL))

        # Find all durations
        duration_matches = list(re.finditer(duration_pattern, json_str))

        # Pair them up (assume they appear in order)
        for i, keystroke_match in enumerate(keystroke_matches):
            keystrokes = keystroke_match.group(1)
            # Unescape common JSON escapes
            keystrokes = keystrokes.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")

            # Try to find the corresponding duration
            duration = 1.0  # default (matches official)
            # Check if this duration comes after this keystroke
            if i < len(duration_matches) and duration_matches[i].start() > keystroke_match.end():
                duration = float(duration_matches[i].group(1))

            # Don't strip keystrokes - preserve exact whitespace
            commands.append(Command(keystrokes=keystrokes, duration=duration))

        # Extract task_complete
        task_complete = False
        task_complete_match = re.search(r'"task_complete"\s*:\s*(true|false)', json_str, re.IGNORECASE)
        if task_complete_match:
            task_complete = task_complete_match.group(1).lower() == "true"

        # Extract thoughts (optional) - be very broad
        thoughts = None
        thoughts_match = re.search(r'"thoughts"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', json_str, re.DOTALL)
        if thoughts_match:
            thoughts = thoughts_match.group(1).replace('\\"', '"').replace("\\n", "\n")

        # Only return if we extracted at least some data
        if commands or task_complete:
            return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts)

        return None
