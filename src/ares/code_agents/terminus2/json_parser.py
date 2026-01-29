# Original code: https://github.com/laude-institute/terminal-bench/tree/main/terminal_bench/agents/terminus_2
# Copyright (c) 2025 Laude Institute
# Licensed under the Apache-2.0 License.
#
# Modifications Copyright (c) 2026 Martian

"""JSON parser for Terminus 2 agent responses.

Based on terminal-bench implementation with enhanced auto-fixes and validation:
- Auto-fix for incomplete JSON (missing braces/brackets)
- Field order validation
- Command formatting validation (missing newlines)
- Regex fallback for malformed JSON
"""

import dataclasses
import json
import logging
import re

_LOGGER = logging.getLogger(__name__)

# Expected field order in JSON responses
_EXPECTED_FIELD_ORDER = ["analysis", "plan", "commands", "task_complete"]


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

    def parse(self, response_text: str) -> tuple[ParsedResponse | None, str | None]:
        """Parse the agent's response.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A tuple of (ParsedResponse | None, feedback):
            - (ParsedResponse, None) - success with no warnings
            - (ParsedResponse, warnings) - success with warnings
            - (None, error_message) - parse failure
        """
        warnings = []

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
                return (None, "ERROR: Could not find JSON in response. Please provide a valid JSON response.")

        # Try auto-fix for incomplete JSON first
        original_json_str = json_str
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            _LOGGER.debug("Initial JSON parsing failed: %s. Attempting auto-fix.", e)
            json_str = self._auto_fix_json(json_str)

            try:
                data = json.loads(json_str)
                _LOGGER.info("Successfully auto-fixed JSON")
                warnings.append("Note: JSON was auto-fixed (added missing closing braces/brackets).")
            except json.JSONDecodeError as e2:
                # Auto-fix failed, try regex fallback
                _LOGGER.warning("Auto-fix failed: %s. Using regex fallback", e2)
                try:
                    fallback_parsed = self._parse_with_regex(original_json_str)
                    if fallback_parsed:
                        _LOGGER.info("Successfully recovered data using regex fallback")
                        return fallback_parsed, None
                except Exception as regex_error:
                    _LOGGER.warning("Regex fallback also failed: %s", regex_error)

                return (
                    None,
                    f"ERROR: Invalid JSON: {e2}. Please provide valid JSON.",
                )

        # Validate field order
        self._validate_field_order(data, warnings)

        # Parse commands
        commands = []
        raw_commands = data.get("commands", [])
        if not isinstance(raw_commands, list):
            return (None, "ERROR: 'commands' field must be an array.")

        for i, cmd in enumerate(raw_commands):
            if not isinstance(cmd, dict):
                return (None, f"ERROR: Command at index {i} must be an object.")

            keystrokes = cmd.get("keystrokes")
            if not keystrokes or not isinstance(keystrokes, str):
                return (None, f"ERROR: Command at index {i} must have a 'keystrokes' string field.")

            duration = cmd.get("duration", 1.0)
            if not isinstance(duration, (int, float)):
                return (None, f"ERROR: Command at index {i} 'duration' must be a number.")

            # Don't strip keystrokes - preserve exact whitespace as in official implementation
            commands.append(Command(keystrokes=keystrokes, duration=float(duration)))

        # Validate command formatting (check for missing newlines between commands)
        if len(commands) > 1:
            self._validate_command_formatting(raw_commands, warnings)

        # Parse task_complete
        task_complete = data.get("task_complete", False)
        if not isinstance(task_complete, bool):
            return (None, "ERROR: 'task_complete' field must be a boolean.")

        # Parse thoughts (optional)
        thoughts = data.get("thoughts")
        if thoughts is not None and not isinstance(thoughts, str):
            thoughts = str(thoughts)

        # Return warnings if any
        feedback = "\n".join(warnings) if warnings else None

        return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts), feedback

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

        # Only return if we extracted at least some meaningful data
        if commands or task_complete_match is not None or thoughts_match is not None:
            return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts)

        return None

    def _auto_fix_json(self, json_str: str) -> str:
        """Auto-fix common JSON errors like missing closing braces/brackets.

        Based on the original terminal-bench implementation.

        Args:
            json_str: The potentially incomplete JSON string.

        Returns:
            The fixed JSON string.
        """
        # Count opening and closing braces/brackets
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")

        # Add missing closing braces
        if open_braces > close_braces:
            missing = open_braces - close_braces
            json_str += "}" * missing
            _LOGGER.debug("Auto-fix: Added %d closing brace(s)", missing)

        # Add missing closing brackets
        if open_brackets > close_brackets:
            missing = open_brackets - close_brackets
            json_str += "]" * missing
            _LOGGER.debug("Auto-fix: Added %d closing bracket(s)", missing)

        return json_str

    def _validate_field_order(self, data: dict, warnings: list[str]) -> None:
        """Validate that fields appear in the expected order.

        Args:
            data: The parsed JSON data.
            warnings: List to append warnings to.
        """
        # Get keys in the actual JSON order, filtering to only expected fields
        present_keys = [key for key in data if key in _EXPECTED_FIELD_ORDER]

        if len(present_keys) < 2:
            # Not enough fields to check order
            return

        # Map keys to their indices in the expected order
        indices = [_EXPECTED_FIELD_ORDER.index(key) for key in present_keys]

        # Check if indices are strictly increasing
        for i in range(len(indices) - 1):
            if indices[i] > indices[i + 1]:
                curr_key = present_keys[i]
                next_key = present_keys[i + 1]
                warnings.append(
                    f"Warning: Fields should be in order: {', '.join(_EXPECTED_FIELD_ORDER)}. "
                    f"Found '{next_key}' before '{curr_key}'."
                )
                break

    def _validate_command_formatting(self, raw_commands: list, warnings: list[str]) -> None:
        """Check if commands are missing newlines between them (common mistake).

        Args:
            raw_commands: The raw command objects from JSON.
            warnings: List to append warnings to.
        """
        # Check if any keystrokes field is missing a trailing newline when there are multiple commands
        # This is a common mistake where the agent forgets to add \n between commands
        for i, cmd in enumerate(raw_commands[:-1]):  # Check all but the last
            keystrokes = cmd.get("keystrokes", "")
            if keystrokes and not keystrokes.endswith("\n"):
                warnings.append(
                    f"Note: Command {i} doesn't end with a newline. "
                    "If executing multiple commands, ensure each ends with \\n."
                )
                break  # Only warn once
