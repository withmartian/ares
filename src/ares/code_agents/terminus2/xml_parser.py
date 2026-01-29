# Original code: https://github.com/laude-institute/terminal-bench/tree/main/terminal_bench/agents/terminus_2
# Copyright (c) 2025 Laude Institute
# Licensed under the Apache-2.0 License.
#
# Modifications Copyright (c) 2026 Martian

"""XML parser for Terminus 2 agent responses.

Based on terminal-bench implementation with enhanced validation:
- Field order validation for XML elements
- Double-escaped entity detection
- Multiple command blocks warning
- XML salvage for incomplete responses
"""

import dataclasses
import logging
import re
import xml.etree.ElementTree as ET

_LOGGER = logging.getLogger(__name__)

# Expected field order in XML responses
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


class Terminus2XMLParser:
    """Parser for XML-formatted Terminus 2 responses."""

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

        # Try to extract XML from code blocks or raw text
        xml_match = re.search(r"```xml\s*\n(.*?)\n```", response_text, re.DOTALL)
        if xml_match:
            xml_str = xml_match.group(1)
        else:
            # Try to find <response> tag in the text
            xml_match = re.search(r"<response>.*?</response>", response_text, re.DOTALL)
            if xml_match:
                xml_str = xml_match.group(0)
            else:
                return (None, "ERROR: Could not find <response> XML in response. Please provide a valid XML response.")

        # Check for common XML mistakes
        self._validate_xml_content(xml_str, warnings)

        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            return (None, f"ERROR: Invalid XML: {e}. Please provide valid XML.")

        if root.tag != "response":
            return (None, "ERROR: Root element must be <response>.")

        # Validate field order
        self._validate_field_order(root, warnings)

        # Parse commands
        commands = []
        commands_elem = root.find("commands")
        if commands_elem is not None:
            # Check for multiple command blocks (common mistake)
            all_commands_elems = root.findall("commands")
            if len(all_commands_elems) > 1:
                warnings.append(
                    f"Warning: Found {len(all_commands_elems)} <commands> blocks. "
                    "Only the first will be used. Combine all commands into a single <commands> block."
                )

            # Look for <keystrokes> elements directly (not wrapped in <command>)
            for i, keystrokes_elem in enumerate(commands_elem.findall("keystrokes")):
                if keystrokes_elem.text is None:
                    return (None, f"ERROR: Keystrokes element at index {i} must have text content.")

                # Don't strip - preserve exact whitespace as in official implementation
                keystrokes = keystrokes_elem.text

                # Duration is an XML attribute, not a child element
                duration_attr = keystrokes_elem.get("duration")
                if duration_attr is not None:
                    try:
                        duration = float(duration_attr)
                    except ValueError:
                        return (
                            None,
                            f"ERROR: Keystrokes element at index {i} has invalid duration attribute: {duration_attr}",
                        )
                else:
                    duration = 1.0  # Default duration (matches official)

                commands.append(Command(keystrokes=keystrokes, duration=duration))

        # Parse task_complete
        task_complete = False
        task_complete_elem = root.find("task_complete")
        if task_complete_elem is not None and task_complete_elem.text:
            task_complete_text = task_complete_elem.text.strip().lower()
            if task_complete_text in ("true", "1", "yes"):
                task_complete = True
            elif task_complete_text not in ("false", "0", "no"):
                return (None, "ERROR: <task_complete> must be 'true' or 'false'.")

        # Parse thoughts (optional)
        thoughts = None
        thoughts_elem = root.find("thoughts")
        if thoughts_elem is not None and thoughts_elem.text:
            thoughts = thoughts_elem.text.strip()

        # Return warnings if any
        feedback = "\n".join(warnings) if warnings else None

        return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts), feedback

    def salvage_truncated_response(self, truncated_text: str) -> tuple[str | None, bool]:
        """Try to salvage a valid response from truncated XML output.

        This matches the terminal-bench reference implementation's behavior.

        Args:
            truncated_text: The truncated XML response.

        Returns:
            Tuple of (salvaged_response, has_multiple_blocks) where:
            - salvaged_response is the valid XML if found, None otherwise
            - has_multiple_blocks indicates if multiple <response> blocks were found
        """
        # Try to find complete <response> blocks in the truncated output
        import re

        # Find all complete <response>...</response> blocks
        response_pattern = r"<response>.*?</response>"
        matches = list(re.finditer(response_pattern, truncated_text, re.DOTALL))

        if not matches:
            return None, False

        # If we found multiple blocks, that's unusual
        has_multiple_blocks = len(matches) > 1

        # Try to parse the first complete block
        for match in matches:
            response_xml = match.group(0)
            try:
                # Try to parse it to verify it's valid
                root = ET.fromstring(response_xml)
                if root.tag == "response":
                    # Valid response found!
                    return response_xml, has_multiple_blocks
            except ET.ParseError:
                continue

        return None, has_multiple_blocks

    def _validate_xml_content(self, xml_str: str, warnings: list[str]) -> None:
        """Validate XML content for common mistakes.

        Args:
            xml_str: The XML string to validate.
            warnings: List to append warnings to.
        """
        # Check if XML entities are escaped when they shouldn't be (common mistake)
        # In XML content, < > & should be written as &lt; &gt; &amp;
        # But sometimes LLMs write &amp;lt; which is double-escaped
        if "&amp;lt;" in xml_str or "&amp;gt;" in xml_str or "&amp;amp;" in xml_str:
            warnings.append(
                "Warning: Found double-escaped XML entities (&amp;lt;, &amp;gt;, &amp;amp;). "
                "Use single escapes (&lt;, &gt;, &amp;) or write content without escaping if appropriate."
            )

    def _validate_field_order(self, root: ET.Element, warnings: list[str]) -> None:
        """Validate that child elements appear in the expected order.

        Args:
            root: The root XML element.
            warnings: List to append warnings to.
        """
        # Get the tags of child elements
        child_tags = [child.tag for child in root]

        # Get the tags that are present in both children and expected order
        present_tags = [tag for tag in _EXPECTED_FIELD_ORDER if tag in child_tags]

        if len(present_tags) < 2:
            # Not enough fields to check order
            return

        # Check if they appear in the expected order in the actual XML
        actual_positions = {tag: child_tags.index(tag) for tag in present_tags}

        for i in range(len(present_tags) - 1):
            curr_tag = present_tags[i]
            next_tag = present_tags[i + 1]

            # Check if current appears before next in actual XML
            if actual_positions[curr_tag] > actual_positions[next_tag]:
                warnings.append(
                    f"Warning: Child elements should be in order: {', '.join(_EXPECTED_FIELD_ORDER)}. "
                    f"Found <{next_tag}> before <{curr_tag}>."
                )
                break
