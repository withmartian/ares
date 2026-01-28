"""XML parser for Terminus 2 agent responses."""

import dataclasses
import re
import xml.etree.ElementTree as ET


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

    def parse(self, response_text: str) -> tuple[ParsedResponse, str | None]:
        """Parse the agent's response.

        Args:
            response_text: The raw text response from the LLM.

        Returns:
            A tuple of (ParsedResponse, feedback) where feedback is None if parsing
            succeeded, or an error message if parsing failed.
        """
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
                return (
                    ParsedResponse(commands=[], task_complete=False),
                    "WARNINGS: Could not find <response> XML in response. Please provide a valid XML response.",
                )

        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            return (
                ParsedResponse(commands=[], task_complete=False),
                f"WARNINGS: Invalid XML: {e}. Please provide valid XML.",
            )

        if root.tag != "response":
            return (
                ParsedResponse(commands=[], task_complete=False),
                "WARNINGS: Root element must be <response>.",
            )

        # Parse commands
        commands = []
        commands_elem = root.find("commands")
        if commands_elem is not None:
            # Look for <keystrokes> elements directly (not wrapped in <command>)
            for i, keystrokes_elem in enumerate(commands_elem.findall("keystrokes")):
                if keystrokes_elem.text is None:
                    return (
                        ParsedResponse(commands=[], task_complete=False),
                        f"WARNINGS: Keystrokes element at index {i} must have text content.",
                    )

                # Don't strip - preserve exact whitespace as in official implementation
                keystrokes = keystrokes_elem.text

                # Duration is an XML attribute, not a child element
                duration_attr = keystrokes_elem.get("duration")
                if duration_attr is not None:
                    try:
                        duration = float(duration_attr)
                    except ValueError:
                        return (
                            ParsedResponse(commands=[], task_complete=False),
                            (
                                f"WARNINGS: Keystrokes element at index {i} "
                                f"has invalid duration attribute: {duration_attr}"
                            ),
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
                return (
                    ParsedResponse(commands=commands, task_complete=False),
                    "WARNINGS: <task_complete> must be 'true' or 'false'.",
                )

        # Parse thoughts (optional)
        thoughts = None
        thoughts_elem = root.find("thoughts")
        if thoughts_elem is not None and thoughts_elem.text:
            thoughts = thoughts_elem.text.strip()

        return ParsedResponse(commands=commands, task_complete=task_complete, thoughts=thoughts), None

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
