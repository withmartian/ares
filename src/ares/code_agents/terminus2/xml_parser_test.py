"""Unit tests for XML parser.

Tests focus on the critical fixes:
1. Parser looks for <keystrokes> directly (not <command> wrappers)
2. Duration is parsed from attribute (not child element)
3. Default duration is 1.0 (not 5.0)
4. Whitespace is preserved in keystrokes (not stripped)
"""

import pytest

from ares.code_agents.terminus2 import xml_parser


class TestXMLParserStructure:
    """Test that parser correctly handles XML structure from template."""

    def test_keystrokes_direct_not_wrapped(self):
        """Test parser extracts <keystrokes> elements directly, not wrapped in <command>."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test plan</plan>
<commands>
<keystrokes duration="0.1">ls -la
</keystrokes>
<keystrokes duration="1.0">pwd
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None, f"Parser should succeed, got error: {error}"
        assert result is not None
        assert len(result.commands) == 2, f"Should extract 2 commands, got {len(result.commands)}"
        assert result.commands[0].keystrokes == "ls -la\n"
        assert result.commands[1].keystrokes == "pwd\n"

    def test_duration_from_attribute(self):
        """Test duration is parsed from XML attribute, not child element."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="2.5">echo test
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 1
        assert result.commands[0].duration == 2.5

    def test_default_duration_is_one(self):
        """Test default duration is 1.0 seconds (not 5.0)."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes>echo test
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 1
        assert result.commands[0].duration == 1.0, f"Default should be 1.0, got {result.commands[0].duration}"

    def test_whitespace_preserved(self):
        """Test whitespace in keystrokes is preserved, not stripped."""
        # Test leading spaces and newlines
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">  echo test
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 1
        # Should preserve leading spaces and newline
        # Note: XML parsing naturally trims trailing whitespace before closing tag
        assert result.commands[0].keystrokes.startswith("  ")
        assert result.commands[0].keystrokes.endswith("\n")

    def test_newline_preserved(self):
        """Test that newline at end of keystrokes is preserved."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">ls -la
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert result.commands[0].keystrokes.endswith("\n"), "Should preserve trailing newline"


class TestXMLParserValidation:
    """Test parser validation and error handling."""

    def test_invalid_duration_attribute(self):
        """Test error when duration attribute is not a valid number."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="not-a-number">echo test
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert result is None  # Parser should fail
        assert error is not None
        assert "invalid duration attribute" in error.lower()

    def test_empty_keystrokes(self):
        """Test error when keystrokes element has no text."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="1.0"></keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        _result, error = parser.parse(response)

        assert error is not None
        assert "must have text content" in error.lower()

    def test_multiple_commands(self):
        """Test parsing multiple commands."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">cd /tmp
</keystrokes>
<keystrokes duration="0.1">ls
</keystrokes>
<keystrokes duration="1.0">pwd
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 3
        assert result.commands[0].keystrokes == "cd /tmp\n"
        assert result.commands[0].duration == 0.1
        assert result.commands[1].keystrokes == "ls\n"
        assert result.commands[1].duration == 0.1
        assert result.commands[2].keystrokes == "pwd\n"
        assert result.commands[2].duration == 1.0

    def test_task_complete_true(self):
        """Test parsing task_complete flag."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">echo done
</keystrokes>
</commands>
<task_complete>true</task_complete>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert result.task_complete is True

    def test_task_complete_false(self):
        """Test task_complete defaults to false."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">echo test
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert result.task_complete is False


class TestXMLParserEdgeCases:
    """Test edge cases and special scenarios."""

    def test_no_commands(self):
        """Test response with no commands."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 0

    def test_special_characters_in_keystrokes(self):
        """Test keystrokes with special characters."""
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">echo "hello world" | grep 'hello'
</keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 1
        assert 'echo "hello world"' in result.commands[0].keystrokes

    def test_multiline_keystrokes(self):
        """Test keystrokes spanning multiple lines."""
        # Note: Using CDATA to avoid XML parsing issues with special characters
        response = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="1.0"><![CDATA[cat <<EOF
line 1
line 2
EOF
]]></keystrokes>
</commands>
</response>"""

        parser = xml_parser.Terminus2XMLParser()
        result, error = parser.parse(response)

        assert error is None
        assert result is not None
        assert len(result.commands) == 1
        assert "line 1" in result.commands[0].keystrokes
        assert "line 2" in result.commands[0].keystrokes
        assert "EOF" in result.commands[0].keystrokes


class TestXMLParserSalvage:
    """Test salvage_truncated_response functionality."""

    def test_salvage_complete_response(self):
        """Test salvaging a complete response from truncated output."""
        truncated = """<response>
<analysis>Test</analysis>
<plan>Test</plan>
<commands>
<keystrokes duration="0.1">ls
</keystrokes>
</commands>
</response>
... [MORE TRUNCATED TEXT] ..."""

        parser = xml_parser.Terminus2XMLParser()
        salvaged, has_multiple = parser.salvage_truncated_response(truncated)

        assert salvaged is not None
        assert has_multiple is False
        assert "<response>" in salvaged
        assert "</response>" in salvaged

    def test_salvage_no_complete_response(self):
        """Test salvaging when no complete response exists."""
        truncated = """<response>
<analysis>Test</analysis>
<plan>Test"""

        parser = xml_parser.Terminus2XMLParser()
        salvaged, has_multiple = parser.salvage_truncated_response(truncated)

        assert salvaged is None
        assert has_multiple is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
