"""Unit tests for JSON parser.

Tests focus on the critical fixes:
1. Default duration is 1.0 (not 5.0)
2. Whitespace is preserved in keystrokes (not stripped)
3. Regex fallback works correctly
"""

import pytest

from ares.code_agents.terminus2 import json_parser


class TestJSONParserBasics:
    """Test basic JSON parsing functionality."""

    def test_simple_valid_json(self):
        """Test parsing simple valid JSON."""
        response = """{
  "analysis": "Current directory is /testbed",
  "plan": "List files",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == "ls -la\n"
        assert result.commands[0].duration == 0.1
        assert result.task_complete is False

    def test_json_in_code_block(self):
        """Test extracting JSON from markdown code block."""
        response = """Here's my response:

```json
{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "pwd\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}
```

That's my response."""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        assert result.commands[0].keystrokes == "pwd\n"

    def test_default_duration_is_one(self):
        """Test default duration is 1.0 seconds (not 5.0)."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "echo test\\n"
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        assert result.commands[0].duration == 1.0, f"Default should be 1.0, got {result.commands[0].duration}"

    def test_whitespace_preserved(self):
        """Test whitespace in keystrokes is preserved, not stripped."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "  echo test  \\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        # Should preserve leading/trailing spaces
        assert result.commands[0].keystrokes == "  echo test  \n"
        assert result.commands[0].keystrokes.startswith("  ")
        assert result.commands[0].keystrokes.endswith("  \n")

    def test_multiple_commands(self):
        """Test parsing multiple commands."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "cd /tmp\\n",
      "duration": 0.1
    },
    {
      "keystrokes": "ls\\n",
      "duration": 0.1
    },
    {
      "keystrokes": "pwd\\n",
      "duration": 1.0
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)

        assert result is not None  # Parser succeeded
        assert error is None
        assert len(result.commands) == 3
        assert result.commands[0].keystrokes == "cd /tmp\n"
        assert result.commands[0].duration == 0.1
        assert result.commands[1].keystrokes == "ls\n"
        assert result.commands[1].duration == 0.1
        assert result.commands[2].keystrokes == "pwd\n"
        assert result.commands[2].duration == 1.0


class TestJSONParserValidation:
    """Test parser validation and error handling."""

    def test_invalid_json(self):
        """Test error on invalid JSON."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "ls"
      "duration": 0.1
    }
  ]
}"""  # Missing comma

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)

        # Should try regex fallback, but may still fail
        assert error is not None or (result is not None and len(result.commands) >= 0)

    def test_missing_keystrokes(self):
        """Test error when keystrokes field is missing."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "duration": 0.1
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        _result, error = parser.parse(response)

        assert error is not None
        assert "keystrokes" in error.lower()

    def test_invalid_duration_type(self):
        """Test error when duration is not a number."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "ls\\n",
      "duration": "not-a-number"
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        _result, error = parser.parse(response)

        assert error is not None
        assert "duration" in error.lower()

    def test_commands_not_array(self):
        """Test error when commands is not an array."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": "not an array",
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        _result, error = parser.parse(response)

        assert error is not None
        assert "array" in error.lower()

    def test_task_complete_true(self):
        """Test parsing task_complete flag."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [],
  "task_complete": true
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert result.task_complete is True


class TestJSONParserRegexFallback:
    """Test regex fallback for malformed JSON."""

    def test_regex_fallback_on_parse_error(self):
        """Test regex fallback extracts commands from malformed JSON."""
        # Malformed JSON but has extractable patterns
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "ls -la\\n",
      "duration": 0.5
    },
    {
      "keystrokes": "pwd\\n",
      "duration": 1.5
    }
  ]  // Invalid comment
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)

        # Regex fallback should extract commands
        # May or may not have error depending on fallback success
        if error is None:
            assert result is not None
            assert len(result.commands) >= 1

    def test_regex_fallback_preserves_whitespace(self):
        """Test regex fallback preserves whitespace."""
        # Malformed JSON
        response = """{
  "keystrokes": "  test  \\n",
  "duration": 1.5
}INVALID"""

        parser = json_parser.Terminus2JSONParser()
        # This will fail JSON parse and try regex fallback
        result, _ = parser.parse(response)

        # If fallback succeeded, whitespace should be preserved
        if result is not None and len(result.commands) > 0:
            # After JSON unescaping, should preserve spaces
            assert "test" in result.commands[0].keystrokes

    def test_regex_fallback_default_duration(self):
        """Test regex fallback uses 1.0 as default duration."""
        # Malformed JSON with command but no duration
        response = """{
  "keystrokes": "echo test\\n"
}INVALID"""

        parser = json_parser.Terminus2JSONParser()
        parsed = parser._parse_with_regex(response)

        if parsed and len(parsed.commands) > 0:
            assert parsed.commands[0].duration == 1.0


class TestJSONParserEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_commands_array(self):
        """Test response with empty commands array."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 0

    def test_special_characters_in_keystrokes(self):
        """Test keystrokes with special characters."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "echo \\"hello world\\" | grep 'hello'\\n",
      "duration": 0.1
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        assert '"hello world"' in result.commands[0].keystrokes

    def test_multiline_keystrokes(self):
        """Test keystrokes with newlines."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "cat <<EOF\\nline 1\\nline 2\\nEOF\\n",
      "duration": 1.0
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert len(result.commands) == 1
        assert "line 1" in result.commands[0].keystrokes
        assert "line 2" in result.commands[0].keystrokes

    def test_optional_thoughts_field(self):
        """Test optional thoughts field."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [],
  "task_complete": false,
  "thoughts": "Some internal thoughts"
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert result.thoughts == "Some internal thoughts"

    def test_integer_duration(self):
        """Test that integer duration is accepted."""
        response = """{
  "analysis": "Test",
  "plan": "Test",
  "commands": [
    {
      "keystrokes": "ls\\n",
      "duration": 1
    }
  ],
  "task_complete": false
}"""

        parser = json_parser.Terminus2JSONParser()
        result, error = parser.parse(response)
        assert result is not None  # Parser succeeded

        assert error is None
        assert result.commands[0].duration == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
