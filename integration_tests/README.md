# Integration Tests

This directory contains integration tests for ARES that test the full system with real containers, LLM clients, and datasets.

## Running Integration Tests

**Note:** Integration tests are currently **manually run** and not included in the standard test suite.

### Prerequisites

- Valid API keys configured in `.env`:
  - `DAYTONA_API_KEY` and `DAYTONA_API_URL` for container creation
  - `CHAT_COMPLETION_API_KEY` for LLM inference
- Development dependencies installed: `uv sync --group dev`

### Running Tests

Run all integration tests in parallel:
```bash
uv run pytest integration_tests/ -n auto -v
```

Run a specific integration test:
```bash
uv run pytest integration_tests/test_default_workdir.py -v
```

Run a specific test case:
```bash
uv run pytest integration_tests/test_default_workdir.py::test_default_workdir[sbv-mswea:0-/testbed] -v
```

### Important Notes

- Integration tests create real containers on Daytona and may incur costs
- Tests use the `:0` selector to always test the first task from each dataset for consistency
- Tests can be run in parallel using `pytest-xdist` (`-n auto` flag)
- Each test cleans up its resources, but verify containers are removed if tests fail

## Current Tests

### `test_default_workdir.py`

Tests that environments use the correct default working directory for each benchmark:
- `sbv-mswea:0` - Verifies SWE-bench uses `/testbed`
- `tbench-mswea:0` - Verifies terminal-bench uses `/app`

This ensures that our `default_workdir` container configuration is properly set for each dataset.
