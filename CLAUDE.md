# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARES (Agentic Research and Evaluation Suite) is an RL-first framework for training and evaluating code agents. It implements an async version of DeepMind's dm_env specification, treating LLM requests as observations and LLM responses as actions within a standard RL loop.

## Development Commands

### Setup
```bash
# Install all dependencies including dev tools
uv sync --all-groups

# Install only main dependencies
uv sync

# Install with specific groups
uv sync --group dev
uv sync --group examples
```

### Testing
```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest src/ares/config_test.py

# Run tests matching a pattern
uv run pytest -k "test_pattern_name"
```

Unit tests follow the `*_test.py` naming pattern (preferred) or `test_*.py` and are colocated with source files in `src/`. Integration and end-to-end tests may live under `integration_tests/` at the root.

### Code Quality
```bash
# Lint (check for issues)
uv run ruff check

# Format code
uv run ruff format

# Type checking
uv run pyright

# Fix auto-fixable linting issues
uv run ruff check --fix
```

The project uses ruff for linting/formatting (line length: 120) and follows Google-style docstrings.

### Running Examples
```bash
# Run the minimal loop example with local Docker
uv run -m examples.01_minimal_loop

# Run with local LLM
uv run -m examples.02_local_llm
```

Examples demonstrate basic usage patterns and require the examples dependency group.

## High-Level Architecture

### Core Abstraction: The RL Loop

ARES treats code agent interactions as a reinforcement learning problem:

1. **Environment** emits an **Observation** (LLM request with task context)
2. **Agent** receives observation and returns an **Action** (LLM response with code/commands)
3. **Environment** processes action, executes commands in container, returns next observation
4. Loop continues until episode terminates (success, step limit, or explicit submission)
5. **Reward** is computed at episode end (e.g., 1.0 if tests pass, 0.0 otherwise)

### Key Components

#### 1. Environments (`src/ares/environments/`)

**Base (`base.py`):**
- `Environment` protocol defines the dm_env interface (`reset()`, `step()`, `close()`)
- `TimeStep` namedtuple for observations, rewards, and episode signals
- `create_container()` helper for creating containers from images or Dockerfiles
- `Janitor` class for emergency cleanup of containers on abnormal termination

**Implementation (`code_env.py`):**
- `CodeEnvironment` - Concrete environment for Harbor-compatible datasets (including SWE-bench)
- Orchestrates the entire RL loop: manages container lifecycle, code agent execution, and LLM request interception
- Builds containers from Dockerfiles with configurable resources (CPU, memory, disk)
- Reads reward from `/reward.txt` or `/reward.json` in the container
- Uses async context manager pattern (`async with env:`) for guaranteed cleanup

**Episode Termination:**
- Step limit reached (default: 250 steps for CodeEnvironment)
- Agent explicitly submits (signals completion)
- Container or agent error

#### 2. Code Agents (`src/ares/code_agents/`)

**Interface:**
```python
class CodeAgent(Protocol):
    async def run(self, task: str) -> None
```

**Main Implementation:**
- `MiniSWECodeAgent` (`mini_swe_agent.py`) - Wraps the mini-swe-agent library
  - Uses Jinja2-rendered system/instance templates
  - Parses bash commands from markdown code blocks
  - Handles format errors, timeouts, and submission signals
  - Tracks performance statistics via `StatTracker`

**Factory Pattern:**
- `CodeAgentFactory[T]` protocol creates agents with container and LLM client dependencies
- Enables dependency injection and easy agent swapping

#### 3. Containers (`src/ares/containers/`)

**Abstract Protocol (`containers.py`):**
```python
class Container(Protocol):
    async def start(env: dict[str, str] | None) -> None
    async def exec_run(command, workdir, env, timeout_s) -> ExecResult
    async def upload_files/download_files/upload_dir/download_dir
    def stop_and_remove() -> None  # Sync for atexit cleanup
```

**Implementations:**
- `DaytonaContainer` (`daytona.py`) - Cloud containers via Daytona API (default)
  - Auto-stop and auto-delete configuration
  - Retry logic with exponential backoff
  - Supports resource specs (CPU, memory, disk, GPU)
- `DockerContainer` (`docker.py`) - Local Docker containers via docker-py
  - Builds images from Dockerfiles on-demand
  - Uses `tail -f /dev/null` to keep containers alive
  - Tar-based file upload/download

**Janitor Pattern:**
- Both implementations register cleanup with `atexit` for emergency shutdown
- Ensures containers are removed even on abnormal termination

#### 4. LLM Clients (`src/ares/llms/`)

**Core Abstractions:**
- `LLMRequest` - Dataclass with messages and optional temperature
- `LLMResponse` - Dataclass with ChatCompletion and cost tracking
- `LLMClient` Protocol - `async def __call__(request: LLMRequest) -> LLMResponse`

**Key Pattern: Queue-Mediated LLM Client (`queue_mediated_client.py`):**

This is the **most critical pattern** in ARES. It enables the RL abstraction by:
1. Intercepting LLM calls from code agents using `asyncio.Queue`
2. Exposing intercepted requests as observations to the RL environment
3. Pairing requests with responses via futures
4. Allowing agents to be written naturally (making direct LLM calls) while the environment controls the RL loop

Without this pattern, agents would need explicit RL-aware interfaces, breaking the abstraction.

**Other Implementations:**
- `ChatCompletionCompatibleLLMClient` (`chat_completions_compatible.py`) - OpenAI-compatible API client
  - Uses Martian API by default
  - Retry logic with tenacity (3 attempts, exponential backoff)
  - Integrated cost tracking via `accounting.py`

#### 5. Supporting Modules

**Configuration (`config.py`):**
- Pydantic Settings-based configuration from `.env`
- Required: `DAYTONA_API_KEY`, `DAYTONA_API_URL`, `CHAT_COMPLETION_API_KEY`
- Auto-detects user from environment variables for logging/tracking

**Statistics Tracking (`stat_tracker.py`):**
- `StatTracker` Protocol with context manager for timing: `with tracker.timeit(name):`
- Implementations: `NullStatTracker` (no-op), `LoggingStatTracker`, `TensorboardStatTracker`
- Non-intrusive performance monitoring

**Async Utilities (`async_utils.py`):**
- `ValueAndFuture[ValType, FutureType]` - Pairs values with futures for coordination
- Helper functions for async patterns

## Key Design Patterns

### Protocol-Oriented Design
Heavy use of `typing.Protocol` for structural subtyping without inheritance. Key protocols: `Environment`, `CodeAgent`, `Container`, `LLMClient`, `ContainerFactory`, `CodeAgentFactory`, `StatTracker`. Environments implement the `Environment` protocol directly without inheritance hierarchies.

### Factory Pattern
Used for dependency injection - environments receive factories, not concrete instances, allowing easy swapping of implementations (local vs cloud containers, different agents, etc.).

### Context Manager Lifecycle
All major resources use `async with` for guaranteed cleanup. Environments, containers, and other resources implement `__aenter__`/`__aexit__`.

### Dataclass Immutability
Most dataclasses use `frozen=True` to ensure thread-safety in async contexts.

### Queue-Mediated Communication
Async queues bridge linear agent code with the RL environment, enabling "interception" of LLM calls without agents being aware of the RL loop.

### YAGNI (You Aren't Gonna Need It)
Prefer concrete implementations over abstractions. For example, `CodeEnvironment` implements the `Environment` protocol directly without base classes, since Harbor is designed to handle all code agent benchmarks. Abstractions are added only when needed.

## Code Conventions

### Naming
- Private methods: `_method_name` (single underscore)
- Module-level loggers: `_LOGGER`
- Constants: `UPPER_CASE_WITH_UNDERSCORES`
- Abstract methods marked with `@abc.abstractmethod`

### Type Annotations
Full type annotations throughout. Generic types used extensively (e.g., `CodeBaseEnv[TaskType]`).

### Logging
Extensive use of `logging.getLogger(__name__)` with object IDs for tracking across async operations. Example:
```python
_LOGGER.info("Container %s started", id(self))
```

### Error Handling
- Custom exception hierarchies distinguish terminating vs non-terminating errors
- Retry logic with exponential backoff for transient failures (container creation, API calls)

### Imports
Follow Google-style isort configuration:
- Force single-line imports (except `typing` and `collections.abc`)
- No separation between `import` and `from ... import`
- Sort within sections

### Comments
- WHY over WHAT - explain reasoning, edge cases, non-obvious decisions
- HOW only when implementation is genuinely complex

## Environment Variables

Create `.env` file from `.env.example`:

**Required for Daytona (default):**
- `DAYTONA_API_KEY` - API key from daytona.io
- `DAYTONA_API_URL` - Daytona API endpoint

**Required for LLM inference:**
- `CHAT_COMPLETION_API_KEY` - Martian API key from app.withmartian.com

**Optional:**
- `CHAT_COMPLETION_BASE_URL` - Override default API endpoint
- `USER`, `LOGNAME`, `USERNAME` - User identification (auto-detected)

## Testing Patterns

- Unit tests must be under `src/` and live next to their source files
  - Naming: `*_test.py` (preferred) or `test_*.py`
  - Mock external services (containers, API calls) in unit tests
- Integration and end-to-end tests may live under `integration_tests/` at the root
- Use pytest fixtures for common setup
- Async tests use `pytest-asyncio`

## CI/CD

GitHub Actions workflow runs on PRs and main branch:
- Linting with `ruff check`
- Formatting check with `ruff format --check`

Before pushing, ensure:
```bash
uv run pytest        # Tests pass
uv run ruff format   # Code formatted
uv run ruff check    # No lint errors
uv run pyright       # Type checks pass
```
