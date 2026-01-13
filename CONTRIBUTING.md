# Contributing to ARES

Thank you for your interest in contributing to the Agentic Research and Evaluation Suite!

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installing uv

If you don't have `uv` installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Setting Up Your Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/withmartian/ares.git
   cd ares
   ```

2. Install all dependencies, including dev tools:
   ```bash
   uv sync --all-groups
   ```

   This installs the main dependencies plus development tools (pytest, ruff, pyright) and optional example dependencies.

## Running Tests

```bash
uv run pytest
```

Tests are located in the `src/` directory following the patterns `test_*.py` or `*_test.py`.

## Code Quality

### Linting

Check for code issues:
```bash
uv run ruff check
```

### Formatting

Format your code:
```bash
uv run ruff format
```

### Type Checking

Run static type analysis:
```bash
uv run pyright
```

## Environment Variables

Some functionality may require environment variables for API keys or configuration. Check the codebase for specific requirements and create a `.env` file in the repository root as needed.

## Pull Request Guidelines

Before submitting a PR:

1. **Run all checks locally:**
   - [ ] Tests pass (`uv run pytest`)
   - [ ] Code is formatted (`uv run ruff format`)
   - [ ] No linting issues (`uv run ruff check`)
   - [ ] Type checks pass (`uv run pyright`)

2. **Write a clear PR description:**
   - Explain what problem you're solving
   - Link to any related issues
   - Describe your approach if non-obvious

3. **Keep commits focused:**
   - Each commit should represent a logical change
   - Use clear, descriptive commit messages

## Code Style Notes

- **Comments should explain WHY, not WHAT:** The code itself should be clear about what it does. Use comments to explain the reasoning, edge cases, or non-obvious decisions.
- **Comments explaining HOW:** Only add these when the implementation is genuinely complex or uses a non-standard approach that might confuse future maintainers.
- **Follow existing patterns:** Look at similar code in the repository to match the established style.

## Questions?

If you have questions or need help, feel free to open an issue or reach out to the maintainers.
