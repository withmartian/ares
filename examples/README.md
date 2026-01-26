# ARES Examples

This directory contains pedagogical examples demonstrating how to use ARES to evaluate code agents. Each example builds on the previous one, introducing new concepts and capabilities.

## Running Examples

Examples are **not packaged in the ARES wheel** and must be run from the ARES repository directory:

```bash
# Clone the repository
git clone https://github.com/yourusername/ares.git
cd ares

# Install dependencies (including examples group)
uv sync --group examples

# Run an example
uv run -m examples.01_sequential_eval_with_local_llm
```

Each example file contains a detailed docstring at the top explaining what it does, its prerequisites, and usage instructions.

## Examples

- **01_sequential_eval_with_local_llm.py** - Demonstrates the basic ARES RL loop using a local LLM (Qwen2-0.5B) and local Docker containers.
- **02_sequential_eval_with_api.py** - Shows how to swap the local LLM for a frontier model accessed via API (GPT-5-mini through Martian).
- **03_parallel_eval_with_api.py** - Demonstrates parallel evaluation of all SWEBench Verified tasks using cloud containers (Daytona) for scalability.
