# ARES Examples

This directory contains pedagogical examples demonstrating how to use ARES to evaluate code agents. Each example builds on the previous one, introducing new concepts and capabilities.

## Prerequisites

Each example will list its prerequisites in its own docstring.
It's helpful for most of the examples to have both a Martian API key (for an LLM inference API) and a Daytona API key (for cloud sandboxes).

### Martian

1) sign up at https://app.withmartian.com.
1) on the `Billing` tab, add a payment method + top up some credits.
1) on the `API Keys` tab create an API key.
1) write `CHAT_COMPLETION_API_KEY={your-key}` in your `.env`

### Daytona

1) Sign up at https://www.daytona.io.
1) Go to API Keys and create an API key.
1) write `DAYTONA_API_KEY={your-key}` in your `.env`
1) write `DAYTONA_API_URL={api-url}` in your `.env`

### Docker

For some examples, [docker](https://docs.docker.com/engine/install/) should be installed and accessible.

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
- **03_parallel_eval_with_api.py** - Demonstrates parallel evaluation of all SWE-bench Verified tasks using cloud containers (Daytona) for scalability.
- **05_tinker_train.py** - Shows how to train code agents using RL with ARES environments and Tinker's training infrastructure, featuring LoRA fine-tuning, async parallel rollouts, and WandB logging.
