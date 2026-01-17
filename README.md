# ARES

ARES (Agentic Research and Evaluation Suite) is an RL-first framework for training and evaluating agents.

## Quick Start

Get ARES running in minutes - no API keys required!

### Prerequisites

- **Python 3.12 or higher**
- **[Docker](https://docs.docker.com/get-docker/)** - For running code agents in containers
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package installer and resolver

To install `uv`, follow the instructions at https://docs.astral.sh/uv/getting-started/installation/

### Installation

For now, we recommend running ARES locally from this directory:

```bash
uv sync --all-groups
```

and you're ready to get started.

### Your First ARES "Agent"
No API keys needed!

Run the Hello World example to see the RL loop in action:

```bash
uv run -m examples.00_hello_world
```

This example uses:
- ✓ Local Docker containers (no cloud account needed)
- ✓ A mock LLM (no API keys needed)

You'll see how ARES treats code agent interactions as a reinforcement learning problem, with LLM requests as observations and LLM responses as actions.

## Examples

ARES includes several examples that demonstrate different usage patterns:

### 1. Minimal Loop (Local Docker + Real LLM)
**File:** `examples/01_minimal_loop.py`
**What you'll need:** Docker, Martian API key

```bash
# Set up your API key first (see Cloud Setup below)
uv run -m examples.01_minimal_loop
```

Shows the RL loop with a real LLM (via Martian API) and local Docker containers.

### 2. Local LLM (Fully Local)
**File:** `examples/02_local_llm.py`
**What you'll need:** Docker

```bash
uv run -m examples.02_local_llm
```

Demonstrates running ARES completely locally using a local LLM (Qwen2.5-3B-Instruct). No cloud services required.

## Cloud Setup (Optional)

For production use or larger-scale experiments, you can use cloud containers and API-based LLMs.

### Option 1: Using Martian API for LLM Inference

1. Create an account at [https://app.withmartian.com](https://app.withmartian.com)
2. Copy the example environment file: `cp .env.example .env`
3. Add your Martian API key: `CHAT_COMPLETION_API_KEY=your_key_here`

### Option 2: Using Daytona for Cloud Containers

By default, ARES uses Daytona for container management. To set this up:

1. Create a Daytona account at [https://www.daytona.io](https://www.daytona.io/)
2. Copy the example environment file: `cp .env.example .env`
3. Add your Daytona credentials:
   - `DAYTONA_API_KEY=your_key_here`
   - `DAYTONA_API_URL=your_url_here`

See `.env.example` for all available configuration options.

## API Usage

ARES environments use an async version of the [dm_env](https://github.com/google-deepmind/dm_env) spec. Here's a complete example:

```python
import asyncio

from ares.code_agents import mini_swe_agent
from ares.containers import docker  # Use local Docker, or import daytona for cloud
from ares.environments import swebench_env
from ares.llms import chat_completions_compatible


async def main():
    # Create an LLM client (requires CHAT_COMPLETION_API_KEY in .env)
    llm_client = chat_completions_compatible.ChatCompletionCompatibleLLMClient(
        model="openai/gpt-4o-mini"
    )

    # Load SWE-bench tasks
    all_tasks = swebench_env.swebench_verified_tasks()
    tasks = [all_tasks[0]]  # Run on only one task for now

    # Create environment with local Docker and MiniSWE agent
    async with swebench_env.SweBenchEnv(
        tasks=tasks,
        container_factory=docker.DockerContainer,  # Use local Docker
        code_agent_factory=mini_swe_agent.MiniSWECodeAgent,
    ) as env:
        # The RL loop
        ts = await env.reset()
        while not ts.last():
            # Environment sends observation (LLM request) to agent
            action = await llm_client(ts.observation)

            # Environment processes action (LLM response) and returns next state
            ts = await env.step(action)
            print(f"Step complete. Reward: {ts.reward}")


if __name__ == "__main__":
    asyncio.run(main())
```

This example uses:
- **Container backend:** Local Docker (change to `daytona.DaytonaContainer` for cloud)
- **LLM backend:** Martian API (or any OpenAI-compatible API)
- **Code agent:** MiniSWE agent from the mini-swe-agent library