<p align="center">
  <img src="ares_logo.png" width="100" alt="Ares Logo"/>
</p>

<h1 align="center">Ares</h1>

<p align="center">
  🚀 A high-performance, modern framework for training and evaluating code agents with reinforcement learning
</p>

<p align="center">
  <a href="https://github.com/withmartian/ares/stargazers">
    <img src="https://img.shields.io/github/stars/withmartian/ares?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/withmartian/ares/issues">
    <img src="https://img.shields.io/github/issues/withmartian/ares" alt="GitHub issues"/>
  </a>
</p>

---

ARES (Agentic Research and Evaluation Suite) is an RL-first framework for training and evaluating agents. ARES environments use an async version of the [dm_env](https://github.com/google-deepmind/dm_env) spec, making it easy to build and evaluate code agents using reinforcement learning.

## Quick Start (No Accounts or API Keys Required!)

Get started in minutes with local Docker containers and a local LLM - no accounts or API keys needed.

### Prerequisites

- **Python 3.12 or higher**
- **[uv](https://docs.astral.sh/uv/)** - Fast Python package installer and resolver
- **Docker** - For running local containers (install from [docker.com](https://www.docker.com/get-started))

#### Installing uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/withmartian/ares.git
   cd ares
   ```

2. **Install dependencies:**
   ```bash
   uv sync --all-groups
   ```
   
   This installs all dependencies including example dependencies needed for the quick start.

3. **Run your first example (no API keys needed!):**
   ```bash
   uv run -m examples.02_local_llm
   ```

   This example uses:
   - **Local Docker containers** (no cloud account needed)
   - **Local LLM** (Qwen2.5-3B-Instruct, downloaded automatically, no API keys needed)
   - **No `.env` file required** - works out of the box!

   The example will download the model on first run (~6GB), then run a SWE-bench task in a local Docker container.

## Installation Methods

### Local Development (Recommended for Getting Started)

Run ARES directly from the cloned repository:

```bash
uv sync --all-groups
```

This installs:
- Main dependencies
- Development tools (pytest, ruff, pyright)
- Example dependencies (transformers for local LLM support)

### As a Dependency

You can also include ARES as a dependency in your own project's `pyproject.toml` using a relative path:

```toml
[project]
dependencies = [
    "ares @ {path = "../ares", develop = true}",
]
```

**Note:** PyPI installation will be available soon.

## Examples

ARES comes with two example scripts to help you get started:

### Example 1: Local LLM (No API Keys Required) ⭐ Recommended for First Use

```bash
uv run -m examples.02_local_llm
```

**What it does:**
- Uses local Docker containers (no cloud services)
- Uses a local LLM model (Qwen2.5-3B-Instruct)
- Runs a SWE-bench task
- **No accounts or API keys needed!**

**Requirements:**
- Docker installed and running
- ~6GB disk space for the model (downloaded automatically)

### Example 2: Cloud LLM with Local Docker

```bash
uv run -m examples.01_minimal_loop
```

**What it does:**
- Uses local Docker containers (no Daytona account needed)
- Uses a cloud LLM API (requires API key)

**Requirements:**
- Docker installed and running
- API key for LLM service (see [Configuration](#configuration) below)
- **Note:** This example requires a `.env` file with `CHAT_COMPLETION_API_KEY` set

## Configuration

### Using Cloud Services (Optional)

ARES supports cloud services for containers and LLM inference. These are **optional** - you can use ARES entirely with local Docker and local LLMs.

#### Using Cloud LLM APIs

If you want to use cloud LLM APIs (like Martian, OpenAI, etc.) instead of local models, you'll need to set up API keys:

1. **Create a `.env` file** in the repository root:
   ```bash
   touch .env
   ```

2. **Add your API key** (required for `ChatCompletionCompatibleLLMClient`):
   ```bash
   CHAT_COMPLETION_API_KEY=your_api_key_here
   ```

3. **Optionally customize the API base URL** (defaults to Martian):
   ```bash
   CHAT_COMPLETION_API_BASE_URL=https://api.withmartian.com/v1
   ```

**Important:** The local LLM example (`examples/02_local_llm.py`) does **not** require this configuration and works without any `.env` file or API keys.

#### Using Daytona for Cloud Containers (Optional)

By default, ARES uses local Docker containers. If you want to use Daytona cloud containers instead:

1. **Create a Daytona account** at [https://www.daytona.io](https://www.daytona.io/)

2. **Generate an API key** from your Daytona account

3. **Add to your `.env` file:**
   ```bash
   DAYTONA_API_KEY=your_daytona_api_key
   DAYTONA_API_URL=your_daytona_api_url
   ```

4. **Use Daytona containers** in your code:
   ```python
   from ares.containers import daytona
   from ares.environments import swebench_env
   
   async with swebench_env.SweBenchEnv(
       tasks=tasks,
       container_factory=daytona.DaytonaContainer,  # Use Daytona instead of Docker
   ) as env:
       # ... your code
   ```

## Usage

### Basic Example with Local Docker

Here's a minimal example using local Docker containers (no cloud services needed):

```python
import asyncio
from ares.code_agents import mini_swe_agent
from ares.containers import docker
from ares.environments import swebench_env
from ares.llms import chat_completions_compatible

async def main():
    # Create an LLM client (requires API key for cloud LLMs)
    # For local LLMs, see examples/02_local_llm.py
    agent = chat_completions_compatible.ChatCompletionCompatibleLLMClient(
        model="openai/gpt-5-mini"
    )
    
    # Load SWE-bench tasks
    all_tasks = swebench_env.swebench_verified_tasks()
    tasks = [all_tasks[0]]  # Run on one task
    
    # Use local Docker containers (no cloud account needed)
    async with swebench_env.SweBenchEnv(
        tasks=tasks,
        container_factory=docker.DockerContainer,  # Use local Docker
        code_agent_factory=mini_swe_agent.MiniSWECodeAgent,
    ) as env:
        ts = await env.reset()
        
        while not ts.last():
            # Agent processes observation and returns action
            action = await agent(ts.observation)
            
            # Environment steps with action
            ts = await env.step(action)
            
            print(f"Step reward: {ts.reward}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Using Local LLMs

For a complete example using local LLMs (no API keys), see `examples/02_local_llm.py`. This example:
- Downloads and runs a local model (Qwen2.5-3B-Instruct)
- Uses local Docker containers
- Requires no external accounts or API keys

## Architecture

ARES provides a clean RL abstraction for code agents:

- **Environments**: Async dm_env-compatible environments (SWE-bench, Harbor, etc.)
- **Containers**: Pluggable container backends (Docker, Daytona)
- **LLM Clients**: Flexible LLM interfaces (local models, OpenAI-compatible APIs)
- **Code Agents**: Agents that interact with codebases in containers

The framework uses a queue-mediated LLM client pattern that allows agents to be written naturally (making direct LLM calls) while the environment controls the RL loop.

## Requirements Summary

| Feature | Local Setup | Cloud Setup |
|---------|------------|-------------|
| **Containers** | Docker (local) | Daytona account + API key |
| **LLM** | Local model (transformers) | API key (Martian, OpenAI, etc.) |
| **Accounts** | None | Daytona + LLM provider |
| **Setup Time** | ~5 minutes | ~10-15 minutes |

**Recommendation:** Start with the local setup (Docker + local LLM) to get familiar with ARES, then optionally add cloud services as needed.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and contribution guidelines.

## License

See [LICENSE](LICENSE) for details.
