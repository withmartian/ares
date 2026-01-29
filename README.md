<h1 align="center">ARES: Agentic Research & Evaluation Suite</h1>

<p align="center">
  <a href="https://martian-ares.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-readthedocs-blue.svg" alt="Documentation"></a>
  <a href="https://pypi.org/project/martian-ares/"><img src="https://img.shields.io/pypi/v/martian-ares.svg" alt="PyPI version"></a>
  <a href="https://github.com/withmartian/ares/blob/main/LICENSE"><img src="https://img.shields.io/github/license/withmartian/ares.svg" alt="License"></a>
  <a href="https://discord.gg/TFfMTrzw"><img src="https://img.shields.io/badge/discord-join-5865F2?logo=discord" alt="Discord"></a>
</p>

<p align="center">
  <img margin="auto" width="auto" height="312" alt="image" src="https://github.com/user-attachments/assets/ae34ab36-b78f-48de-93c9-01d611a547e3" />
</p>

ARES is an RL-first framework for training and evaluating LLM agents, especially coding agents.

It is a modern [gym](https://github.com/Farama-Foundation/Gymnasium): the environment layer powering RL research.

ARES treats LLMRequests as observations and LLMResponses as actions within the environment, so you can focus on training just the LLM - not the Code Agent surrounding it. The interface is entirely async, and supports scaling up to hundreds or thousands of parallel environments easily - check out [example 3](https://github.com/withmartian/ares/tree/main/examples/03_parallel_eval_with_api.py) to run this yourself.


## Quick Start

### Pre-requisites

- Python >= 3.12

### Getting Started

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```
uv add martian-ares
```

ARES comes packaged with useful presets for different code agent & environment configurations. List them with:

```
uv run python -c "import ares; print(ares.list_presets())"
```

You can get started by using this minimal loop to run mini-swe-agent on SWE-bench Verified sequentially.

Note: to run this particular example you will need:

- Docker (with the daemon running)
- A Martian API key (see below)

```python
import asyncio

import ares
from ares import llms

async def main():
    # This requires `CHAT_COMPLETION_API_KEY` to be set with a Martian API key--see below.
    agent = llms.ChatCompletionCompatibleLLMClient(model="openai/gpt-5-mini")

    async with ares.make("sbv-mswea") as env:
        ts = await env.reset()
        while not ts.last():
            action = await agent(ts.observation)   # observation = LLM request
            ts = await env.step(action)            # action = LLM response
            print(f"{action}\n{ts}")

if __name__ == "__main__":
    asyncio.run(main())
```

To run the example above you'll need a Martian API key set in your `.env` file. To get a key:

1) Go to https://app.withmartian.com
1) on the `Billing` tab, add a payment method + top up some credits.
1) on the `API Keys` tab create an API key.
1) write `CHAT_COMPLETION_API_KEY={your-key}` in your `.env`

Alternatively, you can use another chat completions-compatible endpoint by setting both:

- `CHAT_COMPLETION_API_BASE_URL`
- `CHAT_COMPLETION_API_KEY`

### Next Steps

1. Check out the [examples](https://github.com/withmartian/ares/tree/main/examples)
1. Read the [docs](https://martian-ares.readthedocs.io/en/latest/) to understand ARES and its key abstractions
1. Read our [blog post](https://withmartian.com/post/ares-open-source-infrastructure-for-online-rl-on-coding-agents) about why ARES and what we hope to see
