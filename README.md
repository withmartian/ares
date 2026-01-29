<h1 align="center">ARES: Agentic Research & Evaluation Suite</h1>

<p align="center">
  <a href="https://martian-ares.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/docs-readthedocs-blue.svg" alt="Documentation"></a>
  <a href="https://pypi.org/project/martian-ares/"><img src="https://img.shields.io/pypi/v/martian-ares.svg" alt="PyPI version"></a>
  <a href="https://github.com/withmartian/ares/blob/main/LICENSE"><img src="https://img.shields.io/github/license/withmartian/ares.svg" alt="License"></a>
</p>

<p align="center">
  <img margin="auto" width="auto" height="312" alt="image" src="https://github.com/user-attachments/assets/ae34ab36-b78f-48de-93c9-01d611a547e3" />
</p>

ARES is an RL-first framework for training and evaluating LLM agents, especially coding agents.

It is a modern [gym](https://github.com/Farama-Foundation/Gymnasium): the environment layer powering RL research.


## Quick Start

### Pre-requisites

- Python >= 3.12

### Getting Started

Install with uv:

```
uv add martian-ares
```

ARES comes packaged with useful presets for different agent/environment configurations. List them with:

```
uv run python -c "import ares; ares.list_presets()"
```

You can get started by using this minimal loop to run mini-swe-agent on SWE-bench Verified sequentially.

Note: this example uses a local LLM. You'll need to install additional optional dependencies:
```
uv add martian-ares[llamacpp]
```

```
import asyncio

import ares
from ares.contrib import llama_cpp

async def main():
    agent = llama_cpp.create_qwen2_0_5b_instruct_llama_cpp_client()

    async with ares.make("sbv-mswea") as env:
        ts = await env.reset()
        while not ts.last():
            action = await agent(ts.observation)   # observation = LLM request
            ts = await env.step(action)            # action = LLM response

if __name__ == "__main__":
    asyncio.run(main())
```

That's it!

### Next Steps

1. Check out the [examples](https://github.com/withmartian/ares/tree/main/examples)
1. Read the [docs](https://martian-ares.readthedocs.io/en/latest/)
