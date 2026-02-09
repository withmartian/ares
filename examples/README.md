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
- **06_mi_linear_probe.py** - Pedagogical mechanistic interpretability example showing how to record activations and train linear probes to interpret agent behavior. Uses synthetic data by default for easy experimentation without heavy compute.
- **07_mi_swebench_correctness_probe.py** - Mechanistic interpretability example focused on SWE-bench Verified agents. Demonstrates probing correctness beliefs in multi-turn agent trajectories, comparing self-reported confidence vs internal representations. Explores whether smaller models (7B/8B) have calibrated internal beliefs even when their confidence may be miscalibrated. Includes mock mode with synthetic data and real mode integration stubs.

## Mechanistic Interpretability Examples

The MI examples (06 and 07) demonstrate how to use linear probes to understand what language models "know" internally:

**06_mi_linear_probe.py** - Basic introduction to activation recording and linear probing:
- Single-step predictions from activations
- Binary classification probe training
- Evaluation with accuracy and AUC metrics
- Weight interpretation to identify important dimensions

**07_mi_swebench_correctness_probe.py** - Advanced multi-turn trajectory analysis:
- Multi-turn agent trace generation with confidence labels
- Correctness belief probing across timesteps
- Comparison of self-reported confidence vs probe predictions
- Visualization of calibration across agent trajectories
- Integration patterns for real SWE-bench Verified runs

Both examples are designed to be CPU-friendly and runnable without GPU infrastructure. They use sklearn for robust probe training and support caching for iterative experimentation.
