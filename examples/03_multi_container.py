"""Example demonstrating multi-container distributed agent communication.

This example shows how to run multiple CodeAgents in separate containers,
all communicating with the local RL training loop via a PubSub proxy.

Architecture:
- PubSubContainer: Runs FastAPI proxy that brokers LLM requests
- Agent Containers: Run agents that make HTTP requests to proxy
- Local Machine: Polls proxy for requests, provides responses

Example usage:
    uv run -m examples.03_multi_container
"""

import asyncio

from ares.llms import chat_completions_compatible
from ares.llms import llm_clients
from ares.pubsub_proxy import client as pubsub_client
from ares.pubsub_proxy import container as pubsub_container


async def simulate_agent_making_request(proxy_url: str, agent_id: str) -> None:
    """Simulate an agent making an LLM request via the proxy.

    In a real scenario, this would be a CodeAgent running in a container
    making HTTP requests via an OpenAI client library.

    Args:
        proxy_url: Base URL of the PubSub proxy
        agent_id: Identifier for this agent
    """
    import httpx

    print(f"[{agent_id}] Making LLM request to proxy at {proxy_url}")

    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{proxy_url}/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful coding assistant."},
                    {"role": "user", "content": f"Hello from {agent_id}! What is 2+2?"},
                ],
                "temperature": 0.0,
            },
            timeout=60.0,
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"[{agent_id}] Received response: {content[:100]}")
        else:
            print(f"[{agent_id}] Error: {response.status_code} - {response.text}")


async def main():
    """Run a simple multi-agent example."""
    print("=" * 80)
    print("ARES Multi-Container Example")
    print("=" * 80)
    print()

    # 1. Start the PubSub proxy container
    print("Step 1: Starting PubSub proxy container...")
    proxy = pubsub_container.PubSubContainer(name="ares-pubsub-proxy", port=8000)
    await proxy.start()
    proxy_url = proxy.get_base_url()
    print(f"PubSub proxy running at {proxy_url}")
    print()

    # 2. Create the PubSub-mediated LLM client for local consumption
    print("Step 2: Setting up local LLM client...")
    mediated_client = pubsub_client.PubSubMediatedLLMClient(proxy_url=proxy_url)
    await mediated_client.start_polling()
    print("Local client polling for requests")
    print()

    # 3. Create the actual LLM client that will provide responses
    print("Step 3: Creating LLM client for responses...")
    llm = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="openai/gpt-4o-mini")
    print("LLM client ready")
    print()

    try:
        # 4. Simulate multiple agents making requests concurrently
        print("Step 4: Simulating 3 agents making concurrent requests...")
        print("-" * 80)

        # Start agent request tasks
        agent_tasks = [
            asyncio.create_task(simulate_agent_making_request(proxy_url, f"Agent-{i}")) for i in range(1, 4)
        ]

        # Handle requests from the queue (like Environment would)
        requests_handled = 0
        while requests_handled < 3:
            # Get request from queue (this is what Environment.reset() or .step() does)
            req_and_future = await mediated_client.q.get()
            requests_handled += 1

            print(f"\n[Local] Received request {requests_handled}/3")
            print(f"[Local] Messages: {len(req_and_future.value.messages)} messages")

            # Generate response using the LLM (this is what the RL agent does)
            response = await llm(req_and_future.value)

            # Provide response (this is what Environment.step(action) does)
            req_and_future.future.set_result(response)
            print(f"[Local] Sent response for request {requests_handled}/3")

        # Wait for all agent tasks to complete
        await asyncio.gather(*agent_tasks)

        print()
        print("-" * 80)
        print("All requests handled successfully!")

    finally:
        # Clean up
        print()
        print("Cleaning up...")
        await mediated_client.stop_polling()
        await mediated_client.close()
        await proxy.stop()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
