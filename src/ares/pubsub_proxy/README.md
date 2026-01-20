# PubSub Proxy for Distributed Multi-Container Agents

This module implements an OpenAI-compatible HTTP proxy that enables distributed RL training with multiple CodeAgents running in separate containers.

## Architecture

```
Agent Container 1         PubSub Container             Local Machine
┌──────────────┐         ┌──────────────────┐        ┌──────────────┐
│ CodeAgent    │  HTTP   │  FastAPI Proxy   │  HTTP  │ RL Training  │
│ (any impl)   ├────────>│  - In-memory     ├<───────┤ Loop         │
│              │  POST   │    queues        │  Poll/ │              │
│ base_url=    │  /v1/   │  - OpenAI        │  Post  │ Environment  │
│ "http://     │  chat/  │    compatible    │        │   .reset()   │
│ pubsub:8000" │  compl. │                  │        │   .step()    │
└──────────────┘         └──────────────────┘        └──────────────┘

Agent Container 2
┌──────────────┐
│ CodeAgent    │  HTTP
│ (any impl)   ├─────────┐
└──────────────┘         │
                         ▼
     ...            (same proxy)
```

## Components

### 1. PubSubContainer
Manages a Docker container running a FastAPI proxy that bridges agent HTTP requests to the local RL loop.

**Key Features:**
- Builds Docker image with FastAPI + uvicorn
- Exposes OpenAI-compatible `/v1/chat/completions` endpoint
- Handles port mapping for local access
- Includes health checks

### 2. PubSubMediatedLLMClient
Local client that polls the proxy for requests and posts responses back.

**Key Features:**
- Background polling loop using long-polling HTTP
- Exposes requests via `asyncio.Queue` (compatible with existing Environment pattern)
- Handles request/response correlation via unique IDs
- OpenAI response format conversion

### 3. FastAPI Proxy (proxy.py)
The actual HTTP server running inside the PubSub container.

**Endpoints:**
- `POST /v1/chat/completions` - Agent requests (blocks until response)
- `GET /requests?timeout=30` - Local polling endpoint (long-polling)
- `POST /responses/{request_id}` - Local response submission
- `GET /health` - Health check

**In-Memory State:**
- `asyncio.Queue` for pending requests
- `dict[request_id, Future]` for response coordination

## Usage

### Basic Example

See `examples/03_multi_container.py` for a complete working example.

```python
from ares.pubsub_proxy import container, client
from ares.llms import chat_completions_compatible

# 1. Start PubSub proxy container
proxy = container.PubSubContainer(name="ares-pubsub-proxy", port=8000)
await proxy.start()

# 2. Create mediated client for local consumption
mediated_client = client.PubSubMediatedLLMClient(proxy_url="http://localhost:8000")
await mediated_client.start_polling()

# 3. Configure agents to use proxy as base_url
# In agent containers, set: OPENAI_BASE_URL="http://pubsub-container:8000/v1"

# 4. Handle requests in RL loop
llm = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model="...")

while True:
    # Get request from queue (like Environment.reset() or .step())
    req_and_future = await mediated_client.q.get()

    # Generate response using RL policy/LLM
    response = await llm(req_and_future.value)

    # Provide response (like Environment.step(action))
    req_and_future.future.set_result(response)
```

### Agent Configuration

Agents running in containers just need their `base_url` pointed to the proxy:

```python
# For OpenAI client
import openai
client = openai.AsyncOpenAI(
    base_url="http://pubsub-container:8000/v1",
    api_key="dummy",  # Not validated by proxy
)

# For Harbor agents
export OPENAI_BASE_URL="http://pubsub-container:8000/v1"
harbor-agent run --task "..."
```

### Integration with Environment

For integration with ARES environments, you can:

1. Start `PubSubContainer` in `_start_container()`
2. Use `PubSubMediatedLLMClient` instead of `QueueMediatedLLMClient`
3. Configure agent containers with `OPENAI_BASE_URL` env var
4. Rest of the RL loop remains the same (reset/step still use the queue)

## Design Decisions

### Why HTTP + In-Memory vs Redis?

**Chosen: HTTP with in-memory queues**
- ✓ Simpler (single process, no Redis dependency)
- ✓ Easier to debug (all state in FastAPI app)
- ✓ Sufficient for RL training use cases
- ✓ Long-polling is fast enough

**Alternative: Redis Pub/Sub**
- More complex (separate Redis process)
- Better for persistent state across restarts
- Overkill for ephemeral RL episodes

### Why OpenAI-Compatible API?

- ✓ **Zero agent modification** - Any agent using OpenAI client works
- ✓ Standard protocol - Well-documented, battle-tested
- ✓ Easy testing - Can use curl or any HTTP client

### Networking: Docker vs Daytona

**Local Docker:**
- Containers communicate via Docker network names
- Local machine connects via `localhost:{port}`
- Port mapping: container's 8000 → host's configurable port

**Daytona (Cloud):**
- PubSub container runs in cloud (Daytona)
- Agent containers connect via container-to-container networking
- Local machine connects outbound to cloud (allowed through firewalls)
- Update `get_base_url()` for Daytona-specific addressing

## Configuration

### Timeout Configuration

The proxy uses a default 300-second timeout for agent requests waiting for responses. To configure per-environment:

```python
# In proxy.py, modify:
DEFAULT_TIMEOUT_S = 300.0  # Adjust as needed

# Or make it configurable via environment variable:
import os
timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
```

### Port Configuration

```python
proxy = PubSubContainer(
    name="my-proxy",
    port=8000,  # Host port for local access
)
```

## Testing

Run the test suite:

```bash
uv run pytest src/ares/pubsub_proxy/proxy_test.py -v
```

Tests cover:
- Health endpoint functionality
- Full request/response flow
- Concurrent multi-agent requests

## Scaling Considerations

**Current implementation handles:**
- 1-5 containers: ✓ Works great
- 100-200 containers: ✓ Should work (FastAPI is async)
- 1000+ containers: May need optimization (connection pooling, Redis, etc.)

**For large-scale deployments, consider:**
- Using Redis instead of in-memory queues
- Running multiple proxy instances with load balancing
- Implementing connection pooling and request batching
- Adding metrics and monitoring (Prometheus, etc.)

## Future Enhancements

Potential improvements (not yet implemented):

1. **Daytona Support** - Update `PubSubContainer` to support Daytona cloud containers
2. **Configurable Timeout** - Make timeout configurable per-agent or per-request
3. **Request Batching** - Batch multiple requests for efficiency
4. **Redis Backend** - Optional Redis backend for persistence
5. **Metrics** - Add Prometheus metrics for monitoring
6. **Rate Limiting** - Per-agent rate limiting
7. **Auth** - Optional authentication for agents

## Troubleshooting

### Container won't start
- Check Docker is running: `docker ps`
- Check port isn't already in use: `lsof -i :8000`
- Check container logs: `docker logs <container-name>`

### Health check times out
- Verify port mapping is correct (8000 inside → configured port outside)
- Check firewall rules
- Increase timeout in `_wait_for_ready()`

### Requests not reaching local machine
- Verify polling is started: `await mediated_client.start_polling()`
- Check proxy URL is correct
- Look for errors in polling loop logs

### Agent requests time out
- Verify RL loop is consuming from queue: `await mediated_client.q.get()`
- Check timeout configuration (default 300s)
- Ensure responses are being provided: `future.set_result(response)`
