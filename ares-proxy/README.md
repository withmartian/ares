# ares-proxy

A lightweight HTTP proxy server that intercepts OpenAI-compatible chat completion requests and routes them through a queue-mediated polling system. Designed for use with the ARES (Agentic Research and Evaluation Suite) framework to enable RL-based control of LLM interactions.

The proxy doesn't validate the request/response types; this should be handled by the clients.

## Overview

ares-proxy acts as a man-in-the-middle proxy between LLM clients (like code agents) and LLM providers. Instead of directly forwarding requests to an LLM API, it:

1. **Queues incoming requests** from clients
2. **Exposes requests via polling** to external controllers
3. **Routes responses back** to waiting clients

This architecture enables the ARES RL loop to intercept and control LLM interactions without agents needing to be RL-aware.

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────────┐
│ LLM Client  │────────▶│  ares-proxy  │◀────────│ ARES Controller │
│ (Agent)     │         │              │         │ (RL Environment)│
└─────────────┘         └──────────────┘         └─────────────────┘
     │                         │                          │
     │ POST /v1/chat/          │ GET /poll                │
     │   completions           │ (retrieve requests)      │
     │ (blocks waiting)        │◀─────────────────────────┤
     │                         │                          │
     │                         │ POST /respond            │
     │                         │ (send response)          │
     │◀────────────────────────┤◀─────────────────────────┤
     │ (receives response)     │                          │
```

### Components

#### Broker (`broker.go`)
The core coordination engine that manages:
- **Request queue**: Holds pending LLM requests
- **Response channels**: Maps request IDs to response delivery channels
- **Timeout handling**: Cleans up stale requests after a configurable timeout (i.e. removes them)

#### HTTP Endpoints (`main.go`)

1. **`POST /v1/chat/completions`**
   - OpenAI-compatible endpoint for LLM clients
   - Accepts standard chat completion requests
   - Blocks until a response is available or timeout occurs
   - Returns the response as JSON

2. **`GET /poll`**
   - Retrieves all pending requests from the queue
   - Clears the queue atomically
   - Returns an array of pending requests, each containing:
     - `id`: Unique request identifier
     - `timestamp`: When the request was submitted
     - `request`: The original request payload (chat completion JSON)

3. **`POST /respond`**
   - Sends a response back to a waiting request
   - Requires request ID and response payload
   - Returns error if request ID not found (e.g., timed out)

## Configuration

Configure via environment variables:

| Variable          | Description                      | Default |
|-------------------|----------------------------------|---------|
| `PORT`            | HTTP server port                 | `8080`  |
| `TIMEOUT_MINUTES` | Request timeout in minutes       | `15`    |

Example:
```bash
export PORT=9000
export TIMEOUT_MINUTES=30
./ares-proxy
```

## Usage

### Running the Server

```bash
# Using Makefile (recommended)
make build   # Build the binary
make run     # Run with defaults (port 8080, 15 min timeout)
make test    # Run all tests
make clean   # Remove build artifacts

# Or manually
go build -o ares-proxy
./ares-proxy

# Run with custom configuration
PORT=9000 TIMEOUT_MINUTES=30 ./ares-proxy
```

### Client Usage

Configure your LLM client to point at the proxy:

```python
from openai import OpenAI

# Point the client at ares-proxy instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy"  # Proxy doesn't validate keys
)

# Use normally - this will block until proxy receives a response
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Controller Usage

External controllers (like ARES) poll for requests and send responses:

```python
import requests
import json

# Poll for pending requests
response = requests.get("http://localhost:8080/poll")
requests_list = response.json()

for req in requests_list:
    request_id = req["id"]
    request_body = req["request"]

    # Process the request (e.g., send to real LLM)
    llm_response = process_request(request_body)

    # Send response back to waiting client
    requests.post(
        "http://localhost:8080/respond",
        json={
            "id": request_id,
            "response": llm_response
        }
    )
```

## Development

### Running Tests

```bash
# Using Makefile (recommended)
make test

# Or manually
go test -v                                  # Run all tests
go test -v -run TestBroker_SubmitAndPoll   # Run specific test
go test -cover                              # Run with coverage
```

Tests run automatically in CI via GitHub Actions when any files in `ares-proxy/` are modified.

### Project Structure

```
ares-proxy/
├── main.go         # HTTP server and endpoint handlers
├── broker.go       # Core request/response coordination logic
├── broker_test.go  # Unit tests for broker
├── config.go       # Configuration loading
├── types.go        # Data structures (PendingRequest, RespondRequest)
├── go.mod          # Go module definition
└── README.md       # This file
```

## Integration with ARES

ares-proxy is designed to work with ARES's `QueueMediatedLLMClient`. The integration flow:

1. Code agent makes LLM call → routed to ares-proxy
2. ares-proxy queues request and blocks
3. ARES environment polls proxy for requests
4. ARES converts request to RL observation
5. Agent/policy returns action (LLM response)
6. ARES posts response back to proxy
7. Proxy unblocks and returns response to code agent

This architecture allows ARES to treat LLM interactions as part of the RL loop without agents needing to be modified.

## Error Handling

### Timeouts
If no response is received within `TIMEOUT_MINUTES`, the client request returns an error:
```
request timeout after 15m0s
```

### Invalid Request IDs
If responding to a non-existent or timed-out request:
```
request ID abc-123 not found (may have timed out)
```

### Context Cancellation
If the client disconnects, the request is cleaned up automatically and returns:
```
context canceled
```

## Concurrency

ares-proxy is fully concurrent and thread-safe:
- Multiple clients can submit requests simultaneously
- Polling and responding can happen concurrently
- All shared state is protected by mutexes
- Tested with concurrent workloads (see `broker_test.go`)

## Performance

The proxy is designed for low latency:
- Minimal processing overhead (just queuing/routing)
- Buffered channels prevent blocking on response delivery
- Atomic queue operations minimize lock contention
- No external dependencies (stdlib only)

## License

Part of the ARES project. See parent repository for license information.
