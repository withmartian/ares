package main

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Broker manages pending requests and coordinates responses
type Broker struct {
	// pendingRequests maps request ID to a response channel
	pendingRequests map[string]chan json.RawMessage
	// requestQueue holds requests waiting to be polled
	requestQueue []PendingRequest
	// mu protects both maps from concurrent access
	mu sync.Mutex
	// timeout for how long to wait for a response
	timeout time.Duration
}

// NewBroker creates a new broker with the given timeout
func NewBroker(timeout time.Duration) *Broker {
	return &Broker{
		pendingRequests: make(map[string]chan json.RawMessage),
		requestQueue:    make([]PendingRequest, 0),
		timeout:         timeout,
	}
}

// SubmitRequest adds a request to the queue and waits for a response
// This is called by the /v1/chat/completions endpoint
func (b *Broker) SubmitRequest(ctx context.Context, requestBody json.RawMessage) (json.RawMessage, error) {
	// Generate a unique ID for this request
	id := uuid.New().String()

	// Create a channel to receive the response
	responseChan := make(chan json.RawMessage, 1)

	// Add to pending requests
	b.mu.Lock()
	b.pendingRequests[id] = responseChan
	b.requestQueue = append(b.requestQueue, PendingRequest{
		ID:        id,
		Request:   requestBody,
		Timestamp: time.Now(),
	})
	b.mu.Unlock()

	// Wait for response with timeout
	select {
	case response := <-responseChan:
		// Got a response!
		return response, nil
	case <-time.After(b.timeout):
		// Timeout - clean up both map and queue
		b.mu.Lock()
		delete(b.pendingRequests, id)
		b.removePendingRequestFromQueue(id)
		b.mu.Unlock()
		return nil, fmt.Errorf("request timeout after %s", b.timeout)
	case <-ctx.Done():
		// Client disconnected - clean up both map and queue
		b.mu.Lock()
		delete(b.pendingRequests, id)
		b.removePendingRequestFromQueue(id)
		b.mu.Unlock()
		return nil, ctx.Err()
	}
}

// removePendingRequestFromQueue removes a request from the queue by ID
// Must be called with b.mu held
func (b *Broker) removePendingRequestFromQueue(id string) {
	// Filter out the request with matching ID
	filtered := make([]PendingRequest, 0, len(b.requestQueue))
	for _, req := range b.requestQueue {
		if req.ID != id {
			filtered = append(filtered, req)
		}
	}
	b.requestQueue = filtered
}

// PollRequests returns all pending requests and clears the queue
// This is called by the /poll endpoint
func (b *Broker) PollRequests() []PendingRequest {
	b.mu.Lock()
	defer b.mu.Unlock()

	// Return a copy of the queue
	requests := make([]PendingRequest, len(b.requestQueue))
	copy(requests, b.requestQueue)

	// Clear the queue
	b.requestQueue = make([]PendingRequest, 0)

	return requests
}

// RespondToRequest sends a response back to a waiting request
// This is called by the /respond endpoint
func (b *Broker) RespondToRequest(id string, response json.RawMessage) error {
	b.mu.Lock()
	responseChan, exists := b.pendingRequests[id]
	if !exists {
		b.mu.Unlock()
		return fmt.Errorf("request ID %s not found (may have timed out)", id)
	}
	// Remove from pending requests
	delete(b.pendingRequests, id)
	b.mu.Unlock()

	// Send response (non-blocking since channel is buffered)
	responseChan <- response
	close(responseChan)

	return nil
}
