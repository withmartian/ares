package main

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func TestBroker_SubmitAndPoll(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	// Create a test request
	testRequest := json.RawMessage(`{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}`)

	// Submit request in background
	responseChan := make(chan json.RawMessage, 1)
	go func() {
		ctx := context.Background()
		response, err := broker.SubmitRequest(ctx, testRequest)
		if err != nil {
			t.Errorf("SubmitRequest failed: %v", err)
			return
		}
		responseChan <- response
	}()

	// Give it time to queue
	time.Sleep(100 * time.Millisecond)

	// Poll for pending requests
	pending := broker.PollRequests()
	if len(pending) != 1 {
		t.Fatalf("Expected 1 pending request, got %d", len(pending))
	}

	if string(pending[0].Request) != string(testRequest) {
		t.Errorf("Request mismatch: got %s, want %s", pending[0].Request, testRequest)
	}

	// Send response
	testResponse := json.RawMessage(`{"id": "test", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}`)
	err := broker.RespondToRequest(pending[0].ID, testResponse)
	if err != nil {
		t.Fatalf("RespondToRequest failed: %v", err)
	}

	// Verify the original SubmitRequest call received the response
	select {
	case response := <-responseChan:
		if string(response) != string(testResponse) {
			t.Errorf("Response mismatch: got %s, want %s", response, testResponse)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for response")
	}
}

func TestBroker_MultipleConcurrentRequests(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	numRequests := 5
	responses := make([]chan json.RawMessage, numRequests)

	// Submit multiple requests concurrently
	for i := 0; i < numRequests; i++ {
		responses[i] = make(chan json.RawMessage, 1)
		go func(idx int) {
			ctx := context.Background()
			req := json.RawMessage(`{"request": "` + string(rune('A'+idx)) + `"}`)
			response, err := broker.SubmitRequest(ctx, req)
			if err != nil {
				t.Errorf("Submit %d failed: %v", idx, err)
				return
			}
			responses[idx] <- response
		}(i)
	}

	// Give them time to queue
	time.Sleep(100 * time.Millisecond)

	// Poll should return all requests
	pending := broker.PollRequests()
	if len(pending) != numRequests {
		t.Fatalf("Expected %d pending requests, got %d", numRequests, len(pending))
	}

	// Respond to all
	testResponse := json.RawMessage(`{"response": "ok"}`)
	for _, req := range pending {
		err := broker.RespondToRequest(req.ID, testResponse)
		if err != nil {
			t.Errorf("Respond failed for %s: %v", req.ID, err)
		}
	}

	// Verify all received responses
	for i := 0; i < numRequests; i++ {
		select {
		case <-responses[i]:
			// Success
		case <-time.After(1 * time.Second):
			t.Errorf("Timeout waiting for response %d", i)
		}
	}
}

func TestBroker_PollEmptyQueue(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	// Poll with no pending requests
	pending := broker.PollRequests()
	if len(pending) != 0 {
		t.Errorf("Expected empty queue, got %d requests", len(pending))
	}
}

func TestBroker_RespondToNonexistentRequest(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	testResponse := json.RawMessage(`{"response": "test"}`)
	err := broker.RespondToRequest("nonexistent-id", testResponse)
	if err == nil {
		t.Error("Expected error when responding to nonexistent request")
	}
	expectedError := "request ID nonexistent-id not found (may have timed out)"
	if err.Error() != expectedError {
		t.Errorf("Expected error %q, got %q", expectedError, err.Error())
	}
}

func TestBroker_Timeout(t *testing.T) {
	// Use a very short timeout for testing
	broker := NewBroker(100 * time.Millisecond)

	testRequest := json.RawMessage(`{"test": "request"}`)
	ctx := context.Background()

	// Submit will timeout because no one responds
	_, err := broker.SubmitRequest(ctx, testRequest)
	if err == nil {
		t.Fatal("Expected timeout error, got nil")
	}
	expectedError := "request timeout after 100ms"
	if err.Error() != expectedError {
		t.Errorf("Expected error %q, got %q", expectedError, err.Error())
	}

	// Verify that timed-out request is not in the queue
	pending := broker.PollRequests()
	if len(pending) != 0 {
		t.Errorf("Expected empty queue after timeout, got %d requests", len(pending))
	}
}

func TestBroker_ContextCancellation(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	testRequest := json.RawMessage(`{"test": "request"}`)
	ctx, cancel := context.WithCancel(context.Background())

	// Submit in background
	errChan := make(chan error, 1)
	go func() {
		_, err := broker.SubmitRequest(ctx, testRequest)
		errChan <- err
	}()

	// Give it time to register
	time.Sleep(50 * time.Millisecond)

	// Cancel the context
	cancel()

	// Should get context cancelled error
	select {
	case err := <-errChan:
		if err == nil {
			t.Fatal("Expected context cancellation error, got nil")
		}
		if err != context.Canceled {
			t.Errorf("Expected context.Canceled error, got %v", err)
		}
	case <-time.After(1 * time.Second):
		t.Fatal("Timeout waiting for context cancellation")
	}

	// Verify that cancelled request is not in the queue
	pending := broker.PollRequests()
	if len(pending) != 0 {
		t.Errorf("Expected empty queue after cancellation, got %d requests", len(pending))
	}
}

func TestBroker_PollClearsQueue(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	// Submit a request
	go func() {
		ctx := context.Background()
		testRequest := json.RawMessage(`{"test": "request"}`)
		broker.SubmitRequest(ctx, testRequest)
	}()

	// Give it time to queue
	time.Sleep(100 * time.Millisecond)

	// First poll should return the request
	pending1 := broker.PollRequests()
	if len(pending1) != 1 {
		t.Fatalf("Expected 1 pending request, got %d", len(pending1))
	}

	// Second poll should be empty (queue was cleared)
	pending2 := broker.PollRequests()
	if len(pending2) != 0 {
		t.Errorf("Expected empty queue after poll, got %d requests", len(pending2))
	}
}

func TestBroker_RequestTimestamp(t *testing.T) {
	broker := NewBroker(1 * time.Minute)

	before := time.Now()

	// Submit a request
	go func() {
		ctx := context.Background()
		testRequest := json.RawMessage(`{"test": "request"}`)
		broker.SubmitRequest(ctx, testRequest)
	}()

	// Give it time to queue
	time.Sleep(100 * time.Millisecond)

	after := time.Now()

	pending := broker.PollRequests()
	if len(pending) != 1 {
		t.Fatalf("Expected 1 pending request, got %d", len(pending))
	}

	// Timestamp should be between before and after
	ts := pending[0].Timestamp
	if ts.Before(before) || ts.After(after) {
		t.Errorf("Timestamp %v not between %v and %v", ts, before, after)
	}
}

func TestBroker_ExactlyOnceDelivery(t *testing.T) {
	broker := NewBroker(200 * time.Millisecond)

	// Submit 3 requests: one normal, one that will timeout, one that will be cancelled
	ctx1 := context.Background()
	ctx2 := context.Background()
	ctx3, cancel3 := context.WithCancel(context.Background())

	testRequest1 := json.RawMessage(`{"request": "1"}`)
	testRequest2 := json.RawMessage(`{"request": "2"}`)
	testRequest3 := json.RawMessage(`{"request": "3"}`)

	// Submit all requests
	go func() {
		broker.SubmitRequest(ctx1, testRequest1)
	}()
	go func() {
		broker.SubmitRequest(ctx2, testRequest2) // Will timeout
	}()
	go func() {
		broker.SubmitRequest(ctx3, testRequest3)
	}()

	// Give them time to queue
	time.Sleep(50 * time.Millisecond)

	// First poll should return all 3 requests
	pending1 := broker.PollRequests()
	if len(pending1) != 3 {
		t.Fatalf("Expected 3 pending requests, got %d", len(pending1))
	}

	// Second poll should be empty (exactly-once: already polled)
	pending2 := broker.PollRequests()
	if len(pending2) != 0 {
		t.Errorf("Expected empty queue after first poll, got %d requests", len(pending2))
	}

	// Respond to request 1 only
	testResponse := json.RawMessage(`{"response": "ok"}`)
	err := broker.RespondToRequest(pending1[0].ID, testResponse)
	if err != nil {
		t.Errorf("Failed to respond to request 1: %v", err)
	}

	// Cancel request 3
	cancel3()

	// Wait for request 2 to timeout
	time.Sleep(200 * time.Millisecond)

	// Poll again - should still be empty (stale requests shouldn't reappear)
	pending3 := broker.PollRequests()
	if len(pending3) != 0 {
		t.Errorf("Expected empty queue after timeout/cancel, got %d requests", len(pending3))
	}

	// Verify we can't respond to timed-out or cancelled requests
	err = broker.RespondToRequest(pending1[1].ID, testResponse)
	if err == nil {
		t.Error("Expected error when responding to timed-out request")
	}

	err = broker.RespondToRequest(pending1[2].ID, testResponse)
	if err == nil {
		t.Error("Expected error when responding to cancelled request")
	}
}
