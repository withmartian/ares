package main

import (
	"encoding/json"
	"time"
)

// PendingRequest represents a request waiting for a response
// This is what gets returned from the /poll endpoint
type PendingRequest struct {
	ID        string          `json:"id"`
	Request   json.RawMessage `json:"request"` // The raw JSON blob from the client
	Timestamp time.Time       `json:"timestamp"`
}

// RespondRequest is sent to the /respond endpoint
type RespondRequest struct {
	ID       string          `json:"id"`
	Response json.RawMessage `json:"response"` // The raw JSON response to send back
}
