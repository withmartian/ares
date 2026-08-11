package main

import (
	"encoding/json"
	"fmt"
	"mime"
	"time"
)

// PendingRequest represents a request waiting for a response
// This is what gets returned from the /poll endpoint
type PendingRequest struct {
	ID        string          `json:"id"`
	Endpoint  string          `json:"endpoint"`
	Request   json.RawMessage `json:"request"` // The raw JSON blob from the client
	Timestamp time.Time       `json:"timestamp"`
}

// RespondRequest is sent to the /respond endpoint
type RespondRequest struct {
	ID          string          `json:"id"`
	Response    json.RawMessage `json:"response"`
	ContentType string          `json:"content_type,omitempty"`
}

// ProxyResponse is returned to the intercepted LLM client.
type ProxyResponse struct {
	Body        []byte
	ContentType string
}

// ProxyResponse converts a RespondRequest to a ProxyResponse for the intercepted LLM client.
func (r RespondRequest) ProxyResponse() (ProxyResponse, error) {
	if len(r.Response) == 0 {
		return ProxyResponse{}, fmt.Errorf("response is required")
	}

	contentType := r.ContentType
	if contentType == "" {
		contentType = "application/json"
	}
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return ProxyResponse{}, fmt.Errorf("invalid content type %q: %w", contentType, err)
	}
	contentType = mime.FormatMediaType(mediaType, params)

	if mediaType == "application/json" {
		return ProxyResponse{Body: r.Response, ContentType: contentType}, nil
	}
	if mediaType != "text/event-stream" {
		return ProxyResponse{}, fmt.Errorf("unsupported content type %q", contentType)
	}

	// Non-JSON bodies travel inside /respond's JSON envelope as strings, then become raw HTTP bytes here.
	var body string
	if err = json.Unmarshal(r.Response, &body); err != nil {
		return ProxyResponse{}, fmt.Errorf("response must be a JSON string for content type %q: %w", contentType, err)
	}

	return ProxyResponse{Body: []byte(body), ContentType: contentType}, nil
}
