package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type receivedResponse struct {
	body        string
	contentType string
	statusCode  int
}

func doRequest(client *http.Client, method string, url string, body io.Reader) (receivedResponse, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return receivedResponse{}, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return receivedResponse{}, err
	}
	responseBody, readErr := io.ReadAll(resp.Body)
	closeErr := resp.Body.Close()
	if readErr != nil {
		return receivedResponse{}, readErr
	}
	if closeErr != nil {
		return receivedResponse{}, closeErr
	}
	return receivedResponse{
		body:        string(responseBody),
		contentType: resp.Header.Get("Content-Type"),
		statusCode:  resp.StatusCode,
	}, nil
}

func newTestServer(broker *Broker) *httptest.Server {
	mux := http.NewServeMux()
	registerRoutes(mux, broker)
	return httptest.NewServer(mux)
}

func pollUntilRequest(t *testing.T, client *http.Client, serverURL string) PendingRequest {
	t.Helper()

	deadline := time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := doRequest(client, http.MethodGet, serverURL+"/poll", nil)
		if err != nil {
			t.Fatalf("poll failed: %v", err)
		}
		if resp.statusCode != http.StatusOK {
			t.Fatalf("poll status = %d, body = %s", resp.statusCode, resp.body)
		}

		var pending []PendingRequest
		if err := json.Unmarshal([]byte(resp.body), &pending); err != nil {
			t.Fatalf("failed to decode poll response: %v", err)
		}
		if len(pending) > 0 {
			if len(pending) != 1 {
				t.Fatalf("expected 1 pending request, got %d", len(pending))
			}
			return pending[0]
		}

		time.Sleep(10 * time.Millisecond)
	}

	t.Fatal("timed out waiting for pending request")
	return PendingRequest{}
}

func respondToRequest(
	t *testing.T,
	client *http.Client,
	serverURL string,
	id string,
	response string,
	contentType string,
) {
	t.Helper()

	var responseValue any = json.RawMessage(response)
	if contentType != "application/json" {
		responseValue = response
	}
	body, err := json.Marshal(map[string]any{
		"id":           id,
		"response":     responseValue,
		"content_type": contentType,
	})
	if err != nil {
		t.Fatalf("failed to encode respond request: %v", err)
	}
	resp, err := doRequest(client, http.MethodPost, serverURL+"/respond", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("respond failed: %v", err)
	}
	if resp.statusCode != http.StatusOK {
		t.Fatalf("respond status = %d, body = %s", resp.statusCode, resp.body)
	}
}

func TestLLMEndpoints_RouteRawRequests(t *testing.T) {
	testCases := []struct {
		name        string
		endpoint    string
		request     string
		response    string
		contentType string
	}{
		{
			name:        "chat completions",
			endpoint:    "/v1/chat/completions",
			request:     `{"model":"gpt-4","messages":[{"role":"user","content":"hello"}]}`,
			response:    `{"id":"chatcmpl-test","choices":[{"message":{"role":"assistant","content":"hi"}}]}`,
			contentType: "application/json",
		},
		{
			name:        "openai responses",
			endpoint:    "/v1/responses",
			request:     `{"model":"gpt-5","input":[{"role":"user","content":[{"type":"input_text","text":"hello"}]}],"stream":true}`,
			response:    "event: response.completed\ndata: {\"type\":\"response.completed\"}\n\n",
			contentType: "text/event-stream; charset=utf-8",
		},
		{
			name:        "anthropic messages",
			endpoint:    "/v1/messages",
			request:     `{"model":"claude-sonnet","messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}],"stream":true,"max_tokens":4096}`,
			response:    `{"id":"msg-test","type":"message","role":"assistant","content":[{"type":"text","text":"hi"}]}`,
			contentType: "application/json",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			broker := NewBroker(1 * time.Minute)
			server := newTestServer(broker)
			defer server.Close()

			responseChan := make(chan receivedResponse, 1)
			errorChan := make(chan error, 1)
			go func() {
				resp, err := doRequest(
					server.Client(),
					http.MethodPost,
					server.URL+tc.endpoint,
					bytes.NewBufferString(tc.request),
				)
				if err != nil {
					errorChan <- err
					return
				}
				if resp.statusCode != http.StatusOK {
					errorChan <- fmt.Errorf("status = %d, body = %s", resp.statusCode, resp.body)
					return
				}
				responseChan <- resp
			}()

			pending := pollUntilRequest(t, server.Client(), server.URL)
			if pending.Endpoint != tc.endpoint {
				t.Fatalf("endpoint = %q, want %q", pending.Endpoint, tc.endpoint)
			}
			if string(pending.Request) != tc.request {
				t.Fatalf("request = %s, want %s", pending.Request, tc.request)
			}

			respondToRequest(t, server.Client(), server.URL, pending.ID, tc.response, tc.contentType)

			select {
			case err := <-errorChan:
				t.Fatal(err)
			case response := <-responseChan:
				if response.body != tc.response {
					t.Fatalf("response = %s, want %s", response.body, tc.response)
				}
				if response.contentType != tc.contentType {
					t.Fatalf("content type = %q, want %q", response.contentType, tc.contentType)
				}
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for LLM endpoint response")
			}
		})
	}
}

func TestLLMEndpoints_RejectNonPost(t *testing.T) {
	broker := NewBroker(1 * time.Minute)
	server := newTestServer(broker)
	defer server.Close()

	for _, endpoint := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages"} {
		t.Run(endpoint, func(t *testing.T) {
			resp, err := doRequest(server.Client(), http.MethodGet, server.URL+endpoint, nil)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			if resp.statusCode != http.StatusMethodNotAllowed {
				t.Fatalf("status = %d, want %d", resp.statusCode, http.StatusMethodNotAllowed)
			}
		})
	}
}

func TestLLMEndpoints_RejectInvalidJSON(t *testing.T) {
	broker := NewBroker(1 * time.Minute)
	server := newTestServer(broker)
	defer server.Close()

	resp, err := doRequest(
		server.Client(),
		http.MethodPost,
		server.URL+"/v1/responses",
		bytes.NewBufferString("not-json"),
	)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	if resp.statusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", resp.statusCode, http.StatusBadRequest)
	}
	if pending := broker.PollRequests(); len(pending) != 0 {
		t.Fatalf("invalid request was queued: %v", pending)
	}
}

func TestRespondRejectsMissingResponse(t *testing.T) {
	broker := NewBroker(1 * time.Minute)
	server := newTestServer(broker)
	defer server.Close()

	resp, err := doRequest(
		server.Client(),
		http.MethodPost,
		server.URL+"/respond",
		bytes.NewBufferString(`{"id":"request-id"}`),
	)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	if resp.statusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", resp.statusCode, http.StatusBadRequest)
	}
}
