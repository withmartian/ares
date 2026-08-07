package main

import (
	"bytes"
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
		resp, err := client.Get(serverURL + "/poll")
		if err != nil {
			t.Fatalf("poll failed: %v", err)
		}

		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			t.Fatalf("failed to read poll response: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("poll status = %d, body = %s", resp.StatusCode, body)
		}

		var pending []PendingRequest
		if err := json.Unmarshal(body, &pending); err != nil {
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
	resp, err := client.Post(serverURL+"/respond", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("respond failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("failed to read respond response: %v", err)
		}
		t.Fatalf("respond status = %d, body = %s", resp.StatusCode, body)
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
			contentType: "text/event-stream",
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
				resp, err := server.Client().Post(
					server.URL+tc.endpoint,
					"application/json",
					bytes.NewBufferString(tc.request),
				)
				if err != nil {
					errorChan <- err
					return
				}
				defer resp.Body.Close()

				body, err := io.ReadAll(resp.Body)
				if err != nil {
					errorChan <- err
					return
				}
				if resp.StatusCode != http.StatusOK {
					errorChan <- fmt.Errorf("status = %d, body = %s", resp.StatusCode, body)
					return
				}
				responseChan <- receivedResponse{body: string(body), contentType: resp.Header.Get("Content-Type")}
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
			resp, err := server.Client().Get(server.URL + endpoint)
			if err != nil {
				t.Fatalf("request failed: %v", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusMethodNotAllowed {
				t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusMethodNotAllowed)
			}
		})
	}
}

func TestLLMEndpoints_RejectInvalidJSON(t *testing.T) {
	broker := NewBroker(1 * time.Minute)
	server := newTestServer(broker)
	defer server.Close()

	resp, err := server.Client().Post(
		server.URL+"/v1/responses",
		"application/json",
		bytes.NewBufferString("not-json"),
	)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}
	if pending := broker.PollRequests(); len(pending) != 0 {
		t.Fatalf("invalid request was queued: %v", pending)
	}
}
