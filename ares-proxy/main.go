package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

func main() {
	// Load configuration
	config := LoadConfig()
	log.Printf("Starting server on port %s with timeout %s", config.Port, config.Timeout)

	// Create the broker
	broker := NewBroker(config.Timeout)

	// Set up HTTP routes
	http.HandleFunc("/v1/chat/completions", handleChatCompletion(broker))
	http.HandleFunc("/poll", handlePoll(broker))
	http.HandleFunc("/respond", handleRespond(broker))

	// Start the server
	addr := ":" + config.Port
	log.Printf("Server listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// handleChatCompletion handles POST /v1/chat/completions
// This holds the request and waits for a response
func handleChatCompletion(broker *Broker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Read the request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to read request: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Submit the request and wait for response
		response, err := broker.SubmitRequest(r.Context(), json.RawMessage(body))
		if err != nil {
			http.Error(w, fmt.Sprintf("Request failed: %v", err), http.StatusInternalServerError)
			return
		}

		// Send the response back to the client
		w.Header().Set("Content-Type", "application/json")
		w.Write(response)
	}
}

// handlePoll handles GET /poll
// Returns all pending requests and clears the queue
func handlePoll(broker *Broker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Get all pending requests
		requests := broker.PollRequests()

		// Return as JSON
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(requests); err != nil {
			http.Error(w, fmt.Sprintf("Failed to encode response: %v", err), http.StatusInternalServerError)
			return
		}
	}
}

// handleRespond handles POST /respond
// Sends a response back to a waiting request
func handleRespond(broker *Broker) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Parse the request body
		var req RespondRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid JSON: %v", err), http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// Send the response to the waiting request
		if err := broker.RespondToRequest(req.ID, req.Response); err != nil {
			http.Error(w, fmt.Sprintf("Failed to respond: %v", err), http.StatusNotFound)
			return
		}

		// Acknowledge success
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}
}
