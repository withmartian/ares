package main

import (
	"os"
	"strconv"
	"time"
)

// Config holds the server configuration
type Config struct {
	Port    string
	Timeout time.Duration
}

// LoadConfig loads configuration from environment variables with defaults
func LoadConfig() Config {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Timeout in minutes, default 15
	timeoutMinutes := 15
	if timeoutStr := os.Getenv("TIMEOUT_MINUTES"); timeoutStr != "" {
		if parsed, err := strconv.Atoi(timeoutStr); err == nil {
			timeoutMinutes = parsed
		}
	}

	return Config{
		Port:    port,
		Timeout: time.Duration(timeoutMinutes) * time.Minute,
	}
}
