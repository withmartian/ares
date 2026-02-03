.PHONY: build test run clean

# Build the ares-proxy binary
build:
	go build -o ares-proxy .

# Run tests
test:
	go test -v ./...

# Run the server
run:
	go run .

# Clean build artifacts
clean:
	rm -f ares-proxy
