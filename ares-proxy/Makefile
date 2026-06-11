.PHONY: build test run clean

# Build the ares-proxy binary
build:
	cargo build --release
	cp target/release/ares-proxy ares-proxy

# Run tests
test:
	cargo test

# Run the server
run:
	cargo run

# Clean build artifacts
clean:
	cargo clean
	rm -f ares-proxy
