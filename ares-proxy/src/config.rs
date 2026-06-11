//! Configuration loading from environment variables (port of `config.go`).

use std::time::Duration;

const DEFAULT_PORT: u16 = 8080;
const DEFAULT_TIMEOUT_MINUTES: u64 = 15;

/// Server configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub port: u16,
    pub timeout: Duration,
}

impl Config {
    /// Loads configuration from environment variables, falling back to defaults
    /// (`PORT=8080`, `TIMEOUT_MINUTES=15`). Unparseable values fall back to the
    /// defaults, matching the Go implementation's lenient behavior.
    pub fn load() -> Self {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(DEFAULT_PORT);

        let timeout_minutes = std::env::var("TIMEOUT_MINUTES")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_MINUTES);

        Self {
            port,
            timeout: Duration::from_secs(timeout_minutes * 60),
        }
    }
}
