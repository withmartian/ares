//! Data structures shared across endpoints (port of `types.go`).

use chrono::DateTime;
use chrono::Utc;
use serde::Deserialize;
use serde::Serialize;
use serde_json::value::RawValue;

/// A request waiting for a response. This is what gets returned from the `/poll` endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct PendingRequest {
    pub id: String,
    /// The raw JSON blob from the client, passed through untouched.
    pub request: Box<RawValue>,
    pub timestamp: DateTime<Utc>,
}

/// The body sent to the `/respond` endpoint.
#[derive(Debug, Deserialize)]
pub struct RespondRequest {
    pub id: String,
    /// The raw JSON response to send back, passed through untouched.
    pub response: Box<RawValue>,
}
