//! Core request/response coordination logic (port of `broker.go`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;

use chrono::Utc;
use serde_json::value::RawValue;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::types::PendingRequest;

/// Error returned by [`Broker::submit_request`].
#[derive(Debug, PartialEq, Eq)]
pub enum SubmitError {
    /// No response arrived within the broker timeout.
    Timeout(Duration),
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Duration's Debug formatting ("100ms", "900s") is close to Go's
            // duration formatting and good enough for log/error messages.
            SubmitError::Timeout(timeout) => write!(f, "request timeout after {timeout:?}"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Error returned by [`Broker::respond_to_request`].
#[derive(Debug, PartialEq, Eq)]
pub enum RespondError {
    /// The request ID is unknown (never existed, already answered, timed out, or cancelled).
    NotFound(String),
}

impl std::fmt::Display for RespondError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RespondError::NotFound(id) => {
                write!(f, "request ID {id} not found (may have timed out)")
            }
        }
    }
}

impl std::error::Error for RespondError {}

#[derive(Default)]
struct State {
    /// Maps request ID to the channel that delivers its response.
    pending_requests: HashMap<String, oneshot::Sender<Box<RawValue>>>,
    /// Requests waiting to be polled.
    request_queue: Vec<PendingRequest>,
}

/// Manages pending requests and coordinates responses.
pub struct Broker {
    /// Protects both the pending map and the queue from concurrent access.
    state: Mutex<State>,
    /// How long to wait for a response before giving up.
    timeout: Duration,
}

impl Broker {
    /// Creates a new broker with the given response timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            state: Mutex::new(State::default()),
            timeout,
        }
    }

    /// Adds a request to the queue and waits for a response.
    ///
    /// Called by the `/v1/chat/completions` endpoint. If the returned future is
    /// dropped before completion (client disconnect), the request is cleaned up,
    /// mirroring the Go implementation's `ctx.Done()` path.
    pub async fn submit_request(
        &self,
        request: Box<RawValue>,
    ) -> Result<Box<RawValue>, SubmitError> {
        let id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();

        {
            let mut state = self.state.lock().expect("broker mutex poisoned");
            state.pending_requests.insert(id.clone(), tx);
            state.request_queue.push(PendingRequest {
                id: id.clone(),
                request,
                timestamp: Utc::now(),
            });
        }

        // The guard removes this request from both the pending map and the queue
        // unless disarmed. It fires on timeout AND on future drop (cancellation),
        // so stale requests never reappear in later polls.
        let mut guard = CleanupGuard {
            broker: self,
            id: &id,
            armed: true,
        };

        match tokio::time::timeout(self.timeout, rx).await {
            Ok(Ok(response)) => {
                // Got a response. respond_to_request already removed the pending
                // entry; the queue entry was consumed by an earlier poll.
                guard.armed = false;
                Ok(response)
            }
            // The sender was dropped without sending. This cannot happen through the
            // public API (respond_to_request always sends after removing the sender),
            // so treat it like a timeout rather than panicking.
            Ok(Err(_)) | Err(_) => Err(SubmitError::Timeout(self.timeout)),
        }
    }

    /// Returns all pending requests and clears the queue. Called by the `/poll` endpoint.
    pub fn poll_requests(&self) -> Vec<PendingRequest> {
        let mut state = self.state.lock().expect("broker mutex poisoned");
        std::mem::take(&mut state.request_queue)
    }

    /// Sends a response back to a waiting request. Called by the `/respond` endpoint.
    pub fn respond_to_request(
        &self,
        id: &str,
        response: Box<RawValue>,
    ) -> Result<(), RespondError> {
        let tx = {
            let mut state = self.state.lock().expect("broker mutex poisoned");
            state
                .pending_requests
                .remove(id)
                .ok_or_else(|| RespondError::NotFound(id.to_string()))?
        };

        // If the waiter vanished between map removal and send, the response is
        // silently discarded - same as Go's buffered-channel behavior.
        let _ = tx.send(response);
        Ok(())
    }
}

/// Removes a request from broker state on drop unless disarmed.
struct CleanupGuard<'a> {
    broker: &'a Broker,
    id: &'a str,
    armed: bool,
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut state = self.broker.state.lock().expect("broker mutex poisoned");
        state.pending_requests.remove(self.id);
        state.request_queue.retain(|req| req.id != self.id);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn raw(json: &str) -> Box<RawValue> {
        serde_json::value::RawValue::from_string(json.to_string()).expect("invalid test JSON")
    }

    #[tokio::test]
    async fn submit_and_poll() {
        let broker = Arc::new(Broker::new(Duration::from_secs(60)));
        let test_request =
            r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}"#;

        let submit = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(test_request)).await }
        });

        // Give it time to queue.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let pending = broker.poll_requests();
        assert_eq!(pending.len(), 1, "expected 1 pending request");
        assert_eq!(pending[0].request.get(), test_request);

        let test_response =
            r#"{"id": "test", "choices": [{"message": {"role": "assistant", "content": "Hi"}}]}"#;
        broker
            .respond_to_request(&pending[0].id, raw(test_response))
            .expect("respond_to_request failed");

        let response = tokio::time::timeout(Duration::from_secs(1), submit)
            .await
            .expect("timeout waiting for response")
            .expect("submit task panicked")
            .expect("submit_request failed");
        assert_eq!(response.get(), test_response);
    }

    #[tokio::test]
    async fn multiple_concurrent_requests() {
        let broker = Arc::new(Broker::new(Duration::from_secs(60)));
        let num_requests = 5;

        let handles: Vec<_> = (0..num_requests)
            .map(|idx| {
                let broker = Arc::clone(&broker);
                tokio::spawn(async move {
                    let req = format!(r#"{{"request": "{idx}"}}"#);
                    broker.submit_request(raw(&req)).await
                })
            })
            .collect();

        // Give them time to queue.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let pending = broker.poll_requests();
        assert_eq!(pending.len(), num_requests, "expected all requests pending");

        for req in &pending {
            broker
                .respond_to_request(&req.id, raw(r#"{"response": "ok"}"#))
                .expect("respond failed");
        }

        for (i, handle) in handles.into_iter().enumerate() {
            tokio::time::timeout(Duration::from_secs(1), handle)
                .await
                .unwrap_or_else(|_| panic!("timeout waiting for response {i}"))
                .expect("submit task panicked")
                .unwrap_or_else(|e| panic!("submit {i} failed: {e}"));
        }
    }

    #[tokio::test]
    async fn poll_empty_queue() {
        let broker = Broker::new(Duration::from_secs(60));
        assert!(broker.poll_requests().is_empty(), "expected empty queue");
    }

    #[tokio::test]
    async fn respond_to_nonexistent_request() {
        let broker = Broker::new(Duration::from_secs(60));
        let err = broker
            .respond_to_request("nonexistent-id", raw(r#"{"response": "test"}"#))
            .expect_err("expected error when responding to nonexistent request");
        assert_eq!(
            err.to_string(),
            "request ID nonexistent-id not found (may have timed out)"
        );
    }

    #[tokio::test]
    async fn timeout() {
        // Use a very short timeout for testing.
        let broker = Broker::new(Duration::from_millis(100));

        // Submit will time out because no one responds.
        let err = broker
            .submit_request(raw(r#"{"test": "request"}"#))
            .await
            .expect_err("expected timeout error");
        assert_eq!(err.to_string(), "request timeout after 100ms");

        // The timed-out request must not be in the queue.
        assert!(
            broker.poll_requests().is_empty(),
            "expected empty queue after timeout"
        );
    }

    #[tokio::test]
    async fn cancellation() {
        // The Rust analog of Go context cancellation: the submit future is dropped
        // when the client disconnects. Aborting the task drops the future.
        let broker = Arc::new(Broker::new(Duration::from_secs(60)));

        let submit = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"test": "request"}"#)).await }
        });

        // Give it time to register.
        tokio::time::sleep(Duration::from_millis(50)).await;

        submit.abort();
        let join_err = submit.await.expect_err("expected task to be cancelled");
        assert!(join_err.is_cancelled());

        // The cancelled request must not be in the queue.
        assert!(
            broker.poll_requests().is_empty(),
            "expected empty queue after cancellation"
        );
    }

    #[tokio::test]
    async fn poll_clears_queue() {
        let broker = Arc::new(Broker::new(Duration::from_secs(60)));

        let _submit = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"test": "request"}"#)).await }
        });

        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(
            broker.poll_requests().len(),
            1,
            "expected 1 pending request"
        );
        // Second poll should be empty (queue was cleared).
        assert!(
            broker.poll_requests().is_empty(),
            "expected empty queue after poll"
        );
    }

    #[tokio::test]
    async fn request_timestamp() {
        let broker = Arc::new(Broker::new(Duration::from_secs(60)));
        let before = Utc::now();

        let _submit = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"test": "request"}"#)).await }
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        let after = Utc::now();

        let pending = broker.poll_requests();
        assert_eq!(pending.len(), 1, "expected 1 pending request");

        let ts = pending[0].timestamp;
        assert!(
            ts >= before && ts <= after,
            "timestamp {ts} not between {before} and {after}"
        );
    }

    #[tokio::test]
    async fn exactly_once_delivery() {
        let broker = Arc::new(Broker::new(Duration::from_millis(200)));

        // Submit 3 requests: one answered, one left to time out, one cancelled.
        let answered = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"request": "1"}"#)).await }
        });
        let timed_out = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"request": "2"}"#)).await }
        });
        let cancelled = tokio::spawn({
            let broker = Arc::clone(&broker);
            async move { broker.submit_request(raw(r#"{"request": "3"}"#)).await }
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let pending = broker.poll_requests();
        assert_eq!(pending.len(), 3, "expected 3 pending requests");
        // Exactly-once: already polled.
        assert!(
            broker.poll_requests().is_empty(),
            "expected empty queue after first poll"
        );

        // Respond to the first polled request only.
        broker
            .respond_to_request(&pending[0].id, raw(r#"{"response": "ok"}"#))
            .expect("failed to respond to first request");

        cancelled.abort();
        let _ = cancelled.await;

        // Wait for the remaining request to time out.
        tokio::time::sleep(Duration::from_millis(250)).await;
        let _ = answered.await;
        let _ = timed_out.await;

        // Stale requests must not reappear.
        assert!(
            broker.poll_requests().is_empty(),
            "expected empty queue after timeout/cancel"
        );

        // Responding to timed-out or cancelled requests must fail.
        for stale in &pending[1..] {
            broker
                .respond_to_request(&stale.id, raw(r#"{"response": "ok"}"#))
                .expect_err("expected error when responding to stale request");
        }
    }
}
