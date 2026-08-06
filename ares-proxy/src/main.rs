//! HTTP server and endpoint handlers (port of `main.go`).

use std::sync::Arc;

use axum::extract::State;
use axum::http::header;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use axum::Json;
use axum::Router;
use serde_json::value::RawValue;

mod broker;
mod config;
mod types;

use broker::Broker;
use config::Config;
use types::PendingRequest;
use types::RespondRequest;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = Config::load();
    tracing::info!(
        "Starting server on port {} with timeout {:?}",
        config.port,
        config.timeout
    );

    let broker = Arc::new(Broker::new(config.timeout));

    let app = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .route("/poll", get(handle_poll))
        .route("/respond", post(handle_respond))
        .with_state(broker);

    let addr = format!("0.0.0.0:{}", config.port);
    tracing::info!("Server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|err| panic!("Failed to bind {addr}: {err}"));
    if let Err(err) = axum::serve(listener, app).await {
        panic!("Server failed: {err}");
    }
}

/// Handles `POST /v1/chat/completions`.
///
/// Queues the request and blocks until the controller responds or the timeout
/// elapses. If the client disconnects, axum drops this future and the broker's
/// cleanup guard removes the request.
async fn handle_chat_completion(State(broker): State<Arc<Broker>>, body: String) -> Response {
    let request: Box<RawValue> = match serde_json::from_str(&body) {
        Ok(raw) => raw,
        Err(err) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("Failed to read request: {err}"),
            )
                .into_response();
        }
    };

    match broker.submit_request(request).await {
        Ok(response) => (
            [(header::CONTENT_TYPE, "application/json")],
            response.get().to_owned(),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Request failed: {err}"),
        )
            .into_response(),
    }
}

/// Handles `GET /poll`: returns all pending requests and clears the queue.
async fn handle_poll(State(broker): State<Arc<Broker>>) -> Json<Vec<PendingRequest>> {
    Json(broker.poll_requests())
}

/// Handles `POST /respond`: routes a response back to a waiting request.
async fn handle_respond(
    State(broker): State<Arc<Broker>>,
    Json(req): Json<RespondRequest>,
) -> Response {
    match broker.respond_to_request(&req.id, req.response) {
        Ok(()) => Json(serde_json::json!({"status": "ok"})).into_response(),
        Err(err) => (StatusCode::NOT_FOUND, format!("Failed to respond: {err}")).into_response(),
    }
}
