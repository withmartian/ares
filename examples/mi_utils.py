"""Mechanistic interpretability utilities for activation recording and probe training.

This module provides minimal utilities for recording hidden activations from LLMs
and training linear probes for mechanistic interpretability experiments.
"""

import dataclasses
import json
import logging
from pathlib import Path
import pickle
from typing import Any

from ares.llms import llm_clients
from ares.llms import request
from ares.llms import response
import numpy as np
import numpy.typing as npt

try:
    from sklearn import linear_model
    from sklearn import metrics as sklearn_metrics

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class ActivationRecord:
    """A single record of activations and metadata from an LLM call.

    Attributes:
        activations: Hidden state activations, shape (seq_len, hidden_dim) or (hidden_dim,)
        label: Binary label (0 or 1) for probe training
        metadata: Additional information about this record
    """

    activations: npt.NDArray[np.float32]
    label: int
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class ActivationRecordingLLMClient(llm_clients.LLMClient):
    """LLM client wrapper that records activations for mechanistic interpretability.

    This wrapper intercepts LLM calls and extracts hidden activations from responses
    that include them. It delegates actual LLM calls to an underlying client.

    For this pedagogical example, we expect the LLM response to include a special
    metadata field with activations. In practice, you would modify the LLM client
    to extract activations directly from the model.

    Attributes:
        client: The underlying LLM client to delegate calls to
        records: List of activation records collected so far
        layer_index: Which transformer layer to extract activations from
    """

    def __init__(
        self,
        client: llm_clients.LLMClient,
        layer_index: int = -1,
    ):
        """Initialize the activation recording wrapper.

        Args:
            client: The underlying LLM client
            layer_index: Transformer layer to extract activations from (-1 for last layer)
        """
        self.client = client
        self.layer_index = layer_index
        self.records: list[ActivationRecord] = []

    async def __call__(self, req: request.LLMRequest) -> response.LLMResponse:
        """Execute LLM call and record activations if available.

        Args:
            req: The LLM request

        Returns:
            The LLM response from the underlying client
        """
        resp = await self.client(req)

        # In a real implementation, you would extract activations from the model here.
        # For this pedagogical example, we expect activations in metadata.
        # This is a stub - the example will generate synthetic activations.

        return resp

    def add_record(
        self,
        activations: npt.NDArray[np.float32],
        label: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Manually add an activation record.

        This is used in the pedagogical example where we generate synthetic activations.

        Args:
            activations: The activation array
            label: Binary label (0 or 1)
            metadata: Optional metadata dict
        """
        self.records.append(
            ActivationRecord(
                activations=activations,
                label=label,
                metadata=metadata or {},
            )
        )

    def save_records(self, path: Path) -> None:
        """Save activation records to disk.

        Args:
            path: Path to save the records (pickle format)
        """
        with open(path, "wb") as f:
            pickle.dump(self.records, f)
        _LOGGER.info("Saved %d activation records to %s", len(self.records), path)

    def load_records(self, path: Path) -> None:
        """Load activation records from disk.

        Args:
            path: Path to load the records from (pickle format)
        """
        with open(path, "rb") as f:
            self.records = pickle.load(f)
        _LOGGER.info("Loaded %d activation records from %s", len(self.records), path)

    def clear_records(self) -> None:
        """Clear all recorded activations."""
        self.records.clear()


class LinearProbe:
    """Simple logistic regression probe for binary classification.

    Uses numpy for a minimal implementation without heavy ML framework dependencies.

    Attributes:
        weights: Learned weight vector, shape (hidden_dim,)
        bias: Learned bias scalar
        hidden_dim: Dimensionality of input activations
    """

    def __init__(self, hidden_dim: int):
        """Initialize the linear probe.

        Args:
            hidden_dim: Dimensionality of input activations
        """
        self.hidden_dim = hidden_dim
        self.weights = np.zeros(hidden_dim, dtype=np.float32)
        self.bias = 0.0

    def _sigmoid(self, z: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Predict probabilities for input activations.

        Args:
            X: Input activations, shape (n_samples, hidden_dim)

        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        logits = np.dot(X, self.weights) + self.bias
        return self._sigmoid(logits)

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:  # noqa: N803
        """Predict binary labels for input activations.

        Args:
            X: Input activations, shape (n_samples, hidden_dim)

        Returns:
            Predicted labels (0 or 1), shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int32)

    def train(
        self,
        X: npt.NDArray[np.float32],  # noqa: N803
        y: npt.NDArray[np.int32],
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        l2_reg: float = 0.01,
    ) -> list[float]:
        """Train the probe using gradient descent.

        Args:
            X: Training activations, shape (n_samples, hidden_dim)
            y: Training labels (0 or 1), shape (n_samples,)
            learning_rate: Learning rate for gradient descent
            n_epochs: Number of training epochs
            l2_reg: L2 regularization strength

        Returns:
            List of loss values per epoch
        """
        n_samples = X.shape[0]
        losses = []

        for epoch in range(n_epochs):
            # Forward pass
            proba = self.predict_proba(X)

            # Compute loss (binary cross-entropy + L2 regularization)
            eps = 1e-7  # For numerical stability
            bce_loss = -np.mean(y * np.log(proba + eps) + (1 - y) * np.log(1 - proba + eps))
            l2_loss = 0.5 * l2_reg * np.sum(self.weights**2)
            loss = bce_loss + l2_loss
            losses.append(float(loss))

            # Backward pass (gradients)
            error = proba - y
            grad_weights = np.dot(X.T, error) / n_samples + l2_reg * self.weights
            grad_bias = np.mean(error)

            # Update parameters
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias

            if epoch % 20 == 0:
                _LOGGER.debug("Epoch %d/%d, Loss: %.4f", epoch, n_epochs, loss)

        return losses

    def evaluate(
        self,
        X: npt.NDArray[np.float32],  # noqa: N803
        y: npt.NDArray[np.int32],
    ) -> dict[str, float]:
        """Evaluate probe on test data.

        Args:
            X: Test activations, shape (n_samples, hidden_dim)
            y: True labels (0 or 1), shape (n_samples,)

        Returns:
            Dictionary with accuracy and AUC metrics
        """
        proba = self.predict_proba(X)
        pred = self.predict(X)

        # Accuracy
        accuracy = np.mean(pred == y)

        # AUC (using trapezoidal rule)
        auc = self._compute_auc(y, proba)

        return {
            "accuracy": float(accuracy),
            "auc": float(auc),
        }

    def _compute_auc(
        self,
        y_true: npt.NDArray[np.int32],
        y_score: npt.NDArray[np.float32],
    ) -> float:
        """Compute AUC using trapezoidal rule.

        Args:
            y_true: True binary labels
            y_score: Predicted probabilities

        Returns:
            AUC score
        """
        # Sort by predicted score
        sorted_indices = np.argsort(y_score)
        y_true_sorted = y_true[sorted_indices]

        # Compute TPR and FPR at each threshold
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:
            return 0.5  # Undefined, return random

        tpr = np.cumsum(y_true_sorted[::-1]) / n_pos
        fpr = np.cumsum(1 - y_true_sorted[::-1]) / n_neg

        # Compute AUC using trapezoidal rule
        auc = float(np.trapezoid(tpr, fpr))
        return auc

    def save(self, path: Path) -> None:
        """Save probe parameters to disk.

        Args:
            path: Path to save the probe (JSON format)
        """
        with open(path, "w") as f:
            json.dump(
                {
                    "weights": self.weights.tolist(),
                    "bias": float(self.bias),
                    "hidden_dim": self.hidden_dim,
                },
                f,
            )
        _LOGGER.info("Saved probe to %s", path)

    def load(self, path: Path) -> None:
        """Load probe parameters from disk.

        Args:
            path: Path to load the probe from (JSON format)
        """
        with open(path) as f:
            data = json.load(f)
        self.weights = np.array(data["weights"], dtype=np.float32)
        self.bias = float(data["bias"])
        self.hidden_dim = int(data["hidden_dim"])
        _LOGGER.info("Loaded probe from %s", path)


def extract_confidence_from_response(response_text: str) -> float | None:
    """Extract confidence score from agent response.

    This is a simple parser that looks for a JSON field like:
    {"confidence": 0.8}

    In practice, you would use a more robust parser or require a specific
    response format from the agent.

    Args:
        response_text: The agent's text response

    Returns:
        Confidence score between 0 and 1, or None if not found
    """
    try:
        # Look for JSON in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = response_text[start:end]
            data = json.loads(json_str)
            if "confidence" in data:
                return float(data["confidence"])
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return None


class SklearnLogisticProbe:
    """Logistic regression probe using sklearn for more robust training.

    This is a lightweight wrapper around sklearn's LogisticRegression
    that provides a consistent interface with the numpy-based LinearProbe.
    sklearn provides more stable optimization and is better for production use.

    Attributes:
        model: The sklearn LogisticRegression model
        hidden_dim: Dimensionality of input activations
        weights: Learned weight vector after training
    """

    def __init__(self, hidden_dim: int):
        """Initialize the sklearn-based probe.

        Args:
            hidden_dim: Dimensionality of input activations
        """
        if not _HAS_SKLEARN:
            raise ImportError("sklearn is required for SklearnLogisticProbe. Install with: pip install scikit-learn")

        self.hidden_dim = hidden_dim
        self.model = linear_model.LogisticRegression(
            max_iter=100,
            solver="lbfgs",
            random_state=42,
        )
        self._is_trained = False

    def train(
        self,
        X: npt.NDArray[np.float32],  # noqa: N803
        y: npt.NDArray[np.int32],
        max_iter: int = 100,
    ) -> None:
        """Train the probe using sklearn's LogisticRegression.

        Args:
            X: Training activations, shape (n_samples, hidden_dim)
            y: Training labels (0 or 1), shape (n_samples,)
            max_iter: Maximum iterations for optimization
        """
        self.model.max_iter = max_iter
        self.model.fit(X, y)
        self._is_trained = True
        _LOGGER.info("Trained sklearn probe: %d samples, converged=%s", len(y), self.model.n_iter_)

    def predict_proba(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Predict probabilities for input activations.

        Args:
            X: Input activations, shape (n_samples, hidden_dim)

        Returns:
            Predicted probabilities for class 1, shape (n_samples,)
        """
        if not self._is_trained:
            raise RuntimeError("Probe must be trained before prediction")
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int32]:  # noqa: N803
        """Predict binary labels for input activations.

        Args:
            X: Input activations, shape (n_samples, hidden_dim)

        Returns:
            Predicted labels (0 or 1), shape (n_samples,)
        """
        if not self._is_trained:
            raise RuntimeError("Probe must be trained before prediction")
        return self.model.predict(X).astype(np.int32)

    def evaluate(
        self,
        X: npt.NDArray[np.float32],  # noqa: N803
        y: npt.NDArray[np.int32],
    ) -> dict[str, float]:
        """Evaluate probe on test data.

        Args:
            X: Test activations, shape (n_samples, hidden_dim)
            y: True labels (0 or 1), shape (n_samples,)

        Returns:
            Dictionary with accuracy and AUC metrics
        """
        proba = self.predict_proba(X)
        pred = self.predict(X)

        accuracy = float(np.mean(pred == y))
        auc = float(sklearn_metrics.roc_auc_score(y, proba))

        return {
            "accuracy": accuracy,
            "auc": auc,
        }

    @property
    def weights(self) -> npt.NDArray[np.float32]:
        """Get the learned weight vector.

        Returns:
            Weight vector, shape (hidden_dim,)
        """
        if not self._is_trained:
            raise RuntimeError("Probe must be trained before accessing weights")
        return self.model.coef_[0].astype(np.float32)

    @property
    def bias(self) -> float:
        """Get the learned bias term.

        Returns:
            Bias scalar
        """
        if not self._is_trained:
            raise RuntimeError("Probe must be trained before accessing bias")
        return float(self.model.intercept_[0])

    def save(self, path: Path) -> None:
        """Save probe model to disk.

        Args:
            path: Path to save the probe (pickle format)
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        _LOGGER.info("Saved sklearn probe to %s", path)

    def load(self, path: Path) -> None:
        """Load probe model from disk.

        Args:
            path: Path to load the probe from (pickle format)
        """
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self._is_trained = True
        _LOGGER.info("Loaded sklearn probe from %s", path)
