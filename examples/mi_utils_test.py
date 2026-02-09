"""Unit tests for MI utilities."""

# ruff: noqa: N806
# Allow uppercase variable names (X, y) for ML matrices

from pathlib import Path
import tempfile

import numpy as np

from . import mi_utils


def test_linear_probe_training():
    """Test that linear probe can be trained on synthetic data."""
    # Generate simple separable data
    n_samples = 100
    hidden_dim = 10

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples).astype(np.int32)

    # Add signal to make it learnable
    for i in range(n_samples):
        if y[i] == 1:
            X[i, 0] += 1.0

    # Train probe
    probe = mi_utils.LinearProbe(hidden_dim=hidden_dim)
    losses = probe.train(X, y, learning_rate=0.1, n_epochs=50, l2_reg=0.01)

    # Check that training reduces loss
    assert len(losses) == 50
    assert losses[-1] < losses[0], "Loss should decrease during training"

    # Check that probe can predict reasonably well
    predictions = probe.predict(X)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.6, f"Accuracy should be > 0.6, got {accuracy}"


def test_linear_probe_evaluation():
    """Test probe evaluation metrics."""
    # Create perfect predictions
    hidden_dim = 10
    probe = mi_utils.LinearProbe(hidden_dim=hidden_dim)

    # Set weights that perfectly separate the data
    X = np.array([[1.0] + [0.0] * 9, [0.0] * 10], dtype=np.float32)
    y = np.array([1, 0], dtype=np.int32)

    probe.weights[0] = 10.0  # High weight on first dimension

    metrics = probe.evaluate(X, y)

    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0


def test_linear_probe_save_load():
    """Test saving and loading probe parameters."""
    hidden_dim = 10
    probe = mi_utils.LinearProbe(hidden_dim=hidden_dim)
    probe.weights = np.arange(hidden_dim, dtype=np.float32)
    probe.bias = 5.0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "probe.json"

        # Save
        probe.save(path)
        assert path.exists()

        # Load into new probe
        probe2 = mi_utils.LinearProbe(hidden_dim=hidden_dim)
        probe2.load(path)

        # Check parameters match
        np.testing.assert_array_equal(probe.weights, probe2.weights)
        assert probe.bias == probe2.bias


def test_activation_recording_client():
    """Test activation recording wrapper."""
    # Create a mock client that we won't actually call
    recording_client = mi_utils.ActivationRecordingLLMClient(
        client=None,  # type: ignore
        layer_index=-1,
    )

    # Manually add some records
    activations1 = np.random.randn(512).astype(np.float32)
    activations2 = np.random.randn(512).astype(np.float32)

    recording_client.add_record(activations1, label=1, metadata={"step": 0})
    recording_client.add_record(activations2, label=0, metadata={"step": 1})

    assert len(recording_client.records) == 2
    assert recording_client.records[0].label == 1
    assert recording_client.records[1].label == 0
    assert recording_client.records[0].metadata["step"] == 0


def test_activation_recording_save_load():
    """Test saving and loading activation records."""
    recording_client = mi_utils.ActivationRecordingLLMClient(
        client=None,  # type: ignore
        layer_index=-1,
    )

    # Add some records
    for i in range(5):
        activations = np.random.randn(10).astype(np.float32)
        recording_client.add_record(activations, label=i % 2, metadata={"id": i})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "records.pkl"

        # Save
        recording_client.save_records(path)
        assert path.exists()

        # Load into new client
        recording_client2 = mi_utils.ActivationRecordingLLMClient(
            client=None,  # type: ignore
            layer_index=-1,
        )
        recording_client2.load_records(path)

        # Check records match
        assert len(recording_client2.records) == 5
        for i in range(5):
            assert recording_client2.records[i].label == i % 2
            assert recording_client2.records[i].metadata["id"] == i


def test_extract_confidence_from_response():
    """Test confidence extraction from text."""
    # Test valid JSON
    response = 'Let me solve this. {"confidence": 0.8} I think this will work.'
    confidence = mi_utils.extract_confidence_from_response(response)
    assert confidence == 0.8

    # Test no confidence field
    response = "This is a plain response without confidence"
    confidence = mi_utils.extract_confidence_from_response(response)
    assert confidence is None

    # Test invalid JSON
    response = "This has {invalid json} in it"
    confidence = mi_utils.extract_confidence_from_response(response)
    assert confidence is None


def test_sigmoid():
    """Test sigmoid function."""
    probe = mi_utils.LinearProbe(hidden_dim=10)

    # Test known values
    assert abs(probe._sigmoid(np.array([0.0])) - 0.5) < 1e-6
    assert probe._sigmoid(np.array([100.0])) > 0.99
    assert probe._sigmoid(np.array([-100.0])) < 0.01


def test_auc_computation():
    """Test AUC computation."""
    probe = mi_utils.LinearProbe(hidden_dim=10)

    # Perfect predictions
    y_true = np.array([0, 0, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    auc = probe._compute_auc(y_true, y_score)
    assert auc == 1.0

    # Random predictions (all same score)
    # With all identical scores, the order is arbitrary, so AUC is not well-defined
    # We just check it's between 0 and 1
    y_true = np.array([0, 1, 0, 1], dtype=np.int32)
    y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    auc = probe._compute_auc(y_true, y_score)
    assert 0.0 <= auc <= 1.0  # Should be valid AUC

    # All same class (undefined AUC)
    y_true = np.array([1, 1, 1, 1], dtype=np.int32)
    y_score = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    auc = probe._compute_auc(y_true, y_score)
    assert auc == 0.5  # Should return 0.5 for undefined case


def test_sklearn_probe_training():
    """Test sklearn-based probe training."""
    try:
        from sklearn import linear_model  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not available")

    # Generate simple separable data
    n_samples = 100
    hidden_dim = 10

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples).astype(np.int32)

    # Add signal to make it learnable
    for i in range(n_samples):
        if y[i] == 1:
            X[i, 0] += 1.5

    # Train probe
    probe = mi_utils.SklearnLogisticProbe(hidden_dim=hidden_dim)
    probe.train(X, y, max_iter=100)

    # Check that probe can predict reasonably well
    predictions = probe.predict(X)
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.7, f"Accuracy should be > 0.7, got {accuracy}"


def test_sklearn_probe_evaluation():
    """Test sklearn probe evaluation metrics."""
    try:
        from sklearn import linear_model  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not available")

    # Create data
    hidden_dim = 10
    n_samples = 50

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples).astype(np.int32)

    # Add strong signal
    for i in range(n_samples):
        if y[i] == 1:
            X[i, 0] += 2.0

    probe = mi_utils.SklearnLogisticProbe(hidden_dim=hidden_dim)
    probe.train(X, y, max_iter=100)

    metrics = probe.evaluate(X, y)

    assert "accuracy" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["accuracy"] > 0.7  # Should be decent on training data


def test_sklearn_probe_save_load():
    """Test saving and loading sklearn probe."""
    try:
        from sklearn import linear_model  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not available")

    hidden_dim = 10
    n_samples = 50

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, hidden_dim).astype(np.float32)
    y = rng.randint(0, 2, size=n_samples).astype(np.int32)

    # Train probe
    probe = mi_utils.SklearnLogisticProbe(hidden_dim=hidden_dim)
    probe.train(X, y, max_iter=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sklearn_probe.pkl"

        # Save
        probe.save(path)
        assert path.exists()

        # Load into new probe
        probe2 = mi_utils.SklearnLogisticProbe(hidden_dim=hidden_dim)
        probe2.load(path)

        # Check predictions match
        pred1 = probe.predict_proba(X)
        pred2 = probe2.predict_proba(X)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=5)


def test_multi_turn_trace_generation():
    """Test the multi-turn trace generation from example 07."""
    # Import the function from the example
    from examples import mi_swebench_correctness_probe as example_07

    # Generate a small trace
    X, y, confidence = example_07.generate_synthetic_multi_turn_trace(n_turns=5, hidden_dim=32, seed=42)

    # Check shapes
    assert X.shape == (5, 32)
    assert y.shape == (5,)
    assert confidence.shape == (5,)

    # Check types
    assert X.dtype == np.float32
    assert y.dtype == np.int32
    assert confidence.dtype == np.float32

    # Check value ranges
    assert np.all((y == 0) | (y == 1))
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_end_to_end_probe_pipeline():
    """Test end-to-end probe pipeline with synthetic multi-turn data."""
    try:
        from sklearn import linear_model  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not available")

    from examples import mi_swebench_correctness_probe as example_07

    # Generate multiple traces
    all_X = []
    all_y = []
    all_conf = []

    for episode in range(10):
        X, y, conf = example_07.generate_synthetic_multi_turn_trace(n_turns=8, hidden_dim=64, seed=episode)
        all_X.append(X)
        all_y.append(y)
        all_conf.append(conf)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    # Split train/test
    n_train = int(len(y_all) * 0.8)
    X_train, X_test = X_all[:n_train], X_all[n_train:]
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    # Train probe
    probe = mi_utils.SklearnLogisticProbe(hidden_dim=64)
    probe.train(X_train, y_train, max_iter=100)

    # Evaluate
    metrics = probe.evaluate(X_test, y_test)

    # Should achieve reasonable performance on synthetic data
    assert metrics["accuracy"] > 0.5
    assert metrics["auc"] > 0.5

    # Get predictions
    proba = probe.predict_proba(X_test)
    assert len(proba) == len(y_test)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
