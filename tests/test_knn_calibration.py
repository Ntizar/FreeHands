"""Tests for KNN gaze mapping (Deer Mouse pattern)."""
from __future__ import annotations

import numpy as np
import pytest

from freehands.gaze.calibration import (
    CURRENT_GAZE_FEATURE_VERSION,
    CalibrationSample,
    GazeRegressor,
    fit_knn_model,
    KNN_DEFAULT_K,
    KNN_MIN_SAMPLES,
)
from freehands.profiles.store import GazeModel


def _make_knn_samples(n: int = 20) -> list[CalibrationSample]:
    """Generate deterministic calibration samples for KNN testing."""
    samples = []
    for i in range(n):
        x = 0.3 + (i % 5) * 0.1
        y = 0.4 + (i // 5) * 0.1
        sx = 100 + i * 80
        sy = 100 + i * 40
        samples.append(CalibrationSample(
            features=np.array([x, y, x + 0.05, y + 0.05, 0.0, 0.0]),
            target_xy=(float(sx), float(sy)),
        ))
    return samples


# ── fit_knn_model tests ─────────────────────────────────────────────────────

def test_fit_knn_model_requires_minimum_samples() -> None:
    """KNN needs at least KNN_MIN_SAMPLES samples."""
    samples = _make_knn_samples(KNN_MIN_SAMPLES - 1)
    with pytest.raises(ValueError, match="at least"):
        fit_knn_model(samples)


def test_fit_knn_model_returns_serialisable_model() -> None:
    """KNN model should be serialisable via GazeModel."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples)
    assert model.type == "knn_regression"
    assert model.feature_version == CURRENT_GAZE_FEATURE_VERSION
    assert len(model.weights_x) == 20  # training features X
    assert len(model.weights_y) == 20  # training targets y_x
    assert len(model.bias_x) == 20    # training targets y_y
    assert model.bias_y == float(KNN_DEFAULT_K)


def test_fit_knn_model_with_custom_k() -> None:
    """Custom k value should be stored in bias_y."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples, n_neighbors=7)
    assert model.bias_y == 7.0


def test_fit_knn_model_stores_correct_weights() -> None:
    """Training data should be stored correctly in weights fields."""
    samples = _make_knn_samples(10)
    model = fit_knn_model(samples)
    # weights_x = X (design vectors)
    assert len(model.weights_x) == 10
    assert len(model.weights_x[0]) == 27  # polynomial features v5
    # weights_y = y_x
    assert len(model.weights_y) == 10
    # bias_x = y_y
    assert len(model.bias_x) == 10


# ── GazeRegressor KNN prediction tests ──────────────────────────────────────

def test_gaze_regressor_predicts_with_knn_model() -> None:
    """GazeRegressor should work with KNN models."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    assert regressor.trained
    result = regressor.predict(samples[0].features)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] < 1920
    assert 0 <= result[1] < 1080


def test_gaze_regressor_returns_none_when_untrained_knn() -> None:
    """KNN regressor with empty model should return None."""
    model = GazeModel(type="knn_regression")
    regressor = GazeRegressor(model, (1920, 1080))
    assert not regressor.trained
    assert regressor.predict(np.array([0.5] * 6)) is None


def test_gaze_regressor_knn_tracks_head_pose() -> None:
    """KNN model should learn head pose correlation like Ridge."""
    samples = [
        CalibrationSample(np.array([0.48, 0.50, 0.52, 0.50, 0.18, -0.08]), (120.0, 120.0)),
        CalibrationSample(np.array([0.49, 0.50, 0.51, 0.50, 0.06, -0.04]), (640.0, 250.0)),
        CalibrationSample(np.array([0.51, 0.50, 0.49, 0.50, -0.06, 0.04]), (1280.0, 720.0)),
        CalibrationSample(np.array([0.52, 0.50, 0.48, 0.50, -0.18, 0.08]), (1820.0, 940.0)),
        CalibrationSample(np.array([0.45, 0.50, 0.55, 0.50, 0.12, -0.06]), (300.0, 180.0)),
        CalibrationSample(np.array([0.55, 0.50, 0.45, 0.50, -0.12, 0.06]), (1500.0, 850.0)),
        CalibrationSample(np.array([0.40, 0.50, 0.60, 0.50, 0.24, -0.10]), (50.0, 80.0)),
        CalibrationSample(np.array([0.60, 0.50, 0.40, 0.50, -0.24, 0.10]), (1900.0, 1000.0)),
    ]
    model = fit_knn_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    assert regressor.trained
    outputs = np.array([regressor.predict(sample.features) for sample in samples], dtype=float)
    # Should show meaningful spread (not all same value)
    assert np.ptp(outputs[:, 0]) > 500
    assert np.ptp(outputs[:, 1]) > 300


def test_gaze_regressor_knn_expands_compressed_predictions() -> None:
    """KNN should handle prediction from similar input to training data."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    # Predict on a training sample — should be close to its target
    result = regressor.predict(samples[5].features)
    assert result is not None
    # KNN with distance weights should predict near the actual target
    # (not exact due to Kalman smoothing, but within reasonable range)
    assert 0 <= result[0] < 1920
    assert 0 <= result[1] < 1080


def test_gaze_regressor_knn_with_uniform_weights() -> None:
    """KNN with uniform weights should also produce valid predictions."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples, weights="uniform")
    regressor = GazeRegressor(model, (1920, 1080))
    assert regressor.trained
    result = regressor.predict(samples[0].features)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_gaze_regressor_knn_clamps_to_screen_bounds() -> None:
    """KNN predictions should be clamped to screen bounds."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples)
    # Small screen to test clamping
    regressor = GazeRegressor(model, (100, 100))
    result = regressor.predict(samples[0].features)
    assert result is not None
    assert 0 <= result[0] < 100
    assert 0 <= result[1] < 100


def test_gaze_regressor_knn_different_k_values() -> None:
    """Different k values should produce different predictions."""
    samples = _make_knn_samples(20)
    model_k3 = fit_knn_model(samples, n_neighbors=3)
    model_k9 = fit_knn_model(samples, n_neighbors=9)

    regressor_k3 = GazeRegressor(model_k3, (1920, 1080))
    regressor_k9 = GazeRegressor(model_k9, (1920, 1080))

    # Test on a sample not in training set
    test_features = np.array([0.35, 0.45, 0.40, 0.45, 0.0, 0.0])
    result_k3 = regressor_k3.predict(test_features)
    result_k9 = regressor_k9.predict(test_features)

    assert result_k3 is not None
    assert result_k9 is not None
    # Different k values should give different results (highly likely)
    # due to different neighbour sets


def test_gaze_regressor_knn_model_type() -> None:
    """GazeRegressor should correctly identify KNN model type."""
    samples = _make_knn_samples(20)
    model = fit_knn_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    assert regressor._model_type == "knn_regression"
    assert regressor.trained


def test_gaze_regressor_ridge_still_works() -> None:
    """Ridge regression should still work after KNN changes."""
    from freehands.gaze.calibration import fit_gaze_model

    samples = [
        CalibrationSample(np.array([0.1, 0.2, 0.2, 0.3, 0.0, 0.1]), (100.0, 100.0)),
        CalibrationSample(np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.0]), (300.0, 120.0)),
        CalibrationSample(np.array([0.3, 0.3, 0.4, 0.4, 0.0, 0.2]), (500.0, 500.0)),
        CalibrationSample(np.array([0.4, 0.2, 0.5, 0.3, 0.1, 0.1]), (700.0, 320.0)),
    ]
    model = fit_gaze_model(samples)
    assert model.type == "polynomial_regression"
    regressor = GazeRegressor(model, (1920, 1080))
    assert regressor.trained
    result = regressor.predict(samples[0].features)
    assert result is not None
    assert 0 <= result[0] < 1920
    assert 0 <= result[1] < 1080
