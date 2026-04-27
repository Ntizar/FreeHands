import numpy as np

from freehands.gaze.calibration import (
    CURRENT_GAZE_FEATURE_VERSION,
    LEGACY_GAZE_FEATURE_VERSION,
    CalibrationSample,
    GazeRegressor,
    aggregate_gaze_features,
    build_gaze_design_vector,
    calibrate_output_axis,
    fit_gaze_model,
    gaze_model_has_signal,
    gaze_model_is_current,
    gaze_model_is_usable,
    gaze_model_weight_norm,
)
from freehands.profiles.store import GazeModel


def test_build_gaze_design_vector_keeps_legacy_features() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(
        build_gaze_design_vector(raw, LEGACY_GAZE_FEATURE_VERSION),
        raw,
    )


def test_build_gaze_design_vector_reduces_head_pose_weight_in_current_version() -> None:
    raw = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    expected = np.array([0.2, 0.4, 0.6, 0.8, 0.35, 0.42])
    np.testing.assert_allclose(
        build_gaze_design_vector(raw, CURRENT_GAZE_FEATURE_VERSION),
        expected,
    )


def test_aggregate_gaze_features_uses_weights() -> None:
    vectors = [np.array([0.0, 0.0]), np.array([10.0, 20.0])]
    combined = aggregate_gaze_features(vectors, weights=[1.0, 3.0])
    np.testing.assert_allclose(combined, np.array([7.5, 15.0]))


def test_calibrate_output_axis_expands_compressed_predictions() -> None:
    predicted = np.array([100.0, 200.0, 300.0])
    target = np.array([50.0, 250.0, 450.0])
    gain, bias = calibrate_output_axis(predicted, target)
    corrected = predicted * gain + bias
    np.testing.assert_allclose(corrected, target)


def test_gaze_regressor_supports_legacy_models() -> None:
    model = GazeModel(
        feature_version=LEGACY_GAZE_FEATURE_VERSION,
        weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    )
    regressor = GazeRegressor(model, (500, 500))
    assert regressor.predict(np.array([1.2, 2.4, 0.0, 0.0, 0.0, 0.0])) == (120, 240)


def test_fit_gaze_model_marks_current_feature_version() -> None:
    samples = [
        CalibrationSample(np.array([0.1, 0.2, 0.2, 0.3, 0.0, 0.1]), (100.0, 100.0)),
        CalibrationSample(np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.0]), (300.0, 120.0)),
        CalibrationSample(np.array([0.3, 0.3, 0.4, 0.4, 0.0, 0.2]), (500.0, 500.0)),
        CalibrationSample(np.array([0.4, 0.2, 0.5, 0.3, 0.1, 0.1]), (700.0, 320.0)),
    ]
    model = fit_gaze_model(samples)
    assert model.feature_version == CURRENT_GAZE_FEATURE_VERSION
    assert len(model.weights_x) == 6
    assert len(model.weights_y) == 6


def test_fit_gaze_model_can_expand_small_feature_ranges_to_screen_space() -> None:
    samples = [
        CalibrationSample(np.array([0.42, 0.48, 0.58, 0.47, 0.02, 0.00]), (120.0, 120.0)),
        CalibrationSample(np.array([0.46, 0.47, 0.54, 0.46, 0.01, -0.01]), (640.0, 160.0)),
        CalibrationSample(np.array([0.52, 0.46, 0.48, 0.45, -0.01, 0.00]), (1280.0, 190.0)),
        CalibrationSample(np.array([0.57, 0.45, 0.43, 0.44, -0.02, 0.01]), (1820.0, 220.0)),
    ]
    model = fit_gaze_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    outputs = np.array([regressor.predict(sample.features) for sample in samples], dtype=float)
    assert np.ptp(outputs[:, 0]) > 900


def test_gaze_model_has_signal_detects_dead_centered_model() -> None:
    dead = GazeModel(
        feature_version=CURRENT_GAZE_FEATURE_VERSION,
        weights_x=[0.0] * 6,
        weights_y=[0.0] * 6,
        bias_x=960.0,
        bias_y=540.0,
    )
    assert gaze_model_weight_norm(dead) == 0.0
    assert not gaze_model_has_signal(dead)


def test_gaze_model_is_usable_requires_current_feature_version() -> None:
    old = GazeModel(
        feature_version=LEGACY_GAZE_FEATURE_VERSION,
        weights_x=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    current = GazeModel(
        feature_version=CURRENT_GAZE_FEATURE_VERSION,
        weights_x=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert not gaze_model_is_current(old)
    assert not gaze_model_is_usable(old)
    assert gaze_model_is_usable(current)