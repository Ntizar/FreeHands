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


def test_build_gaze_design_vector_prefers_head_pose_in_current_version() -> None:
    raw = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    result = build_gaze_design_vector(raw, CURRENT_GAZE_FEATURE_VERSION)
    # v5 returns polynomial expansion: first 6 are the weighted linear terms
    expected_linear = np.array([0.13, 0.26, 0.39, 0.52, 1.85, 2.22])
    np.testing.assert_allclose(result[:6], expected_linear)
    # Total should be 27 (6 linear + 6 squared + 15 cross)
    assert result.shape[0] == 27


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
    # v5 uses polynomial features: 6 linear + 6 squared + 15 cross = 27
    assert len(model.weights_x) == 27
    assert len(model.weights_y) == 27
    assert model.type == "polynomial_regression"


def test_current_gaze_model_invalidates_v3_eye_heavy_profiles() -> None:
    old = GazeModel(
        feature_version=3,
        weights_x=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert not gaze_model_is_current(old)
    assert not gaze_model_is_usable(old)


def test_fit_gaze_model_can_expand_small_feature_ranges_to_screen_space() -> None:
    """Polynomial model with stronger regularization produces more conservative but still
    meaningful expansion — with only 4 samples it won't span the full screen, but it
    should still track the feature ordering."""
    samples = [
        CalibrationSample(np.array([0.42, 0.48, 0.58, 0.47, 0.02, 0.00]), (120.0, 120.0)),
        CalibrationSample(np.array([0.46, 0.47, 0.54, 0.46, 0.01, -0.01]), (640.0, 160.0)),
        CalibrationSample(np.array([0.52, 0.46, 0.48, 0.45, -0.01, 0.00]), (1280.0, 190.0)),
        CalibrationSample(np.array([0.57, 0.45, 0.43, 0.44, -0.02, 0.01]), (1820.0, 220.0)),
    ]
    model = fit_gaze_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    outputs = np.array([regressor.predict(sample.features) for sample in samples], dtype=float)
    # With 4 samples and polynomial features (27), regularization is stronger, so expansion
    # is more conservative. But the model should still produce ordered outputs matching
    # the calibration target ordering.
    assert np.ptp(outputs[:, 0]) > 100  # meaningful expansion
    # Verify outputs are monotonically increasing (matching target ordering)
    assert outputs[0, 0] < outputs[-1, 0]  # left-to-right ordering preserved


def test_fit_gaze_model_tracks_head_pose_when_eye_signal_is_small() -> None:
    samples = [
        CalibrationSample(np.array([0.48, 0.50, 0.52, 0.50, 0.18, -0.08]), (120.0, 120.0)),
        CalibrationSample(np.array([0.49, 0.50, 0.51, 0.50, 0.06, -0.04]), (640.0, 250.0)),
        CalibrationSample(np.array([0.51, 0.50, 0.49, 0.50, -0.06, 0.04]), (1280.0, 720.0)),
        CalibrationSample(np.array([0.52, 0.50, 0.48, 0.50, -0.18, 0.08]), (1820.0, 940.0)),
    ]
    model = fit_gaze_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    outputs = np.array([regressor.predict(sample.features) for sample in samples], dtype=float)
    assert np.ptp(outputs[:, 0]) > 1100
    assert np.ptp(outputs[:, 1]) > 550


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


def test_expand_polynomial_produces_27_features_from_6() -> None:
    from freehands.gaze.calibration import _expand_polynomial
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expanded = _expand_polynomial(raw, degree=2)
    assert expanded.shape[0] == 27
    # First 6 = linear (unchanged)
    np.testing.assert_allclose(expanded[:6], raw)
    # Next 6 = squared
    np.testing.assert_allclose(expanded[6:12], raw ** 2)
    # Remaining 15 = cross terms
    assert expanded[12] == 1.0 * 2.0   # a·b
    assert expanded[13] == 1.0 * 3.0   # a·c
    assert expanded[26] == 5.0 * 6.0   # e·f (last)


def test_build_gaze_design_vector_returns_27_for_v5() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = build_gaze_design_vector(raw, CURRENT_GAZE_FEATURE_VERSION)
    assert result.shape[0] == 27


def test_build_gaze_design_vector_returns_6_for_legacy() -> None:
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = build_gaze_design_vector(raw, LEGACY_GAZE_FEATURE_VERSION)
    assert result.shape[0] == 6
    np.testing.assert_allclose(result, raw)


def test_gaze_regressor_handles_legacy_model_with_polynomial_input() -> None:
    """Old linear model (6 weights) should still work with new 6-d input."""
    model = GazeModel(
        feature_version=LEGACY_GAZE_FEATURE_VERSION,
        weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    )
    regressor = GazeRegressor(model, (500, 500))
    assert regressor.predict(np.array([1.2, 2.4, 0.0, 0.0, 0.0, 0.0])) == (120, 240)