import numpy as np
import pytest

from freehands.gaze.calibration import (
    CURRENT_GAZE_FEATURE_VERSION,
    LEGACY_GAZE_FEATURE_VERSION,
    CalibrationSample,
    GazeRegressor,
    aggregate_gaze_features,
    build_gaze_design_vector,
    calibrate_output_axis,
    fit_gaze_model,
    fit_gp_model,
    gp_model_has_data,
    gp_predict,
    gaze_model_has_signal,
    gaze_model_is_current,
    gaze_model_is_usable,
    gaze_model_weight_norm,
    update_gp_model,
    GP_MIN_SAMPLES,
)
from freehands.profiles.store import GPGazeModel, GazeModel


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
    samples = [
        CalibrationSample(np.array([0.42, 0.48, 0.58, 0.47, 0.02, 0.00]), (120.0, 120.0)),
        CalibrationSample(np.array([0.46, 0.47, 0.54, 0.46, 0.01, -0.01]), (640.0, 160.0)),
        CalibrationSample(np.array([0.52, 0.46, 0.48, 0.45, -0.01, 0.00]), (1280.0, 190.0)),
        CalibrationSample(np.array([0.57, 0.45, 0.43, 0.44, -0.02, 0.01]), (1820.0, 220.0)),
    ]
    model = fit_gaze_model(samples)
    regressor = GazeRegressor(model, (1920, 1080))
    outputs = np.array([regressor.predict(sample.features) for sample in samples], dtype=float)
    assert np.ptp(outputs[:, 0]) > 100
    assert outputs[0, 0] < outputs[-1, 0]


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
    np.testing.assert_allclose(expanded[:6], raw)
    np.testing.assert_allclose(expanded[6:12], raw ** 2)
    assert expanded[12] == 1.0 * 2.0
    assert expanded[13] == 1.0 * 3.0
    assert expanded[26] == 5.0 * 6.0


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
    model = GazeModel(
        feature_version=LEGACY_GAZE_FEATURE_VERSION,
        weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
    )
    regressor = GazeRegressor(model, (500, 500))
    assert regressor.predict(np.array([1.2, 2.4, 0.0, 0.0, 0.0, 0.0])) == (120, 240)


# ── Gaussian Process auto-calibration tests ──────────────────────────────

def _make_gp_samples(n: int = 20) -> list[CalibrationSample]:
    """Generate deterministic calibration samples for GP testing."""
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


def test_gp_model_requires_minimum_samples() -> None:
    from freehands.gaze.calibration import GP_MIN_SAMPLES
    samples = _make_gp_samples(GP_MIN_SAMPLES - 1)
    with pytest.raises(ValueError, match="at least"):
        fit_gp_model(samples)


def test_fit_gp_model_returns_serialisable_model() -> None:
    from freehands.profiles.store import GPGazeModel
    samples = _make_gp_samples(20)
    model = fit_gp_model(samples)
    assert isinstance(model, GPGazeModel)
    assert model.n_samples == 20
    assert model.feature_version == CURRENT_GAZE_FEATURE_VERSION
    assert len(model.training_features) == 20
    assert len(model.training_targets_x) == 20
    assert len(model.training_targets_y) == 20


def test_gp_regressor_predicts_with_fitted_model() -> None:
    from freehands.gaze.calibration import GPGazeRegressor
    samples = _make_gp_samples(20)
    gp = fit_gp_model(samples)
    regressor = GPGazeRegressor(gp, (1920, 1080))
    assert regressor.trained
    result = regressor.predict(samples[0].features)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert 0 <= result[0] < 1920
    assert 0 <= result[1] < 1080


def test_gp_regressor_returns_none_when_untrained() -> None:
    from freehands.gaze.calibration import GPGazeRegressor
    gp = GPGazeModel(n_samples=0)
    regressor = GPGazeRegressor(gp, (1920, 1080))
    assert not regressor.trained
    assert regressor.predict(np.array([0.5, 0.5, 0.5, 0.5, 0.0, 0.0])) is None


def test_gp_update_adds_new_samples() -> None:
    samples = _make_gp_samples(20)
    gp = fit_gp_model(samples)
    assert gp.n_samples == 20
    new_samples = _make_gp_samples(5)
    for s in new_samples:
        s.target_xy = (s.target_xy[0] + 500, s.target_xy[1] + 300)
    gp_updated = update_gp_model(gp, new_samples)
    assert gp_updated.n_samples == 25


def test_gp_update_respects_max_samples_window() -> None:
    from freehands.gaze.calibration import GP_MAX_SAMPLES
    samples = _make_gp_samples(150)
    gp = fit_gp_model(samples)
    new_samples = _make_gp_samples(100)
    gp_updated = update_gp_model(gp, new_samples)
    assert gp_updated.n_samples <= GP_MAX_SAMPLES


def test_gp_model_has_data() -> None:
    assert not gp_model_has_data(GPGazeModel(n_samples=0))
    assert not gp_model_has_data(GPGazeModel(n_samples=GP_MIN_SAMPLES - 1))
    assert gp_model_has_data(GPGazeModel(n_samples=GP_MIN_SAMPLES))
    assert gp_model_has_data(GPGazeModel(n_samples=100))


def test_gp_regressor_update_method() -> None:
    from freehands.gaze.calibration import GPGazeRegressor
    samples = _make_gp_samples(20)
    gp = fit_gp_model(samples)
    regressor = GPGazeRegressor(gp, (1920, 1080))
    assert regressor.trained
    new_samples = _make_gp_samples(5)
    regressor.update(new_samples)
    assert regressor._gp.n_samples == 25


def test_gp_predict_module_function() -> None:
    samples = _make_gp_samples(20)
    gp = fit_gp_model(samples)
    result = gp_predict(gp, samples[0].features)
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_gp_predict_returns_none_with_insufficient_data() -> None:
    gp = GPGazeModel(n_samples=5)
    assert gp_predict(gp, np.array([0.5] * 6)) is None


def test_gp_model_serialisable_to_dict() -> None:
    """Ensure GPGazeModel can be round-tripped through JSON."""
    samples = _make_gp_samples(10)
    gp = fit_gp_model(samples)
    data = gp.model_dump()
    assert "training_features" in data
    assert "n_samples" in data
    assert data["n_samples"] == 10
    gp2 = GPGazeModel.model_validate(data)
    assert gp2.n_samples == 10
    assert len(gp2.training_features) == 10
