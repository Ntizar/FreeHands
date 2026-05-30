"""Tests for the Kalman cursor smoothing filter.

Covers: basic update, velocity tracking, reset, parameter tuning,
and integration with GazeRegressor.
"""
from __future__ import annotations

import numpy as np

from freehands.gaze.kalman_filter import KalmanCursorFilter, KalmanParams
from freehands.gaze.calibration import GazeRegressor
from freehands.profiles.store import GazeModel


class TestKalmanBasic:
    """Core Kalman filter behaviour."""

    def test_first_update_returns_measurement(self) -> None:
        kf = KalmanCursorFilter()
        result = kf.update((100.0, 200.0))
        # First frame: filter returns the raw measurement (uninitialised)
        assert result == (100.0, 200.0)
        assert kf.is_initialized

    def test_second_frame_is_smoothed(self) -> None:
        kf = KalmanCursorFilter()
        kf.update((100.0, 200.0))  # initialise
        # Second frame slightly different — should be smoothed toward first
        result = kf.update((102.0, 201.0))
        # Should be between the two measurements, closer to first
        assert 100.0 < result[0] < 102.0
        assert 200.0 < result[1] < 201.0

    def test_rapid_movement_tracks_correctly(self) -> None:
        """Large jumps should be tracked with some latency (Kalman property)."""
        kf = KalmanCursorFilter()
        kf.update((100.0, 100.0))
        # Jump far away
        result = kf.update((500.0, 500.0))
        # Should move significantly toward the new position
        assert result[0] > 100.0
        assert result[1] > 100.0
        # But not fully there (smoothness)
        assert result[0] < 500.0
        assert result[1] < 500.0

    def test_reset_clears_state(self) -> None:
        kf = KalmanCursorFilter()
        kf.update((100.0, 200.0))
        assert kf.is_initialized
        kf.reset()
        assert not kf.is_initialized
        # After reset, first update should return raw measurement again
        result = kf.update((300.0, 400.0))
        assert result == (300.0, 400.0)


class TestKalmanVelocity:
    """Velocity estimation from the Kalman state."""

    def test_zero_velocity_at_start(self) -> None:
        kf = KalmanCursorFilter()
        kf.update((100.0, 200.0))
        vx, vy = kf.velocity
        assert vx == 0.0
        assert vy == 0.0

    def test_velocity_builds_with_movement(self) -> None:
        kf = KalmanCursorFilter()
        kf.update((100.0, 100.0))
        # Move in a consistent direction
        for i in range(1, 10):
            kf.update((100.0 + i * 10, 100.0 + i * 10))
        vx, vy = kf.velocity
        # Both should be positive and roughly 10 px/frame
        assert vx > 5.0
        assert vy > 5.0

    def test_velocity_reverses_on_direction_change(self) -> None:
        kf = KalmanCursorFilter()
        kf.update((100.0, 100.0))
        for i in range(1, 5):
            kf.update((100.0 + i * 20, 100.0))
        # Now reverse direction
        for i in range(5, 10):
            kf.update((200.0 - (i - 5) * 20, 100.0))
        vx, _ = kf.velocity
        # Velocity should be negative (moving left)
        assert vx < 0.0


class TestKalmanParameters:
    """Parameter tuning effects."""

    def test_high_process_noise_more_responsive(self) -> None:
        """High process noise = faster tracking, more jitter."""
        kf = KalmanCursorFilter(KalmanParams(process_noise=200.0))
        kf.update((100.0, 100.0))
        result = kf.update((500.0, 500.0))
        # With high process noise, should be closer to the measurement
        assert result[0] > 300.0  # closer to 500 than default

    def test_low_process_noise_smoother(self) -> None:
        """Low process noise = smoother, more latency."""
        kf = KalmanCursorFilter(KalmanParams(process_noise=0.01))
        kf.update((100.0, 100.0))
        result = kf.update((500.0, 500.0))
        # With very low process noise, should lag significantly behind
        assert result[0] < 450.0  # notably behind 500

    def test_high_measurement_noise_smoothing(self) -> None:
        """High measurement noise = trust the model more."""
        kf = KalmanCursorFilter(KalmanParams(measurement_noise=20000.0))
        kf.update((100.0, 100.0))
        # Add a noisy jump
        result = kf.update((500.0, 500.0))
        # Should be very smooth — barely move from 100
        assert result[0] < 250.0

    def test_default_params_reasonable(self) -> None:
        """Default params should produce sensible smoothing."""
        kf = KalmanCursorFilter()
        # Feed a steady stream of identical measurements
        converged = (0.0, 0.0)
        for _ in range(20):
            converged = kf.update((500.0, 500.0))
        # Should converge to exactly 500, 500
        assert abs(converged[0] - 500.0) < 1.0
        assert abs(converged[1] - 500.0) < 1.0


class TestKalmanEdgeCases:
    """Boundary and edge cases."""

    def test_unclamped_coordinates(self) -> None:
        """Kalman itself doesn't clamp (done in GazeRegressor.predict)."""
        kf = KalmanCursorFilter()
        result = kf.update((-1000.0, -1000.0))
        assert result[0] < 0.0  # Kalman doesn't clamp
        assert result[1] < 0.0

    def test_repeated_identical_measurements_converge(self) -> None:
        """Repeated identical inputs should converge to that value."""
        kf = KalmanCursorFilter()
        kf.update((960.0, 540.0))
        converged = (0.0, 0.0)
        for _ in range(50):
            converged = kf.update((960.0, 540.0))
        assert abs(converged[0] - 960.0) < 0.5
        assert abs(converged[1] - 540.0) < 0.5

    def test_multiple_reset_cycle(self) -> None:
        """Multiple reset/update cycles should work correctly."""
        kf = KalmanCursorFilter()
        for start_x in [100, 300, 500, 700]:
            kf.reset()
            result = kf.update((float(start_x), 0.0))
            assert result == (float(start_x), 0.0)


class TestKalmanIntegration:
    """Integration with GazeRegressor — verify the Kalman replaces EMA."""

    def test_regressor_has_kalman_attribute(self) -> None:
        """GazeRegressor should have a _kalman attribute."""
        model = GazeModel(
            feature_version=1,
            weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        )
        regressor = GazeRegressor(model, (500, 500))
        assert hasattr(regressor, "_kalman")
        assert regressor._kalman.is_initialized is False

    def test_regressor_predict_smooths_with_kalman(self) -> None:
        """Consecutive predictions should be smoothed by Kalman."""
        model = GazeModel(
            feature_version=1,
            weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        )
        regressor = GazeRegressor(model, (500, 500))

        # First prediction: raw (uninitialised Kalman)
        r1 = regressor.predict(np.array([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
        assert r1 == (100, 200)

        # Second prediction with slightly different input
        r2 = regressor.predict(np.array([1.02, 2.02, 0.0, 0.0, 0.0, 0.0]))
        # Should be smoothed — not exactly 102, 202
        assert r2 is not None
        assert 100 <= r2[0] <= 102
        assert 200 <= r2[1] <= 202

    def test_regressor_with_custom_kalman_params(self) -> None:
        """GazeRegressor should accept custom KalmanParams."""
        model = GazeModel(
            feature_version=1,
            weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        )
        params = KalmanParams(process_noise=1.0)  # very smooth
        regressor = GazeRegressor(model, (500, 500), kalman_params=params)
        assert regressor._kalman._params.process_noise == 1.0

    def test_regressor_legacy_model_still_works(self) -> None:
        """Legacy models should work with the Kalman filter."""
        model = GazeModel(
            feature_version=1,
            weights_x=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            weights_y=[0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        )
        regressor = GazeRegressor(model, (500, 500))
        result = regressor.predict(np.array([1.2, 2.4, 0.0, 0.0, 0.0, 0.0]))
        assert result == (120, 240)
