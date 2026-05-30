"""2-D Kalman filter for gaze-cursor smoothing.

Replaces the simple exponential moving average (EMA) used previously in
:mod:`freehands.gaze.calibration.GazeRegressor` with a proper Kalman filter
that fuses noisy gaze predictions with a constant-velocity motion model.

Key design decisions
-------------------
* **State vector**: [x, y, vx, vy] — position + velocity in screen pixels.
* **Process noise**: tuned so the filter tracks moderate head movements
  without being jittery.  The ``process_noise`` constructor parameter lets
  callers trade off smoothness vs. latency.
* **Measurement noise**: defaults to 400 px² (≈ 20 px stddev) which is
  generous enough to ignore per-frame jitter but tight enough to keep
  the cursor on target.
* **No external deps** — pure NumPy.

Usage
-----
>>> kf = KalmanCursorFilter()
>>> smoothed = kf.update((960, 540))  # returns (x, y)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanParams:
    """Tunable parameters for the gaze cursor Kalman filter."""

    # Process noise — how much we expect the cursor to move between frames.
    # Higher = more responsive, more jitter. Lower = smoother, more latency.
    process_noise: float = 25.0

    # Measurement noise — how much we trust the raw gaze prediction.
    # Higher = trust the model more (smoother), lower = trust the sensor.
    measurement_noise: float = 400.0

    # Initial estimate uncertainty.
    initial_uncertainty: float = 1000.0


class KalmanCursorFilter:
    """2-D constant-velocity Kalman filter for cursor smoothing.

    State: [x, y, vx, vy]  (4 elements)
    Measurement: [x, y]     (2 elements)
    """

    def __init__(self, params: KalmanParams | None = None) -> None:
        self._params = params or KalmanParams()
        self._initialized = False
        # State vector: [x, y, vx, vy]
        self._x: np.ndarray = np.zeros(4, dtype=float)
        # State covariance: 4×4
        self._P: np.ndarray = (
            self._params.initial_uncertainty * np.eye(4, dtype=float)
        )
        # State transition matrix F (constant velocity model)
        self._F = np.eye(4, dtype=float)
        # Control input (dt=1 frame interval, set in reset)
        self._dt = 1.0
        self._update_F()
        # Observation matrix H: we observe x, y directly
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        # Process noise covariance Q
        self._Q = self._build_Q()
        # Measurement noise covariance R
        self._R = np.array(
            [[self._params.measurement_noise, 0],
             [0, self._params.measurement_noise]], dtype=float
        )

    def _update_F(self) -> None:
        """Rebuild F with the current time step."""
        self._F = np.array([
            [1, 0, self._dt, 0],
            [0, 1, 0, self._dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

    def _build_Q(self) -> np.ndarray:
        """Build process noise covariance for constant-velocity model."""
        q = self._params.process_noise
        dt = self._dt
        return np.array([
            [dt**4 / 4 * q, 0, dt**3 / 2 * q, 0],
            [0, dt**4 / 4 * q, 0, dt**3 / 2 * q],
            [dt**3 / 2 * q, 0, dt**2 * q, 0],
            [0, dt**3 / 2 * q, 0, dt**2 * q],
        ], dtype=float)

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        """Run one Kalman predict + update cycle.

        Args:
            measurement: Raw (x, y) cursor position from the gaze regressor.

        Returns:
            Smoothed (x, y) position.
        """
        if not self._initialized:
            self._init_from_measurement(measurement)
            return measurement

        # ── Predict ───────────────────────────────────────────────────────
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q

        # ── Update ────────────────────────────────────────────────────────
        z = np.array(measurement, dtype=float)
        y = z - self._H @ self._x  # innovation
        S = self._H @ self._P @ self._H.T + self._R  # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain
        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ self._H) @ self._P

        return float(self._x[0]), float(self._x[1])

    def _init_from_measurement(self, measurement: tuple[float, float]) -> None:
        """Initialize state from the first measurement with zero velocity."""
        self._x = np.array([
            measurement[0], measurement[1], 0.0, 0.0
        ], dtype=float)
        self._P = self._params.initial_uncertainty * np.eye(4, dtype=float)
        self._initialized = True

    def reset(self) -> None:
        """Reset the filter to uninitialised state."""
        self._initialized = False
        self._x = np.zeros(4, dtype=float)
        self._P = self._params.initial_uncertainty * np.eye(4, dtype=float)

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/frame."""
        return float(self._x[2]), float(self._x[3])

    @property
    def is_initialized(self) -> bool:
        return self._initialized
