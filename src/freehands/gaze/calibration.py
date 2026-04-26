"""Personalised gaze→screen mapping (ridge regression, à la WebGazer).

Given pairs of (eye-feature vector, target screen pixel), fits two ridge
regressors (one per axis) and stores the weights in the user profile.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from ..profiles import Profile
from ..profiles.store import GazeModel


@dataclass
class CalibrationSample:
    features: np.ndarray
    target_xy: tuple[float, float]   # screen pixels


class GazeRegressor:
    """Predicts (x, y) pixels from a feature vector + Kalman-style smoothing."""

    def __init__(self, model: GazeModel, screen_size: tuple[int, int]) -> None:
        self._wx = np.array(model.weights_x) if model.weights_x else None
        self._wy = np.array(model.weights_y) if model.weights_y else None
        self._bx = model.bias_x
        self._by = model.bias_y
        self._offset = model.personal_offset
        self._screen_w, self._screen_h = screen_size
        self._smoothed: np.ndarray | None = None
        self._alpha = 0.35  # exponential smoothing factor

    @property
    def trained(self) -> bool:
        return self._wx is not None and self._wy is not None

    def predict(self, features: np.ndarray) -> tuple[int, int] | None:
        if not self.trained:
            return None
        x = float(features @ self._wx + self._bx + self._offset["x"])
        y = float(features @ self._wy + self._by + self._offset["y"])
        x = max(0, min(self._screen_w - 1, x))
        y = max(0, min(self._screen_h - 1, y))
        raw = np.array([x, y])
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = self._alpha * raw + (1 - self._alpha) * self._smoothed
        return int(self._smoothed[0]), int(self._smoothed[1])


def fit_gaze_model(samples: list[CalibrationSample], alpha: float = 1.0) -> GazeModel:
    """Train a ridge regressor and return a serialisable :class:`GazeModel`."""
    if len(samples) < 4:
        raise ValueError("Need at least 4 calibration samples")
    X = np.stack([s.features for s in samples])
    y = np.array([s.target_xy for s in samples])
    rx = Ridge(alpha=alpha).fit(X, y[:, 0])
    ry = Ridge(alpha=alpha).fit(X, y[:, 1])
    return GazeModel(
        type="ridge_regression",
        weights_x=[float(v) for v in rx.coef_],
        weights_y=[float(v) for v in ry.coef_],
        bias_x=float(rx.intercept_),
        bias_y=float(ry.intercept_),
    )


def update_profile_with_gaze(profile: Profile, model: GazeModel) -> Profile:
    profile.gaze_model = model
    return profile
