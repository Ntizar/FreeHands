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


LEGACY_GAZE_FEATURE_VERSION = 1
CURRENT_GAZE_FEATURE_VERSION = 4
EYE_SIGNAL_WEIGHT = 0.65
HEAD_POSE_WEIGHT = 1.85
MIN_OUTPUT_GAIN = 0.65
MAX_OUTPUT_GAIN = 25.0
MIN_GAZE_WEIGHT_NORM = 1e-3


@dataclass
class CalibrationSample:
    features: np.ndarray
    target_xy: tuple[float, float]   # screen pixels


def build_gaze_design_vector(
    features: np.ndarray,
    feature_version: int = CURRENT_GAZE_FEATURE_VERSION,
) -> np.ndarray:
    raw = np.asarray(features, dtype=float).reshape(-1)
    if raw.size != 6 or feature_version <= LEGACY_GAZE_FEATURE_VERSION:
        return raw

    left_x, left_y, right_x, right_y, head_x, head_y = raw
    return np.array([
        left_x * EYE_SIGNAL_WEIGHT,
        left_y * EYE_SIGNAL_WEIGHT,
        right_x * EYE_SIGNAL_WEIGHT,
        right_y * EYE_SIGNAL_WEIGHT,
        head_x * HEAD_POSE_WEIGHT,
        head_y * HEAD_POSE_WEIGHT,
    ], dtype=float)


def aggregate_gaze_features(
    vectors: list[np.ndarray],
    weights: list[float] | np.ndarray | None = None,
) -> np.ndarray:
    if not vectors:
        raise ValueError("Need at least one gaze vector to aggregate")

    matrix = np.stack([np.asarray(vector, dtype=float).reshape(-1) for vector in vectors])
    if weights is None:
        return np.mean(matrix, axis=0)

    sample_weights = np.asarray(weights, dtype=float).reshape(-1)
    if sample_weights.size != matrix.shape[0]:
        raise ValueError("Weights must match the number of gaze vectors")
    sample_weights = np.clip(sample_weights, 1e-6, None)
    return np.average(matrix, axis=0, weights=sample_weights)


def calibrate_output_axis(predicted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    tgt = np.asarray(target, dtype=float).reshape(-1)
    if pred.size != tgt.size:
        raise ValueError("Predicted and target gaze axes must have the same length")

    pred_centered = pred - pred.mean()
    tgt_centered = tgt - tgt.mean()
    denom = float(pred_centered @ pred_centered)
    if denom < 1e-6:
        return 1.0, float(tgt.mean() - pred.mean())

    gain = float((pred_centered @ tgt_centered) / denom)
    gain = float(np.clip(gain, MIN_OUTPUT_GAIN, MAX_OUTPUT_GAIN))
    bias = float(tgt.mean() - gain * pred.mean())
    return gain, bias


def gaze_model_weight_norm(model: GazeModel) -> float:
    weights_x = np.asarray(model.weights_x, dtype=float)
    weights_y = np.asarray(model.weights_y, dtype=float)
    if weights_x.size == 0 and weights_y.size == 0:
        return 0.0
    return float(np.linalg.norm(np.concatenate([weights_x, weights_y])))


def gaze_model_has_signal(model: GazeModel, min_norm: float = MIN_GAZE_WEIGHT_NORM) -> bool:
    return gaze_model_weight_norm(model) >= min_norm


def gaze_model_is_current(model: GazeModel) -> bool:
    return model.feature_version == CURRENT_GAZE_FEATURE_VERSION


def gaze_model_is_usable(model: GazeModel) -> bool:
    return gaze_model_is_current(model) and gaze_model_has_signal(model)


class GazeRegressor:
    """Predicts (x, y) pixels from a feature vector + Kalman-style smoothing."""

    def __init__(self, model: GazeModel, screen_size: tuple[int, int]) -> None:
        self._wx = np.array(model.weights_x) if model.weights_x else None
        self._wy = np.array(model.weights_y) if model.weights_y else None
        self._bx = model.bias_x
        self._by = model.bias_y
        self._offset = model.personal_offset
        self._feature_version = model.feature_version
        self._screen_w, self._screen_h = screen_size
        self._smoothed: np.ndarray | None = None
        self._alpha = 0.5  # exponential smoothing factor

    @property
    def trained(self) -> bool:
        return self._wx is not None and self._wy is not None

    def _design_vector(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=float).reshape(-1)
        design = build_gaze_design_vector(raw, self._feature_version)
        if self._wx is not None and design.shape[0] != self._wx.shape[0] and raw.shape[0] == self._wx.shape[0]:
            return raw
        return design

    def predict(self, features: np.ndarray) -> tuple[int, int] | None:
        if not self.trained:
            return None
        design = self._design_vector(features)
        if design.shape[0] != self._wx.shape[0] or design.shape[0] != self._wy.shape[0]:
            return None
        x = float(design @ self._wx + self._bx + self._offset["x"])
        y = float(design @ self._wy + self._by + self._offset["y"])
        x = max(0, min(self._screen_w - 1, x))
        y = max(0, min(self._screen_h - 1, y))
        raw = np.array([x, y])
        if self._smoothed is None:
            self._smoothed = raw
        else:
            self._smoothed = self._alpha * raw + (1 - self._alpha) * self._smoothed
        return int(self._smoothed[0]), int(self._smoothed[1])


def fit_gaze_model(samples: list[CalibrationSample], alpha: float = 0.5) -> GazeModel:
    """Train a ridge regressor and return a serialisable :class:`GazeModel`."""
    if len(samples) < 4:
        raise ValueError("Need at least 4 calibration samples")
    X = np.stack([
        build_gaze_design_vector(s.features, CURRENT_GAZE_FEATURE_VERSION)
        for s in samples
    ])
    y = np.array([s.target_xy for s in samples])
    rx = Ridge(alpha=alpha).fit(X, y[:, 0])
    ry = Ridge(alpha=alpha).fit(X, y[:, 1])
    pred_x = rx.predict(X)
    pred_y = ry.predict(X)
    gain_x, offset_x = calibrate_output_axis(pred_x, y[:, 0])
    gain_y, offset_y = calibrate_output_axis(pred_y, y[:, 1])
    return GazeModel(
        type="ridge_regression",
        feature_version=CURRENT_GAZE_FEATURE_VERSION,
        weights_x=[float(v) for v in (rx.coef_ * gain_x)],
        weights_y=[float(v) for v in (ry.coef_ * gain_y)],
        bias_x=float(rx.intercept_ * gain_x + offset_x),
        bias_y=float(ry.intercept_ * gain_y + offset_y),
    )


def update_profile_with_gaze(profile: Profile, model: GazeModel) -> Profile:
    profile.gaze_model = model
    return profile
