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
from .kalman_filter import KalmanCursorFilter, KalmanParams


LEGACY_GAZE_FEATURE_VERSION = 1
CURRENT_GAZE_FEATURE_VERSION = 5   # polynomial features (degree 2) for non-linear gaze mapping
EYE_SIGNAL_WEIGHT = 0.65
HEAD_POSE_WEIGHT = 1.85
MIN_OUTPUT_GAIN = 0.65
MAX_OUTPUT_GAIN = 25.0
MIN_GAZE_WEIGHT_NORM = 1e-3


@dataclass
class CalibrationSample:
    features: np.ndarray
    target_xy: tuple[float, float]   # screen pixels


def _expand_polynomial(features: np.ndarray, degree: int = 2) -> np.ndarray:
    """Expand a feature vector with polynomial terms (degree 2).

    For a 6-d input [a, b, c, d, e, f] produces:
      [a, b, c, d, e, f,                  # linear
       a², b², c², d², e², f²,             # squared
       a·b, a·c, a·d, a·e, a·f,             # cross (upper triangle)
       b·c, b·d, b·e, b·f,
       c·d, c·e, c·f,
       d·e, d·f,
       e·f]                                 # 6 + 6 + 15 = 27 total
    """
    raw = np.asarray(features, dtype=float).reshape(-1)
    n = raw.size
    terms = list(raw)                          # linear
    terms.extend(raw ** 2)                      # squared
    for i in range(n):
        for j in range(i + 1, n):
            terms.append(raw[i] * raw[j])       # cross terms
    return np.array(terms, dtype=float)


def build_gaze_design_vector(
    features: np.ndarray,
    feature_version: int = CURRENT_GAZE_FEATURE_VERSION,
) -> np.ndarray:
    raw = np.asarray(features, dtype=float).reshape(-1)
    if raw.size != 6 or feature_version <= LEGACY_GAZE_FEATURE_VERSION:
        return raw

    left_x, left_y, right_x, right_y, head_x, head_y = raw

    if feature_version >= 5:
        # Polynomial expansion (degree 2) for non-linear gaze mapping
        linear = np.array([
            left_x * EYE_SIGNAL_WEIGHT,
            left_y * EYE_SIGNAL_WEIGHT,
            right_x * EYE_SIGNAL_WEIGHT,
            right_y * EYE_SIGNAL_WEIGHT,
            head_x * HEAD_POSE_WEIGHT,
            head_y * HEAD_POSE_WEIGHT,
        ], dtype=float)
        return _expand_polynomial(linear, degree=2)
    else:
        # Linear (v2–v4)
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

    POLYNOMIAL_FEATURE_DIM = 27   # 6 linear + 6 squared + 15 cross

    def __init__(
        self,
        model: GazeModel,
        screen_size: tuple[int, int],
        *,
        kalman_params: KalmanParams | None = None,
    ) -> None:
        self._wx = np.array(model.weights_x) if model.weights_x else None
        self._wy = np.array(model.weights_y) if model.weights_y else None
        self._bx = model.bias_x
        self._by = model.bias_y
        self._offset = model.personal_offset
        self._feature_version = model.feature_version
        self._screen_w, self._screen_h = screen_size
        self._kalman = KalmanCursorFilter(kalman_params)

    @property
    def trained(self) -> bool:
        return self._wx is not None and self._wy is not None

    def _design_vector(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=float).reshape(-1)
        design = build_gaze_design_vector(raw, self._feature_version)
        # Handle dimension mismatch: linear model with polynomial input or vice-versa
        if self._wx is not None:
            wx_dim = self._wx.shape[0]
            if design.shape[0] != wx_dim and raw.shape[0] == 6:
                # Input is linear 6-d but model expects expanded features
                if wx_dim == self.POLYNOMIAL_FEATURE_DIM:
                    # Expand linear to polynomial for backward compat
                    linear = np.array([
                        raw[0] * EYE_SIGNAL_WEIGHT,
                        raw[1] * EYE_SIGNAL_WEIGHT,
                        raw[2] * EYE_SIGNAL_WEIGHT,
                        raw[3] * EYE_SIGNAL_WEIGHT,
                        raw[4] * HEAD_POSE_WEIGHT,
                        raw[5] * HEAD_POSE_WEIGHT,
                    ], dtype=float)
                    return _expand_polynomial(linear, degree=2)
                # Otherwise: linear model with linear input — use as-is
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
        # Kalman filter smoothing replaces the old EMA
        smoothed = self._kalman.update((x, y))
        return int(round(smoothed[0])), int(round(smoothed[1]))


def fit_gaze_model(samples: list[CalibrationSample], alpha: float = 0.5) -> GazeModel:
    """Train a polynomial ridge regressor and return a serialisable :class:`GazeModel`.

    For polynomial features (v5+), the feature count jumps from 6 to 27, so we
    use a higher regularisation strength to prevent over-fitting on small datasets.
    """
    if len(samples) < 4:
        raise ValueError("Need at least 4 calibration samples")
    X = np.stack([
        build_gaze_design_vector(s.features, CURRENT_GAZE_FEATURE_VERSION)
        for s in samples
    ])
    y = np.array([s.target_xy for s in samples])
    # Polynomial models (27 features) need stronger regularisation than linear (6 features)
    poly_alpha = alpha * 10.0 if X.shape[1] > 10 else alpha
    rx = Ridge(alpha=poly_alpha).fit(X, y[:, 0])
    ry = Ridge(alpha=poly_alpha).fit(X, y[:, 1])
    pred_x = rx.predict(X)
    pred_y = ry.predict(X)
    gain_x, offset_x = calibrate_output_axis(pred_x, y[:, 0])
    gain_y, offset_y = calibrate_output_axis(pred_y, y[:, 1])
    return GazeModel(
        type="polynomial_regression",
        feature_version=CURRENT_GAZE_FEATURE_VERSION,
        weights_x=[float(v) for v in (rx.coef_ * gain_x)],
        weights_y=[float(v) for v in (ry.coef_ * gain_y)],
        bias_x=float(rx.intercept_ * gain_x + offset_x),
        bias_y=float(ry.intercept_ * gain_y + offset_y),
    )


def update_profile_with_gaze(profile: Profile, model: GazeModel) -> Profile:
    profile.gaze_model = model
    return profile
