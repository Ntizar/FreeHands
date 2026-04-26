"""Eye-feature extraction with MediaPipe FaceMesh.

We extract a compact feature vector per frame that the personalised ridge
regression in :mod:`freehands.gaze.calibration` maps to screen coordinates.
The features intentionally avoid raw pixel coordinates so the model
generalises across small head movements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mediapipe as mp
    _mp_face_mesh = mp.solutions.face_mesh
except Exception:  # pragma: no cover
    mp = None
    _mp_face_mesh = None


# MediaPipe FaceMesh iris and eye-corner landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)     # outer, inner
RIGHT_EYE_CORNERS = (362, 263)   # inner, outer


@dataclass
class GazeFeatures:
    vector: np.ndarray   # shape (N,) — model input
    confidence: float    # 0..1


class GazeTracker:
    """Wraps MediaPipe FaceMesh and yields per-frame :class:`GazeFeatures`."""

    def __init__(self) -> None:
        if _mp_face_mesh is None:
            raise RuntimeError("mediapipe is required for GazeTracker")
        self._mesh = _mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,         # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> GazeFeatures | None:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h])

        # Iris centroids
        left_iris = np.mean([pt(i) for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)

        # Normalise iris position by eye width (per-eye)
        l_outer, l_inner = pt(LEFT_EYE_CORNERS[0]), pt(LEFT_EYE_CORNERS[1])
        r_inner, r_outer = pt(RIGHT_EYE_CORNERS[0]), pt(RIGHT_EYE_CORNERS[1])
        l_w = max(np.linalg.norm(l_outer - l_inner), 1e-6)
        r_w = max(np.linalg.norm(r_outer - r_inner), 1e-6)

        l_rel = (left_iris - l_outer) / l_w     # 2-vector
        r_rel = (right_iris - r_inner) / r_w    # 2-vector

        # Head pose proxy: nose tip relative to inter-eye midpoint
        nose = pt(1)
        eye_mid = (l_outer + r_outer) / 2.0
        head = (nose - eye_mid) / max(np.linalg.norm(l_outer - r_outer), 1e-6)

        feats = np.concatenate([l_rel, r_rel, head])  # 6-d
        return GazeFeatures(vector=feats, confidence=1.0)

    def close(self) -> None:
        self._mesh.close()
