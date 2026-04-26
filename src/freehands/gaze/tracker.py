"""Eye-feature extraction with MediaPipe FaceMesh.

We extract a compact feature vector per frame that the personalised ridge
regression in :mod:`freehands.gaze.calibration` maps to screen coordinates.
The features intentionally avoid raw pixel coordinates so the model
generalises across small head movements.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mediapipe_assets import ensure_model

try:
    import mediapipe as mp
    _mp_solutions = getattr(mp, "solutions", None)
    _mp_face_mesh = _mp_solutions.face_mesh if _mp_solutions is not None else None
    _mp_error: Exception | None = None
except Exception as exc:  # pragma: no cover
    mp = None
    _mp_face_mesh = None
    _mp_error = exc

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _mp_tasks_error: Exception | None = None
except Exception as exc:  # pragma: no cover
    mp_tasks = None
    mp_vision = None
    _mp_tasks_error = exc


# MediaPipe FaceMesh iris and eye-corner landmarks
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (33, 133)     # outer, inner
RIGHT_EYE_CORNERS = (362, 263)   # inner, outer


@dataclass
class GazeFeatures:
    vector: np.ndarray   # shape (N,) — model input
    confidence: float    # 0..1


@dataclass
class GazeDebug:
    backend: str
    face_detected: bool = False
    landmark_count: int = 0
    iris_detected: bool = False
    confidence: float = 0.0
    message: str = "Inicializando"
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


class GazeTracker:
    """Wraps MediaPipe FaceMesh and yields per-frame :class:`GazeFeatures`."""

    def __init__(self) -> None:
        self._backend = "solutions" if _mp_face_mesh is not None else "tasks"
        self.last_debug = GazeDebug(backend=self._backend)
        if _mp_face_mesh is not None:
            self._mesh = _mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,         # enables iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if mp is None or mp_tasks is None or mp_vision is None:
            detail = f": {_mp_error or _mp_tasks_error}" if (_mp_error or _mp_tasks_error) else ""
            raise RuntimeError(
                "MediaPipe FaceMesh is required for GazeTracker" + detail +
                ". Run: FreeHands.bat repair"
            )

        model_path = ensure_model("face_landmarker")
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mesh = mp_vision.FaceLandmarker.create_from_options(options)

    def extract(self, frame_bgr: np.ndarray) -> GazeFeatures | None:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        debug = GazeDebug(backend=self._backend, message="Procesando frame")
        if self._backend == "solutions":
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                debug.message = "No se detecta cara"
                self.last_debug = debug
                return None
            lm = result.multi_face_landmarks[0].landmark
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mesh.detect(image)
            if not result.face_landmarks:
                debug.message = "No se detecta cara"
                self.last_debug = debug
                return None
            lm = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        debug.face_detected = True
        debug.landmark_count = len(lm)

        if len(lm) <= max(RIGHT_EYE_CORNERS):
            debug.message = f"Cara detectada, pero faltan landmarks de ojos ({len(lm)})"
            self.last_debug = debug
            return None

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h])

        # Normalise iris position by eye width (per-eye)
        l_outer, l_inner = pt(LEFT_EYE_CORNERS[0]), pt(LEFT_EYE_CORNERS[1])
        r_inner, r_outer = pt(RIGHT_EYE_CORNERS[0]), pt(RIGHT_EYE_CORNERS[1])
        l_w = max(np.linalg.norm(l_outer - l_inner), 1e-6)
        r_w = max(np.linalg.norm(r_outer - r_inner), 1e-6)

        if len(lm) > max(RIGHT_IRIS):
            left_iris = np.mean([pt(i) for i in LEFT_IRIS], axis=0)
            right_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)
            confidence = 1.0
            debug.iris_detected = True
        else:
            left_iris = (l_outer + l_inner) / 2.0
            right_iris = (r_inner + r_outer) / 2.0
            confidence = 0.65

        l_rel = (left_iris - l_outer) / l_w     # 2-vector
        r_rel = (right_iris - r_inner) / r_w    # 2-vector

        # Head pose proxy: nose tip relative to inter-eye midpoint
        nose = pt(1)
        eye_mid = (l_outer + r_outer) / 2.0
        head = (nose - eye_mid) / max(np.linalg.norm(l_outer - r_outer), 1e-6)

        feats = np.concatenate([l_rel, r_rel, head])  # 6-d
        debug.confidence = confidence
        debug.message = "Ojos e iris detectados" if debug.iris_detected else "Ojos detectados sin iris fino"
        debug.points = {
            "left_outer": tuple(l_outer),
            "left_inner": tuple(l_inner),
            "right_inner": tuple(r_inner),
            "right_outer": tuple(r_outer),
            "left_iris": tuple(left_iris),
            "right_iris": tuple(right_iris),
            "nose": tuple(nose),
        }
        debug.vector = [round(float(v), 3) for v in feats]
        self.last_debug = debug
        return GazeFeatures(vector=feats, confidence=confidence)

    def close(self) -> None:
        self._mesh.close()
