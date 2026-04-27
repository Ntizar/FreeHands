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
LEFT_EYE_VERTICAL = (159, 145)   # upper, lower eyelid
RIGHT_EYE_VERTICAL = (386, 374)  # upper, lower eyelid
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_CONTOUR = [362, 385, 387, 263, 373, 380]
REQUIRED_EYE_LANDMARK = max([
    *LEFT_EYE_CORNERS,
    *RIGHT_EYE_CORNERS,
    *LEFT_EYE_VERTICAL,
    *RIGHT_EYE_VERTICAL,
    *LEFT_EYE_CONTOUR,
    *RIGHT_EYE_CONTOUR,
])


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
    pupil_detected: bool = False
    confidence: float = 0.0
    message: str = "Inicializando"
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    vector: list[float] = field(default_factory=list)


def _detect_dark_pupil(gray: np.ndarray, eye_points: list[np.ndarray]) -> np.ndarray | None:
    import cv2

    polygon = np.array(eye_points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(polygon)
    if w < 8 or h < 5:
        return None

    margin = max(4, int(max(w, h) * 0.22))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray.shape[1], x + w + margin)
    y1 = min(gray.shape[0], y + h + margin)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [local_polygon], 255)
    valid = roi[mask > 0]
    if valid.size < 30:
        return None

    filtered = cv2.bilateralFilter(roi, 5, 20, 20)
    threshold = int(min(np.percentile(valid, 30), float(valid.min()) + 45.0))
    pupil_mask = cv2.inRange(filtered, 0, threshold)
    pupil_mask = cv2.bitwise_and(pupil_mask, mask)
    kernel = np.ones((3, 3), np.uint8)
    pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    mask_area = max(float(cv2.countNonZero(mask)), 1.0)
    eye_center = np.array([roi.shape[1] / 2.0, roi.shape[0] / 2.0])
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 3.0 or area > mask_area * 0.55:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        distance_penalty = float(np.linalg.norm(center - eye_center)) * 0.12
        candidates.append((area - distance_penalty, center + np.array([x0, y0])))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        debug = GazeDebug(backend=self._backend, message="Procesando frame")
        if self._backend == "solutions":
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                debug.message = "No face detected"
                self.last_debug = debug
                return None
            lm = result.multi_face_landmarks[0].landmark
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mesh.detect(image)
            if not result.face_landmarks:
                debug.message = "No face detected"
                self.last_debug = debug
                return None
            lm = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        debug.face_detected = True
        debug.landmark_count = len(lm)

        if len(lm) <= REQUIRED_EYE_LANDMARK:
            debug.message = f"Face detected, but eye landmarks are missing ({len(lm)})"
            self.last_debug = debug
            return None

        def pt(i: int) -> np.ndarray:
            return np.array([lm[i].x * w, lm[i].y * h])

        # Normalise iris position by eye width (per-eye)
        l_outer, l_inner = pt(LEFT_EYE_CORNERS[0]), pt(LEFT_EYE_CORNERS[1])
        r_inner, r_outer = pt(RIGHT_EYE_CORNERS[0]), pt(RIGHT_EYE_CORNERS[1])
        l_w = max(np.linalg.norm(l_outer - l_inner), 1e-6)
        r_w = max(np.linalg.norm(r_outer - r_inner), 1e-6)
        l_top, l_bottom = pt(LEFT_EYE_VERTICAL[0]), pt(LEFT_EYE_VERTICAL[1])
        r_top, r_bottom = pt(RIGHT_EYE_VERTICAL[0]), pt(RIGHT_EYE_VERTICAL[1])
        l_h = max(abs(l_bottom[1] - l_top[1]), 1e-6)
        r_h = max(abs(r_bottom[1] - r_top[1]), 1e-6)

        if len(lm) > max(RIGHT_IRIS):
            left_iris = np.mean([pt(i) for i in LEFT_IRIS], axis=0)
            right_iris = np.mean([pt(i) for i in RIGHT_IRIS], axis=0)
            confidence = 1.0
            debug.iris_detected = True
        else:
            left_iris = (l_outer + l_inner) / 2.0
            right_iris = (r_inner + r_outer) / 2.0
            confidence = 0.65

        left_pupil = _detect_dark_pupil(gray, [pt(i) for i in LEFT_EYE_CONTOUR])
        right_pupil = _detect_dark_pupil(gray, [pt(i) for i in RIGHT_EYE_CONTOUR])
        debug.pupil_detected = left_pupil is not None or right_pupil is not None
        left_signal = left_pupil if left_pupil is not None else left_iris
        right_signal = right_pupil if right_pupil is not None else right_iris
        if left_pupil is not None and right_pupil is not None:
            confidence = 1.0

        l_rel = np.array([(left_signal[0] - l_outer[0]) / l_w, (left_signal[1] - l_top[1]) / l_h])
        r_rel = np.array([(right_signal[0] - r_inner[0]) / r_w, (right_signal[1] - r_top[1]) / r_h])

        # Head pose proxy: nose tip relative to inter-eye midpoint
        nose = pt(1)
        eye_mid = (l_outer + r_outer) / 2.0
        head = (nose - eye_mid) / max(np.linalg.norm(l_outer - r_outer), 1e-6)

        feats = np.concatenate([l_rel, r_rel, head])  # 6-d
        debug.confidence = confidence
        if debug.pupil_detected:
            debug.message = "Dark pupil detected"
        else:
            debug.message = "Eyes and iris detected" if debug.iris_detected else "Eyes detected without fine iris"
        debug.points = {
            "left_outer": tuple(l_outer),
            "left_inner": tuple(l_inner),
            "right_inner": tuple(r_inner),
            "right_outer": tuple(r_outer),
            "left_iris": tuple(left_signal),
            "right_iris": tuple(right_signal),
            "nose": tuple(nose),
        }
        if left_pupil is not None:
            debug.points["left_pupil"] = tuple(left_pupil)
        if right_pupil is not None:
            debug.points["right_pupil"] = tuple(right_pupil)
        debug.vector = [round(float(v), 3) for v in feats]
        self.last_debug = debug
        return GazeFeatures(vector=feats, confidence=confidence)

    def close(self) -> None:
        self._mesh.close()
