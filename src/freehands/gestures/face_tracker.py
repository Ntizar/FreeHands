"""MediaPipe-based facial expression gesture detector.

Detects facial expressions using MediaPipe FaceMesh landmarks:

    * ``smile``          — mouth corners pulled back/up (happy)
    * ``frown``          — mouth corners pulled down (sad/angry)
    * ``surprise``       — mouth open wide + eyebrows raised
    * ``raised_eyebrows`` — both eyebrows raised independently
    * ``furrowed_brows``  — eyebrows drawn together and down (anger/focus)
    * ``mouth_open``      — mouth vertically open (breathing, speech)
    * ``tongue_out``      — tongue landmark visible below lip line (legacy)

Uses geometric ratios on mouth and eyebrow landmarks from the 468-point
FaceMesh set.  All detections are deterministic and rule-based — no ML
model needed beyond FaceMesh itself.

The output is consumed by the fusion layer as instant-action gestures
(same pattern as palm-scroll / air-scroll): no dwell needed, fire
immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

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


# ── MediaPipe FaceMesh landmark indices (468 points) ────────────────────────
# Mouth contour (68 points: 0-67)
# Upper lip: 61-52 (outer→inner), 51-45 (inner→outer), 44-28 (inner→outer), 27-18 (inner→outer)
# Lower lip: 0-17 (outer→inner), 18-27 (inner→outer), 28-44 (inner→outer), 45-51 (inner→outer)
# Simplified: use key landmarks
MOUTH_OUTER_LEFT  = 61   # left mouth corner
MOUTH_OUTER_RIGHT = 291  # right mouth corner
MOUTH_UPPER_CENTER = 13  # upper lip center top
MOUTH_LOWER_CENTER = 14  # lower lip center bottom
MOUTH_UPPER_LIP_37  = 37   # upper lip inner top
MOUTH_LOWER_LIP_177 = 177  # lower lip inner bottom

# Eyebrow landmarks
LEFT_EYEBROW_TIP    = 6   # left eyebrow outer tip
LEFT_EYEBROW_INNER  = 70   # left eyebrow inner (near nose)
LEFT_EYEBROW_CENTER = 55   # left eyebrow center
RIGHT_EYEBROW_TIP   = 300  # right eyebrow outer tip
RIGHT_EYEBROW_INNER = 286  # right eyebrow inner (near nose)
RIGHT_EYEBROW_CENTER = 338  # right eyebrow center (not 291 — that's MOUTH_OUTER_RIGHT)

# Eyelid landmarks (for blink detection as part of expression context)
LEFT_EYE_UPPER = 159
LEFT_EYE_LOWER = 145
RIGHT_EYE_UPPER = 386
RIGHT_EYE_LOWER = 374

# Tongue landmark (legacy)
TONGUE = 15  # chin area where tongue extends


# ── Gesture type definitions ─────────────────────────────────────────────────
# Facial expressions are state-based (hold required), not instant like gestures.
# They map to actions via bindings like any other gesture.

FacialGestureId = Literal[
    "smile", "frown", "surprise", "raised_eyebrows", "furrowed_brows",
    "mouth_open", "tongue_out", "none",
]


# ── Detection thresholds ─────────────────────────────────────────────────────
# Mouth openness ratio (vertical / horizontal mouth width) for smile/frown/surprise
MOUTH_OPEN_THRESHOLD = 0.15       # ratio above which mouth is considered "open"
SMILE_CORNER_RATIO   = 0.03       # corner Y shift relative to mouth width
FROWN_CORNER_RATIO   = -0.02      # negative = corners pulled down
SURPRISE_MOUTH_RATIO = 0.25       # mouth very wide open
SURPRISE_EYEBROW_GAP = 0.02       # eyebrow raised above eye line

# Eyebrow movement thresholds (relative to eye level)
EYEBROW_RAISE_THRESHOLD = 0.015   # eyebrow moved up relative to eye
EYEBROW_FURROW_THRESHOLD = 0.015  # eyebrow moved down relative to eye

# Mouth open detection
MOUTH_OPEN_RATIO = 0.10           # vertical opening ratio for "mouth open"
TONGUE_PROTRUSION = 0.04          # chin-to-tongue distance for tongue-out


@dataclass
class FacialObservation:
    """Result of facial expression detection for a single frame."""
    smile: bool = False
    frown: bool = False
    surprise: bool = False
    raised_eyebrows: bool = False
    furrowed_brows: bool = False
    mouth_open: bool = False
    tongue_out: bool = False
    primary_gesture: FacialGestureId = "none"
    confidence: float = 0.0
    landmarks: np.ndarray | None = None  # shape (468, 3) normalised


class FaceTracker:
    """Detects facial expressions from MediaPipe FaceMesh landmarks.

    Uses geometric ratios on mouth and eyebrow landmarks.  Each expression
    is detected independently — multiple expressions can be active at once
    (e.g. surprise + mouth_open).  The primary_gesture field returns the
    highest-priority expression for use in the fusion layer.

    Priority order (highest first): surprise > smile > frown >
    raised_eyebrows > furrowed_brows > mouth_open > tongue_out
    """

    def __init__(self) -> None:
        self._backend = "solutions" if _mp_face_mesh is not None else "tasks"
        if _mp_face_mesh is not None:
            self._mesh = _mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False,
            )
        elif mp is not None and mp_tasks is not None and mp_vision is not None:
            from ..mediapipe_assets import ensure_model
            model_path = ensure_model("face_landmarker")
            options = mp_vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mesh = mp_vision.FaceLandmarker.create_from_options(options)
        else:
            detail = f": {_mp_error or _mp_tasks_error}" if (_mp_error or _mp_tasks_error) else ""
            raise RuntimeError("MediaPipe FaceMesh is required for FaceTracker" + detail)

    def detect(self, frame_bgr: np.ndarray) -> FacialObservation:
        """Run facial expression detection on a single frame.

        Parameters
        ----------
        frame_bgr :
            BGR image from the camera (same format as hand_tracker expects).

        Returns
        -------
        FacialObservation
            Detected expressions and primary gesture.
        """
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        if self._backend == "solutions":
            result = self._mesh.process(rgb)
            if not result.multi_face_landmarks:
                return FacialObservation()
            lm = result.multi_face_landmarks[0].landmark
        else:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._mesh.detect(image)
            if not result.face_landmarks:
                return FacialObservation()
            lm = result.face_landmarks[0]

        # Convert to numpy array for easier access
        landmarks = np.array([[p.x, p.y, p.z] for p in lm])

        # Check landmark availability
        required = [MOUTH_OUTER_LEFT, MOUTH_OUTER_RIGHT, MOUTH_UPPER_CENTER,
                     MOUTH_LOWER_CENTER, LEFT_EYEBROW_TIP, RIGHT_EYEBROW_TIP,
                     LEFT_EYE_UPPER, RIGHT_EYE_UPPER]
        if not all(idx < len(lm) for idx in required):
            return FacialObservation()

        # ── Compute geometric features ───────────────────────────────────
        # Mouth width (horizontal distance between corners)
        mouth_left = landmarks[MOUTH_OUTER_LEFT]
        mouth_right = landmarks[MOUTH_OUTER_RIGHT]
        mouth_width = abs(mouth_right[0] - mouth_left[0])
        if mouth_width < 1e-6:
            mouth_width = 1e-6

        # Mouth height (vertical distance between upper and lower lip center)
        mouth_upper = landmarks[MOUTH_UPPER_CENTER]
        mouth_lower = landmarks[MOUTH_LOWER_CENTER]
        mouth_height = abs(mouth_lower[1] - mouth_upper[1])

        # Mouth openness ratio
        mouth_open_ratio = mouth_height / mouth_width

        # Mouth corner Y positions relative to mouth center Y
        mouth_center_y = (mouth_upper[1] + mouth_lower[1]) / 2.0
        left_corner_y = mouth_left[1]
        right_corner_y = mouth_right[1]
        corner_y_avg = (left_corner_y + right_corner_y) / 2.0
        corner_shift = (corner_y_avg - mouth_center_y) / mouth_width

        # Eyebrow positions relative to eye positions
        left_eye_upper = landmarks[LEFT_EYE_UPPER]
        right_eye_upper = landmarks[RIGHT_EYE_UPPER]
        left_eyebrow_tip = landmarks[LEFT_EYEBROW_TIP]
        right_eyebrow_tip = landmarks[RIGHT_EYEBROW_TIP]
        left_eyebrow_center = landmarks[LEFT_EYEBROW_CENTER]
        right_eyebrow_center = landmarks[RIGHT_EYEBROW_CENTER]

        # Eyebrow raise: how far above the eye line
        left_eyebrow_raise = (left_eye_upper[1] - left_eyebrow_center[1]) / mouth_width
        right_eyebrow_raise = (right_eye_upper[1] - right_eyebrow_center[1]) / mouth_width

        # Eyebrow furrow: how far the inner corners are pulled down
        left_eyebrow_inner = landmarks[LEFT_EYEBROW_INNER]
        right_eyebrow_inner = landmarks[RIGHT_EYEBROW_INNER]
        left_eyebrow_furrow = (left_eyebrow_inner[1] - left_eye_upper[1]) / mouth_width
        right_eyebrow_furrow = (right_eyebrow_inner[1] - right_eye_upper[1]) / mouth_width

        # ── Detect expressions ───────────────────────────────────────────
        smile = False
        frown = False
        surprise = False
        raised_eyebrows = False
        furrowed_brows = False
        mouth_open = False
        tongue_out = False

        # Smile: mouth corners pulled UP relative to center (negative shift in image coords)
        # and mouth is somewhat open (not a tight line)
        if -SMILE_CORNER_RATIO < corner_shift < 0 and mouth_open_ratio > 0.05:
            smile = True

        # Frown: mouth corners pulled DOWN (positive shift in image coords)
        if corner_shift > abs(FROWN_CORNER_RATIO) and mouth_open_ratio > 0.05:
            frown = True

        # Surprise: mouth very open + eyebrows raised
        if mouth_open_ratio > SURPRISE_MOUTH_RATIO and \
           left_eyebrow_raise > SURPRISE_EYEBROW_GAP and \
           right_eyebrow_raise > SURPRISE_EYEBROW_GAP:
            surprise = True

        # Raised eyebrows: either eyebrow significantly above eye line
        if left_eyebrow_raise > EYEBROW_RAISE_THRESHOLD or \
           right_eyebrow_raise > EYEBROW_RAISE_THRESHOLD:
            raised_eyebrows = True

        # Furrowed brows: inner corners pulled down (negative = toward eye)
        if left_eyebrow_furrow < -EYEBROW_FURROW_THRESHOLD or \
           right_eyebrow_furrow < -EYEBROW_FURROW_THRESHOLD:
            furrowed_brows = True

        # Mouth open: mouth height exceeds threshold ratio
        if mouth_open_ratio > MOUTH_OPEN_RATIO:
            mouth_open = True

        # Tongue out: chin landmark (15) is below mouth lower center
        # This is a legacy/weak signal — tongue tip is typically landmark 15 or near chin
        tongue = landmarks[TONGUE] if TONGUE < len(landmarks) else None
        if tongue is not None:
            tongue_below_mouth = tongue[1] > mouth_lower[1] + TONGUE_PROTRUSION
            if tongue_below_mouth:
                tongue_out = True

        # ── Determine primary gesture (priority order) ───────────────────
        primary_gesture: FacialGestureId = "none"
        confidence = 0.0

        if surprise:
            primary_gesture = "surprise"
            confidence = 0.90
        elif smile:
            primary_gesture = "smile"
            confidence = 0.85
        elif frown:
            primary_gesture = "frown"
            confidence = 0.85
        elif raised_eyebrows:
            primary_gesture = "raised_eyebrows"
            confidence = 0.80
        elif furrowed_brows:
            primary_gesture = "furrowed_brows"
            confidence = 0.80
        elif mouth_open:
            primary_gesture = "mouth_open"
            confidence = 0.75
        elif tongue_out:
            primary_gesture = "tongue_out"
            confidence = 0.70

        return FacialObservation(
            smile=smile,
            frown=frown,
            surprise=surprise,
            raised_eyebrows=raised_eyebrows,
            furrowed_brows=furrowed_brows,
            mouth_open=mouth_open,
            tongue_out=tongue_out,
            primary_gesture=primary_gesture,
            confidence=confidence,
            landmarks=landmarks,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._mesh.close()
