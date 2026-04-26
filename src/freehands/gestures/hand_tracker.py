"""MediaPipe-based hand-gesture detector.

Phase 1 ships a minimal rule-based classifier on top of the 21 hand landmarks:

    * ``thumb_up``   — thumb extended upward, other fingers folded
    * ``thumb_down`` — thumb extended downward, other fingers folded
    * ``pinch``      — index tip close to thumb tip (distance < threshold)
    * ``fist``       — all five fingers folded (used as the always-on pause gesture)

This stays inside the project's "no false positives" rule: rules are coarse but
deterministic; the multi-frame :class:`~freehands.gestures.stabilizer.GestureStabilizer`
is what actually fires actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..mediapipe_assets import ensure_model

try:
    import mediapipe as mp
    _mp_solutions = getattr(mp, "solutions", None)
    _mp_hands = _mp_solutions.hands if _mp_solutions is not None else None
    _mp_error: Exception | None = None
except Exception as exc:  # pragma: no cover
    mp = None
    _mp_hands = None
    _mp_error = exc

try:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _mp_tasks_error: Exception | None = None
except Exception as exc:  # pragma: no cover
    mp_tasks = None
    mp_vision = None
    _mp_tasks_error = exc


GestureId = Literal["thumb_up", "thumb_down", "pinch_open", "pinch_close",
                    "fist_pause", "open_palm", "none"]

# Landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18


@dataclass
class HandObservation:
    gesture: GestureId
    confidence: float
    landmarks: np.ndarray | None  # shape (21, 3) normalised


class HandTracker:
    def __init__(self) -> None:
        self._backend = "solutions" if _mp_hands is not None else "tasks"
        if _mp_hands is not None:
            self._hands = _mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        elif mp is not None and mp_tasks is not None and mp_vision is not None:
            model_path = ensure_model("gesture_recognizer")
            options = mp_vision.GestureRecognizerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = mp_vision.GestureRecognizer.create_from_options(options)
        else:
            detail = f": {_mp_error or _mp_tasks_error}" if (_mp_error or _mp_tasks_error) else ""
            raise RuntimeError("MediaPipe Hands/GestureRecognizer is required" + detail)
        self._prev_pinch_dist: float | None = None

    def detect(self, frame_bgr: np.ndarray) -> HandObservation:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._hands.process(rgb)
            if not result.multi_hand_landmarks:
                self._prev_pinch_dist = None
                return HandObservation("none", 0.0, None)
            lm = result.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm])
            gesture, conf = self._classify(pts)
            return HandObservation(gesture, conf, pts)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._hands.recognize(image)
        if not result.hand_landmarks:
            self._prev_pinch_dist = None
            return HandObservation("none", 0.0, None)

        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
        gesture, conf = self._classify(pts)
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            mapped = self._map_task_gesture(top.category_name)
            if mapped != "none" and float(top.score) >= 0.40:
                gesture, conf = mapped, float(top.score)
        return HandObservation(gesture, conf, pts)

    def _map_task_gesture(self, category: str) -> GestureId:
        return {
            "Thumb_Up": "thumb_up",
            "Thumb_Down": "thumb_down",
            "Open_Palm": "open_palm",
            "Closed_Fist": "fist_pause",
        }.get(category, "none")

    # ── classification ────────────────────────────────────────────────────
    def _classify(self, pts: np.ndarray) -> tuple[GestureId, float]:
        # Finger "extended" tests use y comparison vs PIP joint (image space)
        index_up   = pts[INDEX_TIP, 1]  < pts[INDEX_PIP, 1]
        middle_up  = pts[MIDDLE_TIP, 1] < pts[MIDDLE_PIP, 1]
        ring_up    = pts[RING_TIP, 1]   < pts[RING_PIP, 1]
        pinky_up   = pts[PINKY_TIP, 1]  < pts[PINKY_PIP, 1]

        thumb_up_dir   = pts[THUMB_TIP, 1] < pts[THUMB_IP, 1] - 0.02
        thumb_down_dir = pts[THUMB_TIP, 1] > pts[THUMB_IP, 1] + 0.02

        folded_others = not any([index_up, middle_up, ring_up, pinky_up])
        all_folded = folded_others and abs(pts[THUMB_TIP, 0] - pts[WRIST, 0]) < 0.12

        if all_folded:
            return "fist_pause", 0.95

        if folded_others and thumb_up_dir:
            return "thumb_up", 0.92
        if folded_others and thumb_down_dir:
            return "thumb_down", 0.92

        # Pinch: distance index_tip ↔ thumb_tip
        d = float(np.linalg.norm(pts[INDEX_TIP, :2] - pts[THUMB_TIP, :2]))
        gesture: GestureId = "none"
        conf = 0.0
        if self._prev_pinch_dist is not None:
            delta = d - self._prev_pinch_dist
            if d < 0.05:
                gesture, conf = "pinch_close", 0.90
            elif delta > 0.02 and d > 0.10:
                gesture, conf = "pinch_open", 0.85
        self._prev_pinch_dist = d

        if gesture == "none" and all([index_up, middle_up, ring_up, pinky_up]):
            return "open_palm", 0.85

        return gesture, conf

    def close(self) -> None:
        self._hands.close()
