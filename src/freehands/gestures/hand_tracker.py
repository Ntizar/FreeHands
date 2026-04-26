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

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
except Exception:  # pragma: no cover
    mp = None
    _mp_hands = None


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
        if _mp_hands is None:
            raise RuntimeError("mediapipe is required for HandTracker")
        self._hands = _mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._prev_pinch_dist: float | None = None

    def detect(self, frame_bgr: np.ndarray) -> HandObservation:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            self._prev_pinch_dist = None
            return HandObservation("none", 0.0, None)

        lm = result.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm])

        gesture, conf = self._classify(pts)
        return HandObservation(gesture, conf, pts)

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
