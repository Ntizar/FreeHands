"""MediaPipe-based hand-gesture detector.

Phase 1 ships a minimal rule-based classifier on top of the 21 hand landmarks:

    * ``thumb_up``   — thumb extended upward, other fingers folded
    * ``thumb_down`` — thumb extended downward, other fingers folded
    * ``pinch``      — index tip close to thumb tip (distance < threshold)
    * ``right_open_palm`` — hand fully open (always-on pause gesture)

This stays inside the project's "no false positives" rule: rules are coarse but
deterministic; the multi-frame :class:`~freehands.gestures.stabilizer.GestureStabilizer`
is what actually fires actions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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


GestureId = Literal[
    "thumb_up", "thumb_down", "pointing_up", "middle_up", "two_fingers_up",
    "pinch_open", "pinch_close", "two_hands_together", "two_hands_apart",
    "fist_pause", "open_palm", "left_open_palm", "right_open_palm",
    "left_pointing_up", "right_pointing_up", "left_middle_up", "right_middle_up",
    "left_two_fingers_up", "right_two_fingers_up", "none",
]

SIDE_AWARE_GESTURES = {"pointing_up", "middle_up", "two_fingers_up", "open_palm"}

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
    hands: list[np.ndarray] = field(default_factory=list)
    handedness: list[str] = field(default_factory=list)


class HandTracker:
    def __init__(self) -> None:
        self._swap_handedness = False
        self._backend = "solutions" if _mp_hands is not None else "tasks"
        if _mp_hands is not None:
            self._hands = _mp_hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        elif mp is not None and mp_tasks is not None and mp_vision is not None:
            model_path = ensure_model("gesture_recognizer")
            options = mp_vision.GestureRecognizerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = mp_vision.GestureRecognizer.create_from_options(options)
        else:
            detail = f": {_mp_error or _mp_tasks_error}" if (_mp_error or _mp_tasks_error) else ""
            raise RuntimeError("MediaPipe Hands/GestureRecognizer is required" + detail)
        self._prev_pinch_dist: float | None = None

    def set_handedness_swapped(self, enabled: bool) -> None:
        self._swap_handedness = enabled

    def detect(self, frame_bgr: np.ndarray) -> HandObservation:
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "solutions":
            result = self._hands.process(rgb)
            if not result.multi_hand_landmarks:
                self._prev_pinch_dist = None
                return HandObservation("none", 0.0, None, [])
            hands = [np.array([[p.x, p.y, p.z] for p in item.landmark]) for item in result.multi_hand_landmarks]
            handedness = self._normalize_handedness(self._solution_handedness(result.multi_handedness))
            gesture, conf = self._classify_multi(hands, handedness)
            return HandObservation(gesture, conf, hands[0], hands, handedness)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._hands.recognize(image)
        if not result.hand_landmarks:
            self._prev_pinch_dist = None
            return HandObservation("none", 0.0, None, [])

        hands = [np.array([[p.x, p.y, p.z] for p in item]) for item in result.hand_landmarks]
        handedness = self._normalize_handedness(self._task_handedness(result.handedness))
        gesture, conf = self._classify_multi(hands, handedness)
        if result.gestures and any(result.gestures):
            mapped = self._task_gesture(result.gestures, handedness)
            if gesture not in {"two_hands_together", "two_hands_apart"} and mapped[0] != "none":
                gesture, conf = mapped
        return HandObservation(gesture, conf, hands[0], hands, handedness)

    def _map_task_gesture(self, category: str, handedness: str = "") -> GestureId:
        mapped: GestureId = {
            "Thumb_Up": "thumb_up",
            "Thumb_Down": "thumb_down",
            "Pointing_Up": "pointing_up",
            "Victory": "two_fingers_up",
            "Open_Palm": "open_palm",
            "Closed_Fist": "fist_pause",
        }.get(category, "none")
        return self._with_side(mapped, handedness)

    def _task_gesture(self, gestures: list[list[object]], handedness: list[str]) -> tuple[GestureId, float]:
        best: tuple[GestureId, float] = ("none", 0.0)
        for index, candidates in enumerate(gestures):
            if not candidates:
                continue
            top = candidates[0]
            score = float(top.score)
            if score < 0.40 or score <= best[1]:
                continue
            side = handedness[index] if index < len(handedness) else ""
            best = self._map_task_gesture(top.category_name, side), score
        return best

    def _classify_multi(self, hands: list[np.ndarray], handedness: list[str] | None = None) -> tuple[GestureId, float]:
        handedness = handedness or []
        if len(hands) >= 2:
            left, right = hands[0], hands[1]
            center_a = np.mean(left[:, :2], axis=0)
            center_b = np.mean(right[:, :2], axis=0)
            dist = float(np.linalg.norm(center_a - center_b))
            vertical_gap = abs(float(center_a[1] - center_b[1]))
            if vertical_gap < 0.28 and dist < 0.24:
                return "two_hands_together", 0.92
            if vertical_gap < 0.36 and dist > 0.48:
                return "two_hands_apart", 0.90
            classified = [
                self._classify(hand, handedness[index] if index < len(handedness) else "")
                for index, hand in enumerate(hands)
            ]
            for gesture, conf in classified:
                if gesture in {"left_open_palm", "right_open_palm"}:
                    return gesture, conf
            for gesture, conf in classified:
                if gesture != "none":
                    return gesture, conf
            return "none", 0.0
        return self._classify(hands[0], handedness[0] if handedness else "")

    @staticmethod
    def _solution_handedness(items: list[object] | None) -> list[str]:
        if not items:
            return []
        labels: list[str] = []
        for item in items:
            classifications = getattr(item, "classification", [])
            labels.append(classifications[0].label if classifications else "")
        return labels

    @staticmethod
    def _task_handedness(items: list[list[object]] | None) -> list[str]:
        if not items:
            return []
        labels: list[str] = []
        for item in items:
            labels.append(item[0].category_name if item else "")
        return labels

    def _normalize_handedness(self, labels: list[str]) -> list[str]:
        if not self._swap_handedness:
            return labels
        return [self._swap_label(label) for label in labels]

    @staticmethod
    def _swap_label(label: str) -> str:
        if label == "Left":
            return "Right"
        if label == "Right":
            return "Left"
        return label

    # ── classification ────────────────────────────────────────────────────
    def _classify(self, pts: np.ndarray, handedness: str = "") -> tuple[GestureId, float]:
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

        if index_up and middle_up and not any([ring_up, pinky_up]):
            return self._with_side("two_fingers_up", handedness), 0.90
        if index_up and not any([middle_up, ring_up, pinky_up]):
            return self._with_side("pointing_up", handedness), 0.90
        if middle_up and not any([index_up, ring_up, pinky_up]):
            return self._with_side("middle_up", handedness), 0.88

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
            return self._with_side("open_palm", handedness), 0.90

        return gesture, conf

    @staticmethod
    def _with_side(gesture: GestureId, handedness: str = "") -> GestureId:
        if gesture not in SIDE_AWARE_GESTURES:
            return gesture
        if handedness == "Left":
            return f"left_{gesture}"  # type: ignore[return-value]
        if handedness == "Right":
            return f"right_{gesture}"  # type: ignore[return-value]
        if gesture == "open_palm":
            return "right_open_palm"
        return gesture

    def close(self) -> None:
        self._hands.close()
