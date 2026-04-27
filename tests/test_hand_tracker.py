"""Unit tests for low-level hand gesture mapping."""
from __future__ import annotations

import numpy as np

from freehands.gestures.hand_tracker import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    HandTracker,
)


def test_mediapipe_open_palm_maps_to_pause_gesture() -> None:
    tracker = HandTracker.__new__(HandTracker)

    assert tracker._map_task_gesture("Open_Palm") == "right_open_palm"


def test_rule_based_open_palm_maps_to_pause_gesture() -> None:
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    points = np.zeros((21, 3), dtype=float)
    for tip, pip in [
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]:
        points[tip, 1] = 0.2
        points[pip, 1] = 0.5

    gesture, confidence = tracker._classify(points)

    assert gesture == "right_open_palm"
    assert confidence >= 0.80
