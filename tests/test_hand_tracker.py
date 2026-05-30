"""Unit tests for low-level hand gesture mapping."""
from __future__ import annotations

from types import SimpleNamespace

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


def _points_for_fingers(
    *,
    index_delta: float = -0.1,
    middle_delta: float = -0.1,
    ring_delta: float = -0.1,
    pinky_delta: float = -0.1,
) -> np.ndarray:
    points = np.zeros((21, 3), dtype=float)
    for tip, pip, delta in [
        (INDEX_TIP, INDEX_PIP, index_delta),
        (MIDDLE_TIP, MIDDLE_PIP, middle_delta),
        (RING_TIP, RING_PIP, ring_delta),
        (PINKY_TIP, PINKY_PIP, pinky_delta),
    ]:
        points[pip, 1] = 0.5
        points[tip, 1] = 0.5 - delta
    return points


def test_mediapipe_open_palm_maps_to_pause_gesture() -> None:
    tracker = HandTracker.__new__(HandTracker)

    assert tracker._map_task_gesture("Open_Palm") == "open_palm"
    assert tracker._map_task_gesture("Open_Palm", "Left") == "left_open_palm"
    assert tracker._map_task_gesture("Open_Palm", "Right") == "right_open_palm"
    assert tracker._map_task_gesture("Pointing_Up", "Left") == "left_pointing_up"
    assert tracker._map_task_gesture("Pointing_Up", "Right") == "right_pointing_up"


def test_task_gesture_uses_best_hand_candidate() -> None:
    tracker = HandTracker.__new__(HandTracker)
    gestures = [
        [],
        [SimpleNamespace(category_name="Pointing_Up", score=0.92)],
    ]

    gesture, confidence = tracker._task_gesture(gestures, ["Left", "Right"])

    assert gesture == "right_pointing_up"
    assert confidence == 0.92


def test_handedness_swap_can_correct_camera_left_right_confusion() -> None:
    tracker = HandTracker.__new__(HandTracker)
    tracker._swap_handedness = True

    assert tracker._normalize_handedness(["Left", "Right", ""]) == ["Right", "Left", ""]


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

    assert gesture == "open_palm"
    assert confidence >= 0.80


def test_rule_based_index_uses_handedness_when_available() -> None:
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    points = np.zeros((21, 3), dtype=float)
    points[INDEX_TIP, 1] = 0.2
    points[INDEX_PIP, 1] = 0.5
    for tip, pip in [(MIDDLE_TIP, MIDDLE_PIP), (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        points[tip, 1] = 0.6
        points[pip, 1] = 0.5

    gesture, confidence = tracker._classify(points, "Left")

    assert gesture == "left_pointing_up"
    assert confidence >= 0.80


def test_rule_based_index_tolerates_ambiguous_middle_finger() -> None:
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    points = _points_for_fingers(index_delta=0.30, middle_delta=0.02)

    gesture, confidence = tracker._classify(points, "Right")

    assert gesture == "right_pointing_up"
    assert confidence >= 0.80


def test_rule_based_two_fingers_requires_clear_middle_extension() -> None:
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    points = _points_for_fingers(index_delta=0.30, middle_delta=0.30)

    gesture, confidence = tracker._classify(points, "Right")

    assert gesture == "right_two_fingers_up"
    assert confidence >= 0.80


def test_air_scroll_detects_vertical_swipe() -> None:
    """Air-scroll should detect a vertical swipe when hand centroid moves significantly."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    tracker._air_centroid_x = {}
    tracker._air_centroid_y = {}
    tracker._air_scroll_cooldown = {}
    # Simulate first frame — just set up tracking state
    pts = np.zeros((21, 3), dtype=float)
    pts[:, 0] = 0.5  # centered
    pts[:, 1] = 0.3  # upper half
    handedness = ["Right"]

    # First frame: just records position
    gesture = tracker._detect_air_scroll("pointing_up", [pts], handedness)
    assert gesture == "pointing_up"  # no previous position to compare

    # Second frame: hand moved down AND slightly right (delta_y > threshold, delta_x > min)
    pts2 = pts.copy()
    pts2[:, 1] += 0.05  # moved down 5% of screen height
    pts2[:, 0] += 0.015  # small horizontal movement to qualify as swipe
    gesture = tracker._detect_air_scroll("pointing_up", [pts2], handedness)
    assert gesture == "right_air_scroll_up"  # down movement → scroll up


def test_air_scroll_detects_upward_swipe() -> None:
    """Air-scroll should detect upward swipe → scroll_down."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    tracker._air_centroid_x = {}
    tracker._air_centroid_y = {}
    tracker._air_scroll_cooldown = {}
    pts = np.zeros((21, 3), dtype=float)
    pts[:, 0] = 0.5
    pts[:, 1] = 0.7

    # First frame
    tracker._detect_air_scroll("pointing_up", [pts], ["Left"])

    # Second frame: hand moved up AND slightly left (delta_y < 0, delta_x < 0)
    pts2 = pts.copy()
    pts2[:, 1] -= 0.05  # moved up
    pts2[:, 0] -= 0.015  # small horizontal movement to qualify as swipe
    gesture = tracker._detect_air_scroll("pointing_up", [pts2], ["Left"])
    assert gesture == "left_air_scroll_down"


def test_air_scroll_ignores_small_jitter() -> None:
    """Air-scroll should NOT fire for small hand movements (below threshold)."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    tracker._air_centroid_x = {}
    tracker._air_centroid_y = {}
    tracker._air_scroll_cooldown = {}
    pts = np.zeros((21, 3), dtype=float)
    pts[:, 0] = 0.5
    pts[:, 1] = 0.5

    # First frame
    tracker._detect_air_scroll("pointing_up", [pts], ["Right"])

    # Second frame: tiny movement (below AIR_SCROLL_THRESHOLD = 0.03)
    pts2 = pts.copy()
    pts2[:, 1] += 0.005  # only 0.5% movement
    pts2[:, 0] += 0.001  # tiny horizontal too
    gesture = tracker._detect_air_scroll("pointing_up", [pts2], ["Right"])
    assert gesture == "pointing_up"  # should NOT become air_scroll


def test_air_scroll_requires_horizontal_movement() -> None:
    """Air-scroll should require both vertical AND horizontal movement."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    tracker._air_centroid_x = {}
    tracker._air_centroid_y = {}
    tracker._air_scroll_cooldown = {}
    pts = np.zeros((21, 3), dtype=float)
    pts[:, 0] = 0.5
    pts[:, 1] = 0.5

    # First frame
    tracker._detect_air_scroll("pointing_up", [pts], ["Right"])

    # Second frame: only vertical movement, no horizontal
    pts2 = pts.copy()
    pts2[:, 1] += 0.05  # big vertical move
    # x stays the same (no horizontal movement)
    gesture = tracker._detect_air_scroll("pointing_up", [pts2], ["Right"])
    assert gesture == "pointing_up"  # should NOT fire without horizontal component


def test_air_scroll_cooldown_prevents_rapid_retrigger() -> None:
    """Air-scroll should have a cooldown to prevent rapid-fire scroll events."""
    tracker = HandTracker.__new__(HandTracker)
    tracker._prev_pinch_dist = None
    tracker._air_centroid_x = {}
    tracker._air_centroid_y = {}
    tracker._air_scroll_cooldown = {}
    pts = np.zeros((21, 3), dtype=float)
    pts[:, 0] = 0.5
    pts[:, 1] = 0.5

    # First frame: set position
    tracker._detect_air_scroll("pointing_up", [pts], ["Right"])

    # Second frame: trigger scroll (with horizontal component)
    pts2 = pts.copy()
    pts2[:, 1] += 0.05
    pts2[:, 0] += 0.015
    gesture = tracker._detect_air_scroll("pointing_up", [pts2], ["Right"])
    assert gesture == "right_air_scroll_up"

    # Third frame: same big movement but cooldown active
    pts3 = pts2.copy()
    pts3[:, 1] += 0.05
    pts3[:, 0] += 0.015
    gesture = tracker._detect_air_scroll("pointing_up", [pts3], ["Right"])
    assert gesture == "pointing_up"  # cooldown active, should not fire
