"""Unit tests for the gesture stabiliser (anti-FP layer 2)."""
from __future__ import annotations

from freehands.gestures import GestureStabilizer


def test_emits_after_required_frames():
    s = GestureStabilizer(required_frames=4, confidence_min=0.8)
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) == "thumb_up"


def test_does_not_emit_on_mixed_gestures():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("thumb_up", 0.9)
    s.update("thumb_down", 0.9)
    assert s.update("thumb_up", 0.9) is None


def test_does_not_emit_below_confidence():
    s = GestureStabilizer(required_frames=3, confidence_min=0.95)
    s.update("thumb_up", 0.6)
    s.update("thumb_up", 0.6)
    assert s.update("thumb_up", 0.6) is None


def test_edge_trigger_no_repeat_until_change():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("thumb_up", 0.9)
    s.update("thumb_up", 0.9)
    assert s.update("thumb_up", 0.9) == "thumb_up"
    # Holding the same gesture should not re-emit
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None


def test_rearms_same_gesture_after_release():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("pointing_up", 0.9)
    s.update("pointing_up", 0.9)
    assert s.update("pointing_up", 0.9) == "pointing_up"

    s.update("none", 0.0)
    s.update("none", 0.0)
    s.update("none", 0.0)
    s.update("pointing_up", 0.9)
    s.update("pointing_up", 0.9)
    assert s.update("pointing_up", 0.9) == "pointing_up"


def test_rearms_after_two_release_frames_for_fast_clicks():
    s = GestureStabilizer(required_frames=2, confidence_min=0.5, rearm_frames=2)
    s.update("right_pointing_up", 0.9)
    assert s.update("right_pointing_up", 0.9) == "right_pointing_up"

    s.update("none", 0.0)
    s.update("none", 0.0)
    s.update("right_pointing_up", 0.9)
    assert s.update("right_pointing_up", 0.9) == "right_pointing_up"


def test_fist_to_index_transition_emits_click_gesture() -> None:
    s = GestureStabilizer(required_frames=1, confidence_min=0.5, rearm_frames=1)

    assert s.update("fist_pause", 0.9) == "fist_pause"
    assert s.update("right_pointing_up", 0.9) == "right_pointing_up"

    s.update("none", 0.0)
    assert s.update("right_pointing_up", 0.9) == "right_pointing_up"


def test_side_jitter_emits_generic_click_family_gesture():
    s = GestureStabilizer(required_frames=2, confidence_min=0.5, rearm_frames=2)

    s.update("right_pointing_up", 0.9)

    assert s.update("left_pointing_up", 0.9) == "pointing_up"


def test_open_palms_do_not_merge_left_and_right_for_safety():
    s = GestureStabilizer(
        required_frames=2,
        confidence_min=0.5,
        per_gesture={"right_open_palm": (3, 0.5), "left_open_palm": (2, 0.5)},
    )

    s.update("right_open_palm", 0.9)
    assert s.update("left_open_palm", 0.9) is None


def test_hold_progress_counts_exact_gesture_frames():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)

    s.update("right_open_palm", 0.9)
    s.update("right_open_palm", 0.9)

    assert s.hold_progress("right_open_palm", 4) == 0.5
    assert s.hold_progress("left_open_palm", 4) == 0.0


def test_single_noisy_frame_does_not_rearm_held_gesture():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("pointing_up", 0.9)
    s.update("pointing_up", 0.9)
    assert s.update("pointing_up", 0.9) == "pointing_up"

    s.update("none", 0.0)
    s.update("pointing_up", 0.9)
    s.update("pointing_up", 0.9)
    assert s.update("pointing_up", 0.9) is None


def test_low_confidence_same_gesture_rearms_click() -> None:
    s = GestureStabilizer(required_frames=1, confidence_min=0.5, rearm_frames=1)

    assert s.update("pointing_up", 0.9) == "pointing_up"
    assert s.update("pointing_up", 0.1) is None
    assert s.update("pointing_up", 0.9) == "pointing_up"


def test_held_confident_click_gesture_does_not_repeat() -> None:
    s = GestureStabilizer(required_frames=1, confidence_min=0.5, rearm_frames=1)

    assert s.update("pointing_up", 0.9) == "pointing_up"
    assert s.update("pointing_up", 0.9) is None
    assert s.update("pointing_up", 0.9) is None


def test_none_never_emits():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("none", 1.0)
    s.update("none", 1.0)
    assert s.update("none", 1.0) is None


def test_per_gesture_thresholds_override_defaults():
    s = GestureStabilizer(
        required_frames=8,
        confidence_min=0.95,
        per_gesture={"pointing_up": (3, 0.7)},
    )
    assert s.update("pointing_up", 0.8) is None
    assert s.update("pointing_up", 0.8) is None
    assert s.update("pointing_up", 0.8) == "pointing_up"
