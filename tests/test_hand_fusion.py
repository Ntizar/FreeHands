"""Tests for the bimanual hand fusion module.

Verifies that:
- Single hand → no bimanual action
- Two hands → left hand controls scroll/zoom, right hand provides cursor offset
- Scroll detection works with vertical hand movement
- Zoom detection works with pinch gestures
- Reset clears state
"""
from __future__ import annotations

import numpy as np
import pytest

from freehands.gestures.hand_fusion import (
    HandFusion,
    BimanualResult,
    ZOOM_PINCH_CLOSE_DIST,
    ZOOM_PINCH_OPEN_DIST,
    ZOOM_PINCH_DELTA,
)


def _make_landmarks(x: float = 0.5, y: float = 0.5, z: float = 0.0) -> np.ndarray:
    """Create a dummy 21-landmark array at position (x, y, z)."""
    return np.full((21, 3), [x, y, z], dtype=np.float64)


def _make_landmarks_varied(base_x: float = 0.5, base_y: float = 0.5) -> np.ndarray:
    """Create landmarks with slight variation (more realistic)."""
    pts = np.full((21, 3), [base_x, base_y, 0.0], dtype=np.float64)
    # Add small jitter
    for i in range(21):
        pts[i, 0] += (i % 5 - 2) * 0.005
        pts[i, 1] += (i % 3 - 1) * 0.005
    return pts


# ── Single hand tests ────────────────────────────────────────────────────────


class TestSingleHand:
    """With only one hand, bimanual fusion should be a no-op."""

    def test_single_right_hand_no_action(self):
        fusion = HandFusion()
        hands = [_make_landmarks_varied(0.6, 0.5)]
        handedness = ["Right"]
        result = fusion.update(hands, handedness, confidence=0.9)
        assert result.scroll_action is None
        assert result.zoom_action is None
        assert result.right_active is True
        assert result.left_active is False

    def test_single_left_hand_no_action(self):
        fusion = HandFusion()
        hands = [_make_landmarks_varied(0.4, 0.5)]
        handedness = ["Left"]
        result = fusion.update(hands, handedness, confidence=0.9)
        assert result.scroll_action is None
        assert result.zoom_action is None
        assert result.right_active is False
        assert result.left_active is True

    def test_no_hands_returns_empty(self):
        fusion = HandFusion()
        result = fusion.update([], [], confidence=0.0)
        assert result.scroll_action is None
        assert result.zoom_action is None
        assert result.left_active is False
        assert result.right_active is False


# ── Bimanual scroll tests ────────────────────────────────────────────────────


class TestBimanualScroll:
    """Left hand vertical movement should produce scroll actions."""

    def test_left_hand_moving_down_scrolls_up(self):
        """Hand moving DOWN (higher Y) → scroll_up."""
        fusion = HandFusion()
        # Frame 1: left hand at top
        hands1 = [
            _make_landmarks_varied(0.6, 0.3),  # Right hand
            _make_landmarks_varied(0.3, 0.3),  # Left hand
        ]
        fusion.update(hands1, ["Right", "Left"], confidence=0.9)

        # Frame 2: left hand moved down
        hands2 = [
            _make_landmarks_varied(0.6, 0.3),  # Right hand
            _make_landmarks_varied(0.3, 0.6),  # Left hand moved down
        ]
        result = fusion.update(hands2, ["Right", "Left"], confidence=0.9)
        assert result.scroll_action == "scroll_up"
        assert result.left_active is True
        assert result.right_active is True

    def test_left_hand_moving_up_scrolls_down(self):
        """Hand moving UP (lower Y) → scroll_down."""
        fusion = HandFusion()
        # Frame 1: left hand at bottom
        hands1 = [
            _make_landmarks_varied(0.6, 0.5),
            _make_landmarks_varied(0.3, 0.7),
        ]
        fusion.update(hands1, ["Right", "Left"], confidence=0.9)

        # Frame 2: left hand moved up
        hands2 = [
            _make_landmarks_varied(0.6, 0.5),
            _make_landmarks_varied(0.3, 0.3),
        ]
        result = fusion.update(hands2, ["Right", "Left"], confidence=0.9)
        assert result.scroll_action == "scroll_down"

    def test_left_hand_no_movement_no_scroll(self):
        """No vertical movement → no scroll."""
        fusion = HandFusion()
        hands = [
            _make_landmarks_varied(0.6, 0.5),
            _make_landmarks_varied(0.3, 0.5),
        ]
        fusion.update(hands, ["Right", "Left"], confidence=0.9)
        # Second frame, same position
        result = fusion.update(hands, ["Right", "Left"], confidence=0.9)
        assert result.scroll_action is None


# ── Bimanual zoom tests ──────────────────────────────────────────────────────


class TestBimanualZoom:
    """Left hand pinch gestures should produce zoom actions."""

    def test_pinch_close_zooms_out(self):
        """Fingers closing → zoom_out."""
        fusion = HandFusion()
        # Frame 1: open hand (pinch distance ~0.15)
        hand_right = _make_landmarks_varied(0.6, 0.5)
        hand_left_open = _make_landmarks_varied(0.3, 0.5)
        # Make index tip and thumb tip far apart
        hand_left_open[8, :2] = [0.35, 0.55]  # INDEX_TIP
        hand_left_open[4, :2] = [0.25, 0.45]  # THUMB_TIP
        fusion.update([hand_right, hand_left_open], ["Right", "Left"], confidence=0.9)

        # Frame 2: fingers closing (pinch close)
        hand_left_close = _make_landmarks_varied(0.3, 0.5)
        hand_left_close[8, :2] = [0.31, 0.51]  # INDEX_TIP close to thumb
        hand_left_close[4, :2] = [0.30, 0.50]  # THUMB_TIP
        result = fusion.update([hand_right, hand_left_close], ["Right", "Left"], confidence=0.9)
        assert result.zoom_action == "zoom_out"

    def test_pinch_open_zooms_in(self):
        """Fingers opening wide → zoom_in."""
        fusion = HandFusion()
        # Frame 1: fingers close (pinch distance < 0.05)
        hand_right = _make_landmarks_varied(0.6, 0.5)
        hand_left_close = _make_landmarks_varied(0.3, 0.5)
        hand_left_close[8, :2] = [0.31, 0.51]
        hand_left_close[4, :2] = [0.30, 0.50]
        fusion.update([hand_right, hand_left_close], ["Right", "Left"], confidence=0.9)

        # Frame 2: fingers open wide
        hand_left_open = _make_landmarks_varied(0.3, 0.5)
        hand_left_open[8, :2] = [0.45, 0.65]  # INDEX_TIP far
        hand_left_open[4, :2] = [0.15, 0.35]  # THUMB_TIP far
        result = fusion.update([hand_right, hand_left_open], ["Right", "Left"], confidence=0.9)
        assert result.zoom_action == "zoom_in"


# ── Cursor offset tests ──────────────────────────────────────────────────────


class TestCursorOffset:
    """Right hand position should produce cursor offset."""

    def test_cursor_offset_when_both_hands_active(self):
        """Both hands → cursor offset based on right hand position."""
        fusion = HandFusion()
        hands = [
            _make_landmarks_varied(0.7, 0.4),  # Right hand (right side)
            _make_landmarks_varied(0.3, 0.6),  # Left hand
        ]
        result = fusion.update(hands, ["Right", "Left"], confidence=0.9)
        assert result.cursor_offset[0] > 0  # Right of center → positive x offset
        assert result.cursor_offset[1] < 0  # Above center → negative y offset

    def test_cursor_offset_zero_when_single_hand(self):
        """Single hand → no cursor offset."""
        fusion = HandFusion()
        hands = [_make_landmarks_varied(0.7, 0.4)]
        result = fusion.update(hands, ["Right"], confidence=0.9)
        assert result.cursor_offset == (0.0, 0.0)


# ── Reset tests ──────────────────────────────────────────────────────────────


class TestReset:
    """Reset should clear all per-hand state."""

    def test_reset_clears_state(self):
        fusion = HandFusion()
        # Trigger some state
        hands = [
            _make_landmarks_varied(0.6, 0.5),
            _make_landmarks_varied(0.3, 0.5),
        ]
        fusion.update(hands, ["Right", "Left"], confidence=0.9)
        fusion.reset()
        # After reset, scroll should not fire immediately
        result = fusion.update(hands, ["Right", "Left"], confidence=0.9)
        # The first frame after reset should not produce scroll
        # (because _left_scroll_y was cleared)
        assert result.scroll_action is None


# ── Cooldown tests ───────────────────────────────────────────────────────────


class TestCooldown:
    """Cooldown should prevent rapid repeated actions."""

    def test_scroll_cooldown(self):
        """After a scroll action, cooldown prevents immediate re-fire."""
        fusion = HandFusion(scroll_cooldown=3)
        hand_right = _make_landmarks_varied(0.6, 0.5)
        hand_left_1 = _make_landmarks_varied(0.3, 0.3)
        hand_left_2 = _make_landmarks_varied(0.3, 0.7)  # Moved down

        # Frame 1: initial position
        fusion.update([hand_right, hand_left_1], ["Right", "Left"], confidence=0.9)
        # Frame 2: scroll fires
        result = fusion.update([hand_right, hand_left_2], ["Right", "Left"], confidence=0.9)
        assert result.scroll_action == "scroll_up"
        # Frame 3: cooldown should prevent re-fire
        result = fusion.update([hand_right, hand_left_2], ["Right", "Left"], confidence=0.9)
        assert result.scroll_action is None
        # Frame 4: still cooldown
        result = fusion.update([hand_right, hand_left_2], ["Right", "Left"], confidence=0.9)
        assert result.scroll_action is None
        # Frame 5: cooldown expired, scroll can fire again
        result = fusion.update([hand_right, hand_left_2], ["Right", "Left"], confidence=0.9)
        # Note: same position, so no scroll (need movement)
        # Let's move again
        hand_left_3 = _make_landmarks_varied(0.3, 0.9)
        result = fusion.update([hand_right, hand_left_3], ["Right", "Left"], confidence=0.9)
        assert result.scroll_action == "scroll_up"
