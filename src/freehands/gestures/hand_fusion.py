"""Bimanual hand fusion: two hands, independent roles.

When both hands are detected, they are assigned complementary roles:

    * **Right hand** — cursor position + click (primary pointer)
    * **Left hand**  — scroll (vertical sweep) + zoom (pinch open/close)

When only one hand is visible, the system falls back to normal
single-hand behaviour: the detected hand controls cursor + click.

Detection logic:

    * Left-hand scroll: vertical centroid movement (same algorithm as
      palm-scroll but works with any left-hand pose).
    * Left-hand zoom: pinch open → zoom_in, pinch close → zoom_out.
    * Right-hand cursor: centroid position mapped to screen coordinates.

The module is stateless per-frame — it takes the current hand observations
and returns a :class:`BimanualResult` with cursor offset, scroll/zoom
actions, and metadata for overlay display.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Landmark indices
WRIST = 0
THUMB_TIP, INDEX_TIP = 4, 8

# Centroid: average of all 21 landmarks (more stable than wrist alone)
ALL_LANDMARKS = tuple(range(21))

# ── Scroll thresholds (left hand) ──────────────────────────────────────────
# Normalised image coords — same scale as palm-scroll.
# Minimum vertical displacement to fire a scroll step.
LEFT_HAND_SCROLL_THRESHOLD = 0.012
# Cooldown frames between scroll steps
LEFT_HAND_SCROLL_COOLDOWN = 3

# ── Zoom thresholds (left hand) ────────────────────────────────────────────
# Pinch distance thresholds for zoom detection.
ZOOM_PINCH_CLOSE_DIST = 0.05    # fingers close → zoom out
ZOOM_PINCH_OPEN_DIST = 0.10     # fingers far → zoom in
ZOOM_PINCH_DELTA = 0.02         # min delta to trigger action
ZOOM_COOLDOWN_FRAMES = 5


@dataclass
class BimanualResult:
    """Result of bimanual fusion for the current frame.

    Attributes:
        cursor_offset: (dx, dy) to add to the right-hand cursor position.
            Applied after the normal gaze cursor — fine-tunes position.
        scroll_action: "scroll_up", "scroll_down", or None.
        zoom_action: "zoom_in", "zoom_out", or None.
        left_hand_centroid: Normalised centroid of left hand (for overlay).
        right_hand_centroid: Normalised centroid of right hand (for overlay).
        left_active: Whether left hand is detected and contributing.
        right_active: Whether right hand is detected and contributing.
    """
    cursor_offset: tuple[float, float] = (0.0, 0.0)
    scroll_action: str | None = None
    zoom_action: str | None = None
    left_hand_centroid: tuple[float, float] | None = None
    right_hand_centroid: tuple[float, float] | None = None
    left_active: bool = False
    right_active: bool = False


class HandFusion:
    """Bimanual fusion: right hand = cursor, left hand = scroll + zoom.

    Usage (in main.py tick loop):
        fusion = HandFusion()
        # inside tick():
        result = fusion.update(hands, handedness)
        if result.scroll_action:
            dispatcher.execute(result.scroll_action)
        if result.zoom_action:
            dispatcher.execute(result.zoom_action)
        # cursor_offset can be added to the right-hand cursor position
    """

    def __init__(
        self,
        scroll_threshold: float = LEFT_HAND_SCROLL_THRESHOLD,
        scroll_cooldown: int = LEFT_HAND_SCROLL_COOLDOWN,
        zoom_cooldown: int = ZOOM_COOLDOWN_FRAMES,
    ) -> None:
        self._scroll_threshold = scroll_threshold
        self._scroll_cooldown = scroll_cooldown
        self._zoom_cooldown = zoom_cooldown

        # Left-hand scroll tracking: (centroid_y, cooldown)
        self._left_scroll_y: float | None = None
        self._left_scroll_cooldown: int = 0

        # Left-hand zoom tracking: last pinch distance
        self._last_pinch_dist: float | None = None
        self._left_zoom_cooldown: int = 0

        # Frame counter for cooldowns
        self._frame_counter = 0

    def update(
        self,
        hands: list[np.ndarray],
        handedness: list[str],
        confidence: float = 0.0,
    ) -> BimanualResult:
        """Run bimanual fusion on the current frame.

        Args:
            hands: List of hand landmark arrays, each shape (21, 3).
            handedness: List of side labels ("Left"/"Right").
            confidence: Overall gesture confidence from the stabilizer.

        Returns:
            A :class:`BimanualResult` with cursor offset, scroll/zoom actions.
        """
        if confidence < 0.50 or not hands:
            return BimanualResult()

        self._frame_counter += 1

        # Separate hands by side
        left_hand: np.ndarray | None = None
        right_hand: np.ndarray | None = None

        for pts, side in zip(hands, handedness):
            if side == "Left":
                left_hand = pts
            elif side == "Right":
                right_hand = pts

        result = BimanualResult()

        # ── Right hand: cursor position (only when bimanual is active) ──
        # Cursor offset is only applied when both hands are visible.
        # With a single hand, normal gaze cursor takes full control.
        if right_hand is not None and left_hand is not None:
            result.right_active = True
            centroid = self._centroid(right_hand)
            result.right_hand_centroid = centroid
            # Cursor offset: small nudges based on right-hand position relative
            # to center. When right hand is above center, cursor nudges up.
            # This gives fine-grained control without overriding gaze.
            cx, cy = centroid
            result.cursor_offset = (
                (cx - 0.5) * 20.0,   # ±10px horizontal
                (cy - 0.5) * 20.0,   # ±10px vertical
            )
        elif right_hand is not None:
            result.right_active = True
            result.right_hand_centroid = self._centroid(right_hand)

        # ── Left hand: scroll + zoom ─────────────────────────────────────
        if left_hand is not None:
            result.left_active = True
            centroid = self._centroid(left_hand)
            result.left_hand_centroid = centroid

            # Decrement cooldowns
            if self._left_scroll_cooldown > 0:
                self._left_scroll_cooldown -= 1
            if self._left_zoom_cooldown > 0:
                self._left_zoom_cooldown -= 1

            # ── Left-hand scroll: vertical movement ──────────────────────
            if self._left_scroll_cooldown <= 0:
                prev_y = self._left_scroll_y
                if prev_y is not None:
                    delta_y = centroid[1] - prev_y
                    # Image coords: Y increases downward.
                    # Hand moving DOWN (delta_y > 0) → scroll_up
                    # Hand moving UP (delta_y < 0) → scroll_down
                    if abs(delta_y) >= self._scroll_threshold:
                        direction = "up" if delta_y > 0 else "down"
                        result.scroll_action = f"scroll_{direction}"
                        self._left_scroll_cooldown = self._scroll_cooldown

                self._left_scroll_y = centroid[1]

            # ── Left-hand zoom: pinch open/close ─────────────────────────
            if self._left_zoom_cooldown <= 0:
                pinch_dist = float(
                    np.linalg.norm(
                        left_hand[INDEX_TIP, :2] - left_hand[THUMB_TIP, :2]
                    )
                )

                if self._last_pinch_dist is not None:
                    delta = pinch_dist - self._last_pinch_dist

                    if pinch_dist < ZOOM_PINCH_CLOSE_DIST:
                        # Fingers close → zoom out
                        result.zoom_action = "zoom_out"
                        self._left_zoom_cooldown = self._zoom_cooldown
                    elif delta > ZOOM_PINCH_DELTA and pinch_dist > ZOOM_PINCH_OPEN_DIST:
                        # Fingers opening wide → zoom in
                        result.zoom_action = "zoom_in"
                        self._left_zoom_cooldown = self._zoom_cooldown

                self._last_pinch_dist = pinch_dist

        return result

    @staticmethod
    def _centroid(pts: np.ndarray) -> tuple[float, float]:
        """Compute centroid of hand landmarks (X, Y only)."""
        return (
            float(np.mean(pts[:, 0])),
            float(np.mean(pts[:, 1])),
        )

    def reset(self) -> None:
        """Reset all per-hand state (e.g. when hands are lost)."""
        self._left_scroll_y = None
        self._left_scroll_cooldown = 0
        self._last_pinch_dist = None
        self._left_zoom_cooldown = 0
