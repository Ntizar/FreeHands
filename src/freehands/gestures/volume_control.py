"""Volume control by hand vertical position.

When the user holds their hand in the upper half of the camera frame,
volume goes up; in the lower half, volume goes down.  The middle zone
is neutral — no volume change.

This is a *static* pose gesture (not motion-based like palm-scroll or
air-scroll): the user simply raises or lowers their hand and holds it.

Detected gestures:
    * ``volume_up`` — hand centroid Y < upper_threshold
    * ``volume_down`` — hand centroid Y > lower_threshold
    * ``volume_mute`` — (optional) both hands at same level (future)

The gesture is debounced with a cooldown so the volume doesn't ramp
continuously — one step per cooldown window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Landmark indices
WRIST = 0
# Centroid: average of all 21 landmarks (more stable than wrist alone)
ALL_LANDMARKS = tuple(range(21))


# ── thresholds ────────────────────────────────────────────────────────────
# Normalised image coordinates (0 = top, 1 = bottom).
# Upper zone: hand centroid Y < 0.35 → volume up
# Lower zone: hand centroid Y > 0.65 → volume down
# Middle zone: 0.35 <= Y <= 0.65 → neutral
VOLUME_UP_THRESHOLD = 0.35
VOLUME_DOWN_THRESHOLD = 0.65

# Cooldown in frames: how many frames must pass before another volume step
VOLUME_COOLDOWN_FRAMES = 15

# Minimum confidence for the hand to be considered valid
VOLUME_MIN_CONFIDENCE = 0.60


@dataclass
class VolumeObservation:
    """Result of a volume-control detection frame.

    Attributes:
        gesture: The detected gesture, or ``None`` if no gesture.
        hand_centroid_y: Normalised Y of the hand centroid (0=top, 1=bottom).
        side: Which hand triggered the gesture (``"Left"``, ``"Right"``, or ``""``).
    """
    gesture: str | None = None
    hand_centroid_y: float = 0.0
    side: str = ""


class VolumeControl:
    """Detects volume-up / volume-down based on hand vertical position.

    Usage (in main.py tick loop):
        vc = VolumeControl()
        # inside tick():
        vol_obs = vc.detect(hand_landmarks, handedness, confidence)
        if vol_obs.gesture:
            dispatcher.execute(vol_obs.gesture)
    """

    def __init__(
        self,
        up_threshold: float = VOLUME_UP_THRESHOLD,
        down_threshold: float = VOLUME_DOWN_THRESHOLD,
        cooldown_frames: int = VOLUME_COOLDOWN_FRAMES,
    ) -> None:
        self._up_threshold = up_threshold
        self._down_threshold = down_threshold
        self._cooldown_frames = cooldown_frames

        # Per-hand tracking: (centroid_y, cooldown_counter)
        self._hand_y: dict[int, float] = {}
        self._cooldown: dict[int, int] = {}
        self._frame_counter = 0
        self._last_gesture: dict[int, str | None] = {}

    def detect(
        self,
        hands: list[np.ndarray],
        handedness: list[str],
        confidence: float = 0.0,
    ) -> VolumeObservation:
        """Run volume detection on the current frame.

        Args:
            hands: List of hand landmark arrays, each shape (21, 3).
            handedness: List of side labels ("Left"/"Right").
            confidence: Overall gesture confidence from the stabilizer.

        Returns:
            A :class:`VolumeObservation` with the detected gesture.
        """
        if confidence < VOLUME_MIN_CONFIDENCE or not hands:
            return VolumeObservation()

        self._frame_counter += 1

        best_obs: VolumeObservation | None = None

        for idx, (pts, side) in enumerate(zip(hands, handedness)):
            if side not in ("Left", "Right"):
                continue

            # Decrement cooldown
            cd = self._cooldown.get(idx, 0)
            if cd > 0:
                self._cooldown[idx] = cd - 1
                continue

            # Compute hand centroid (average of all landmarks, X+Y only)
            centroid_x = float(np.mean(pts[ALL_LANDMARKS, 0]))
            centroid_y = float(np.mean(pts[ALL_LANDMARKS, 1]))

            # Classify zone on every frame (no need for previous frame)
            gesture = self._classify_zone(centroid_y)
            if gesture:
                # Only fire if the gesture is different from last frame
                # (prevents rapid toggling when hand is near threshold)
                last = self._last_gesture.get(idx)
                if gesture != last:
                    self._cooldown[idx] = self._cooldown_frames
                    self._last_gesture[idx] = gesture
                    obs = VolumeObservation(
                        gesture=gesture,
                        hand_centroid_y=centroid_y,
                        side=side,
                    )
                    # Prefer the first detected gesture
                    if best_obs is None:
                        best_obs = obs
            else:
                # Neutral zone — reset last gesture so entering a zone fires
                if self._last_gesture.get(idx) is not None:
                    self._last_gesture[idx] = None

            self._hand_y[idx] = centroid_y

        return best_obs or VolumeObservation()

    @staticmethod
    def _classify_zone(centroid_y: float) -> str | None:
        """Classify hand centroid Y into a volume gesture.

        Args:
            centroid_y: Normalised Y coordinate (0=top, 1=bottom).

        Returns:
            "volume_up", "volume_down", or None (neutral).
        """
        if centroid_y < VOLUME_UP_THRESHOLD:
            return "volume_up"
        if centroid_y > VOLUME_DOWN_THRESHOLD:
            return "volume_down"
        return None

    def reset(self) -> None:
        """Reset all per-hand state (e.g. when hand is lost)."""
        self._hand_y.clear()
        self._cooldown.clear()
        self._last_gesture.clear()
