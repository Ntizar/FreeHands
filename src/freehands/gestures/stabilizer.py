"""Multi-frame stabiliser — anti-false-positive layer 2."""
from __future__ import annotations

from collections import deque


SIDE_PREFIXES = ("left_", "right_")
SIDE_AWARE_BASES = {"pointing_up", "middle_up", "two_fingers_up"}


class GestureStabilizer:
    """Only reports a gesture once it has been seen for ``required_frames`` in a row.

    Parameters
    ----------
    required_frames:
        Minimum consecutive frames the same gesture must be observed.
    confidence_min:
        Mean confidence across the buffer must clear this threshold.
    """

    def __init__(
        self,
        required_frames: int = 8,
        confidence_min: float = 0.85,
        per_gesture: dict[str, tuple[int, float]] | None = None,
        rearm_frames: int = 2,
    ) -> None:
        self._per_gesture = per_gesture or {}
        max_frames = max([required_frames, *[frames for frames, _ in self._per_gesture.values()]])
        self._buf: deque[tuple[str, float]] = deque(maxlen=max_frames)
        self._required = required_frames
        self._threshold = confidence_min
        self._last_emitted: str | None = None
        self._last_emitted_key: str | None = None
        self._rearm_frames = rearm_frames
        self._release_frames = 0

    def update(self, gesture: str, confidence: float) -> str | None:
        gesture_key = self._gesture_key(gesture)
        if self._last_emitted_key is not None:
            if gesture_key == self._last_emitted_key:
                self._release_frames = 0
            else:
                self._release_frames += 1
                if self._release_frames >= self._rearm_frames:
                    self._last_emitted = None
                    self._last_emitted_key = None
                    self._release_frames = 0
                    self._buf.clear()

        self._buf.append((gesture, confidence))
        first = self._gesture_key(self._buf[-1][0])
        required, threshold = self._per_gesture.get(first, (self._required, self._threshold))
        if len(self._buf) < required:
            return None

        recent = list(self._buf)[-required:]
        recent_keys = [self._gesture_key(g) for g, _ in recent]
        first = recent_keys[0]
        if first == "none" or any(key != first for key in recent_keys):
            return None
        avg_conf = sum(c for _, c in recent) / len(recent)
        if avg_conf < threshold:
            return None

        # edge-trigger: don't re-emit while the same gesture is held
        if first == self._last_emitted_key:
            return None
        emitted = self._emit_gesture(recent, first)
        self._last_emitted = emitted
        self._last_emitted_key = first
        return emitted

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None
        self._last_emitted_key = None
        self._release_frames = 0

    def hold_progress(self, gesture: str, required_frames: int) -> float:
        if required_frames <= 0:
            return 0.0
        held = 0
        for recent_gesture, _ in reversed(self._buf):
            if recent_gesture != gesture:
                break
            held += 1
        return min(1.0, held / required_frames)

    def hold_progress_any(self, gestures: tuple[str, ...], required_frames: int) -> float:
        if required_frames <= 0:
            return 0.0
        held = 0
        targets = set(gestures)
        for recent_gesture, _ in reversed(self._buf):
            if recent_gesture not in targets:
                break
            held += 1
        return min(1.0, held / required_frames)

    @staticmethod
    def _gesture_key(gesture: str) -> str:
        for prefix in SIDE_PREFIXES:
            if gesture.startswith(prefix):
                base = gesture[len(prefix):]
                if base in SIDE_AWARE_BASES:
                    return base
        return gesture

    @classmethod
    def _emit_gesture(cls, recent: list[tuple[str, float]], key: str) -> str:
        gestures = {g for g, _ in recent}
        if len(gestures) == 1:
            return recent[-1][0]
        if key in SIDE_AWARE_BASES:
            return key
        return recent[-1][0]
