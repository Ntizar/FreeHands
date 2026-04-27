"""Multi-frame stabiliser — anti-false-positive layer 2."""
from __future__ import annotations

from collections import deque


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
        self._rearm_frames = rearm_frames
        self._release_frames = 0

    def update(self, gesture: str, confidence: float) -> str | None:
        if self._last_emitted is not None:
            if gesture == self._last_emitted:
                self._release_frames = 0
            else:
                self._release_frames += 1
                if self._release_frames >= self._rearm_frames:
                    self._last_emitted = None
                    self._release_frames = 0
                    self._buf.clear()

        self._buf.append((gesture, confidence))
        first = self._buf[-1][0]
        required, threshold = self._per_gesture.get(first, (self._required, self._threshold))
        if len(self._buf) < required:
            return None

        recent = list(self._buf)[-required:]
        first = recent[0][0]
        if first == "none" or any(g != first for g, _ in recent):
            return None
        avg_conf = sum(c for _, c in recent) / len(recent)
        if avg_conf < threshold:
            return None

        # edge-trigger: don't re-emit while the same gesture is held
        if first == self._last_emitted:
            return None
        self._last_emitted = first
        return first

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None
        self._release_frames = 0
