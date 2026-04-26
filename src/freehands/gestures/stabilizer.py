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

    def __init__(self, required_frames: int = 8, confidence_min: float = 0.85) -> None:
        self._buf: deque[tuple[str, float]] = deque(maxlen=required_frames)
        self._required = required_frames
        self._threshold = confidence_min
        self._last_emitted: str | None = None

    def update(self, gesture: str, confidence: float) -> str | None:
        self._buf.append((gesture, confidence))
        if len(self._buf) < self._required:
            return None

        first = self._buf[0][0]
        if first == "none" or any(g != first for g, _ in self._buf):
            return None
        avg_conf = sum(c for _, c in self._buf) / len(self._buf)
        if avg_conf < self._threshold:
            return None

        # edge-trigger: don't re-emit while the same gesture is held
        if first == self._last_emitted:
            return None
        self._last_emitted = first
        return first

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None
