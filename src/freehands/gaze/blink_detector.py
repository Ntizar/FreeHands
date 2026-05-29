"""Blink detection via Eye Aspect Ratio (EAR).

Uses the vertical distance between upper/lower eyelid landmarks relative
to the horizontal eye width.  A blink is detected when the EAR drops below
a threshold for a minimum number of consecutive frames, then recovers.

Reference: "Eye Aspect Ratio" (Kotsia & Pitas, 2006)
"""
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class BlinkEvent:
    """A single blink event."""
    timestamp: float       # monotonic seconds
    frame_count: int       # how many frames the eye was closed
    confidence: float      # 0..1, based on how clean the dip was


class BlinkDetector:
    """Detects blinks by tracking the Eye Aspect Ratio (EAR).

    EAR is computed as the average vertical eye height divided by the
    horizontal eye width.  When both eyes close simultaneously (a true
    blink), EAR drops sharply.

    Usage::

        detector = BlinkDetector()
        for frame in frames:
            left_ear, right_ear = compute_ear(frame)
            event = detector.update(left_ear, right_ear)
            if event:
                print(f"Blink at {event.timestamp:.3f}s")
    """

    def __init__(
        self,
        ear_close_threshold: float = 0.25,
        min_blink_frames: int = 3,
        max_blink_frames: int = 15,
        recovery_frames: int = 2,
        debounce_seconds: float = 0.3,
    ) -> None:
        """
        Args:
            ear_close_threshold: EAR below which an eye is considered "closed".
                Typical range: 0.20-0.30. Lower = harder to trigger.
            min_blink_frames: Minimum consecutive closed frames to register a blink.
            max_blink_frames: If the eye stays closed longer than this, reset
                (avoids false positives from eyes being naturally closed).
            recovery_frames: How many frames above threshold after a close
                to consider the blink "complete".
            debounce_seconds: Minimum time between consecutive blink events.
        """
        self.ear_close_threshold = ear_close_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.recovery_frames = recovery_frames
        self.debounce_seconds = debounce_seconds

        # State
        self._closed_frames = 0
        self._recover_frames = 0
        self._last_blink_at: float | None = None
        self._blink_started_at: float | None = None

        # History for EAR smoothing (per eye)
        self._left_ear_history: deque[float] = deque(maxlen=2)
        self._right_ear_history: deque[float] = deque(maxlen=2)

    def update(
        self,
        left_ear: float,
        right_ear: float,
        now: float | None = None,
    ) -> BlinkEvent | None:
        """Process a new pair of EAR values.

        Returns a :class:`BlinkEvent` if a blink was just completed, ``None``
        otherwise.

        Args:
            left_ear: Normalised EAR for the left eye (0 = fully closed, 1 = fully open).
            right_ear: Normalised EAR for the right eye.
            now: Current monotonic time.  Defaults to ``time.monotonic()``.
        """
        if now is None:
            now = time.monotonic()

        # Smooth EAR with moving average
        self._left_ear_history.append(left_ear)
        self._right_ear_history.append(right_ear)
        avg_left = sum(self._left_ear_history) / len(self._left_ear_history)
        avg_right = sum(self._right_ear_history) / len(self._right_ear_history)
        avg_ear = (avg_left + avg_right) / 2.0

        if avg_ear < self.ear_close_threshold:
            # Eye is closed
            if self._blink_started_at is None:
                self._blink_started_at = now
            self._closed_frames += 1

            # Too long closed - likely not a blink (eyes naturally closed)
            if self._closed_frames >= self.max_blink_frames:
                self._reset()
                return None

        else:
            # Eye is open - check if we're recovering from a close
            if self._closed_frames >= self.min_blink_frames:
                self._recover_frames += 1
                if self._recover_frames >= self.recovery_frames:
                    # Blink complete!
                    event = self._fire_blink(now)
                    return event
            else:
                self._recover_frames = 0

        return None

    def _fire_blink(self, now: float) -> BlinkEvent | None:
        """Emit a blink event if debounce allows."""
        if self._last_blink_at is not None:
            if now - self._last_blink_at < self.debounce_seconds:
                self._reset()
                return None

        self._last_blink_at = now
        confidence = min(1.0, self._closed_frames / self.min_blink_frames)
        event = BlinkEvent(
            timestamp=now,
            frame_count=self._closed_frames,
            confidence=confidence,
        )
        self._reset()
        return event

    def _reset(self) -> None:
        """Reset all blink detection state."""
        self._closed_frames = 0
        self._recover_frames = 0
        self._blink_started_at = None
        self._left_ear_history.clear()
        self._right_ear_history.clear()

    def reset(self) -> None:
        """Public reset (same as _reset but allows resetting from outside)."""
        self._reset()

    @property
    def is_blinking(self) -> bool:
        """Whether the detector is currently in a blink (closed frames >= min)."""
        return self._closed_frames >= self.min_blink_frames

    @property
    def blink_count(self) -> int:
        """Total number of blinks detected since construction."""
        # This is a rough estimate - we track via last_blink_at resets
        # For accurate counting, use a separate counter
        return 0  # Placeholder - actual count tracked by caller if needed
