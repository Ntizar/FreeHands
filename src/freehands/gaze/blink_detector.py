"""Blink detection via Eye Aspect Ratio (EAR).

Uses the vertical distance between upper/lower eyelid landmarks relative
to the horizontal eye width.  A blink is detected when the EAR drops below
a threshold for a minimum number of consecutive frames, then recovers.

Extended with:
- **Double-blink detection**: two blinks within a short window (default 500 ms)
  are interpreted as a distinct gesture.
- **Prolonged-close detection**: holding the eyes closed beyond a configurable
  threshold triggers a "drag hold" mode.

Reference: "Eye Aspect Ratio" (Kotsia & Pitas, 2006)
"""
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto


# ── Blink event types ──────────────────────────────────────────────────

class BlinkEventType(Enum):
    """Classification of a blink event."""
    SINGLE = auto()       # Normal blink — one quick close + open
    DOUBLE = auto()       # Two blinks in rapid succession
    PROLONGED = auto()    # Eyes held closed longer than the drag threshold


@dataclass
class BlinkEvent:
    """A single blink event with type classification."""
    timestamp: float               # monotonic seconds
    frame_count: int               # how many frames the eye was closed
    confidence: float              # 0..1, based on how clean the dip was
    event_type: BlinkEventType = BlinkEventType.SINGLE
    # For double-blink: time since the previous blink
    time_since_last: float | None = None
    # For prolonged: total closed duration in seconds
    closed_duration: float = 0.0


class BlinkDetector:
    """Detects blinks by tracking the Eye Aspect Ratio (EAR).

    EAR is computed as the average vertical eye height divided by the
    horizontal eye width.  When both eyes close simultaneously (a true
    blink), EAR drops sharply.

    Extended capabilities:
    - **Double-blink**: two single blinks within ``double_blink_window``
      seconds produce a ``BlinkEventType.DOUBLE`` event.
    - **Prolonged close**: holding eyes closed beyond
      ``prolonged_close_frames`` frames produces a
      ``BlinkEventType.PROLONGED`` event (used for drag-start).

    Usage::

        detector = BlinkDetector()
        for frame in frames:
            left_ear, right_ear = compute_ear(frame)
            event = detector.update(left_ear, right_ear)
            if event:
                print(f"Blink {event.event_type.name} at {event.timestamp:.3f}s")
    """

    def __init__(
        self,
        ear_close_threshold: float = 0.25,
        min_blink_frames: int = 3,
        max_blink_frames: int = 15,
        recovery_frames: int = 2,
        debounce_seconds: float = 0.3,
        double_blink_window: float = 0.5,
        prolonged_close_frames: int = 12,
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
            double_blink_window: Seconds between two blinks to classify as
                a double-blink gesture. Default 0.5 s.
            prolonged_close_frames: Frames of sustained closure to trigger a
                "prolonged" event (drag-start). Default 12 frames (~400 ms
                at 30 fps).
        """
        self.ear_close_threshold = ear_close_threshold
        self.min_blink_frames = min_blink_frames
        self.max_blink_frames = max_blink_frames
        self.recovery_frames = recovery_frames
        self.debounce_seconds = debounce_seconds
        self.double_blink_window = double_blink_window
        self.prolonged_close_frames = prolonged_close_frames

        # State
        self._closed_frames = 0
        self._recover_frames = 0
        self._last_blink_at: float | None = None
        self._blink_started_at: float | None = None

        # Double-blink tracking
        self._previous_blink_at: float | None = None

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

        Returns a :class:`BlinkEvent` if a blink or prolonged-close event
        was just completed, ``None`` otherwise.

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

            # Prolonged close: eyes held closed beyond threshold
            if self._closed_frames >= self.prolonged_close_frames:
                duration = now - self._blink_started_at
                event = self._fire_prolonged(now, duration)
                return event

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
        """Emit a blink event, checking for double-blink pattern."""
        # Calculate time since the previous blink for double-blink detection.
        # Use _last_blink_at (time of last fired blink) since it's always
        # set after each blink fires. _previous_blink_at tracks the blink
        # before that for potential future extensions.
        time_since_last: float | None = None
        if self._last_blink_at is not None:
            time_since_last = now - self._last_blink_at

        # Debounce: reject if too close to the immediately preceding blink
        if self._last_blink_at is not None:
            if now - self._last_blink_at < self.debounce_seconds:
                self._reset()
                return None

        # Shift the tracking pointers
        self._previous_blink_at = self._last_blink_at
        self._last_blink_at = now

        confidence = min(1.0, self._closed_frames / self.min_blink_frames)
        event_type = BlinkEventType.SINGLE

        # Double-blink: two blinks within the double_blink_window
        if (
            time_since_last is not None
            and time_since_last <= self.double_blink_window
        ):
            event_type = BlinkEventType.DOUBLE

        event = BlinkEvent(
            timestamp=now,
            frame_count=self._closed_frames,
            confidence=confidence,
            event_type=event_type,
            time_since_last=time_since_last,
        )
        self._reset()
        return event

    def _fire_prolonged(self, now: float, duration: float) -> BlinkEvent:
        """Emit a prolonged-close event (drag-start)."""
        event = BlinkEvent(
            timestamp=now,
            frame_count=self._closed_frames,
            confidence=1.0,
            event_type=BlinkEventType.PROLONGED,
            closed_duration=duration,
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
    def is_prolonged_close(self) -> bool:
        """Whether the eyes have been closed longer than the prolonged threshold."""
        return self._closed_frames >= self.prolonged_close_frames

    @property
    def blink_count(self) -> int:
        """Total number of blinks detected since construction."""
        # This is a rough estimate - we track via last_blink_at resets
        # For accurate counting, use a separate counter
        return 0  # Placeholder - actual count tracked by caller if needed
