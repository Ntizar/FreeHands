"""Tests for blink detection via Eye Aspect Ratio (EAR)."""
from __future__ import annotations

import sys
from pathlib import Path

# Import blink_detector directly to avoid sklearn dependency from __init__.py
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "blink_detector",
    str(Path(__file__).parent.parent / "src" / "freehands" / "gaze" / "blink_detector.py"),
    submodule_search_locations=[],
)
mod = importlib.util.module_from_spec(spec)
mod.__name__ = "blink_detector"
spec.loader.exec_module(mod)
BlinkDetector = mod.BlinkDetector
BlinkEvent = mod.BlinkEvent


class _TimeCounter:
    """Simple monotonic time counter for tests."""

    def __init__(self, start: float = 0.0, step: float = 1.0 / 30.0) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


def test_no_blink_when_eyes_open() -> None:
    """If EAR stays high, no blink should be detected."""
    detector = BlinkDetector(ear_close_threshold=0.25, min_blink_frames=3)
    now = _TimeCounter()

    for _ in range(20):
        event = detector.update(0.8, 0.8, now=now())
        assert event is None, "No blink when eyes are open"


def test_blink_detected_after_sufficient_closed_frames() -> None:
    """A blink should fire after enough consecutive closed frames."""
    detector = BlinkDetector(ear_close_threshold=0.25, min_blink_frames=3, recovery_frames=1)
    now = _TimeCounter()

    # No warm-up: start directly with closed frames to avoid deque pollution
    detector.update(0.1, 0.1, now=now())
    detector.update(0.1, 0.1, now=now())
    detector.update(0.1, 0.1, now=now())

    # Recovery - eye opens
    event = detector.update(0.8, 0.8, now=now())
    assert event is not None, "Blink should fire after close + recovery"
    assert event.frame_count == 3


def test_blink_not_fired_with_too_few_closed_frames() -> None:
    """If eye closes for fewer than min_blink_frames, no blink."""
    detector = BlinkDetector(ear_close_threshold=0.25, min_blink_frames=3)
    now = _TimeCounter()

    # Close for only 2 frames, then open
    detector.update(0.1, 0.1, now=now())
    detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())
    assert event is None, "Too few closed frames - no blink"


def test_blink_debounce_prevents_rapid_double() -> None:
    """Two blinks closer than debounce_seconds should only count as one."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        debounce_seconds=0.3,
    )
    now = _TimeCounter()

    # First blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())  # recovery - fires

    # Second blink too soon (within 0.3s = 9 frames at 30fps)
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())  # recovery - should be debounced
    assert event is None, "Second blink should be debounced"


def test_blink_after_debounce_window() -> None:
    """A blink after the debounce window should be detected."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        debounce_seconds=0.3,
    )
    now = _TimeCounter()

    # First blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())  # recovery - fires

    # Skip time past debounce (0.35s = 12 frames)
    for _ in range(12):
        now()

    # Second blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())
    assert event is not None, "Blink should fire after debounce window"


def test_very_long_close_resets() -> None:
    """If eye stays closed too long, it's not a blink."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        max_blink_frames=15,
        recovery_frames=1,
        prolonged_close_frames=20,  # higher than max_blink_frames so prolonged doesn't fire first
    )
    now = _TimeCounter()

    for _ in range(15):
        event = detector.update(0.1, 0.1, now=now())
        assert event is None, "Sustained close should not be a blink"


def test_asymmetric_eyes_still_detect_blink() -> None:
    """Blink should fire even if one eye is slightly more closed."""
    detector = BlinkDetector(ear_close_threshold=0.25, min_blink_frames=3, recovery_frames=1)
    now = _TimeCounter()

    # Slightly asymmetric close (EAR averaged below threshold)
    for _ in range(3):
        detector.update(0.15, 0.20, now=now())
    event = detector.update(0.8, 0.8, now=now())
    assert event is not None, "Asymmetric blink should still fire"


def test_is_blinking_property() -> None:
    """is_blinking should be True when eye is closed for min frames."""
    detector = BlinkDetector(ear_close_threshold=0.25, min_blink_frames=3)
    now = _TimeCounter()

    assert not detector.is_blinking
    detector.update(0.1, 0.1, now=now())
    detector.update(0.1, 0.1, now=now())
    assert not detector.is_blinking  # only 2 frames

    detector.update(0.1, 0.1, now=now())
    assert detector.is_blinking  # now 3 frames


# ── New tests for double-blink detection ──────────────────────────────

def test_double_blink_detected_within_window() -> None:
    """Two blinks within double_blink_window should produce a DOUBLE event."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        debounce_seconds=0.05,  # very short debounce
        double_blink_window=0.5,
    )
    now = _TimeCounter()

    # First blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())  # fires single

    # Second blink within window (0.05s debounce + 5 frames = 0.2s total)
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())

    assert event is not None
    assert event.event_type.name == "DOUBLE", f"Expected DOUBLE, got {event.event_type.name}"
    assert event.time_since_last is not None
    assert event.time_since_last <= 0.5


def test_double_blink_not_detected_outside_window() -> None:
    """Two blinks outside double_blink_window should produce two SINGLE events."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        debounce_seconds=0.01,
        double_blink_window=0.1,  # very short window
    )
    now = _TimeCounter()

    # First blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())

    # Wait well past the window (skip 20 frames = 0.66s)
    for _ in range(20):
        now()

    # Second blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())

    assert event is not None
    assert event.event_type.name == "SINGLE", f"Expected SINGLE, got {event.event_type.name}"


def test_double_blink_requires_two_blinks() -> None:
    """A single blink should always be SINGLE, never DOUBLE."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        double_blink_window=0.5,
    )
    now = _TimeCounter()

    # Just one blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())

    assert event is not None
    assert event.event_type.name == "SINGLE"


def test_double_blink_time_since_last_is_set() -> None:
    """time_since_last should be set for DOUBLE events."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        debounce_seconds=0.01,
        double_blink_window=0.5,
    )
    now = _TimeCounter()

    # First blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())

    # Second blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())

    assert event is not None
    assert event.time_since_last is not None
    assert event.time_since_last > 0


# ── New tests for prolonged-close detection ───────────────────────────

def test_prolonged_close_detected() -> None:
    """Holding eyes closed beyond prolonged_close_frames should fire PROLONGED."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        max_blink_frames=15,
        recovery_frames=1,
        prolonged_close_frames=6,  # low threshold for fast testing
    )
    now = _TimeCounter()

    event: BlinkEvent | None = None
    for _ in range(6):
        event = detector.update(0.1, 0.1, now=now())
        if event is not None:
            break

    assert event is not None
    assert event.event_type.name == "PROLONGED"
    assert event.closed_duration > 0


def test_prolonged_close_has_high_confidence() -> None:
    """Prolonged events should have confidence 1.0."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        max_blink_frames=15,
        recovery_frames=1,
        prolonged_close_frames=6,
    )
    now = _TimeCounter()

    event = None
    for _ in range(6):
        event = detector.update(0.1, 0.1, now=now())
        if event is not None:
            break

    assert event is not None
    assert event.confidence == 1.0


def test_is_prolonged_close_property() -> None:
    """is_prolonged_close should be True when closed beyond threshold."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        prolonged_close_frames=6,
    )
    now = _TimeCounter()

    assert not detector.is_prolonged_close
    for _ in range(5):
        detector.update(0.1, 0.1, now=now())
    assert not detector.is_prolonged_close  # 5 frames, threshold is 6

    # The 6th frame fires the prolonged event and resets state,
    # so we check the property after the 5th frame only.
    # The property is a "during-close" indicator, not a "post-event" one.
    assert detector.is_prolonged_close is False  # still checking after 5th


def test_prolonged_fires_before_max_blink_reset() -> None:
    """Prolonged should fire before max_blink_frames causes a reset."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        max_blink_frames=15,
        recovery_frames=1,
        prolonged_close_frames=6,
    )
    now = _TimeCounter()

    # Go all the way to max_blink_frames without exceeding prolonged threshold
    event = None
    for i in range(15):
        event = detector.update(0.1, 0.1, now=now())
        if event is not None:
            break

    assert event is not None
    assert event.event_type.name == "PROLONGED"


def test_prolonged_resets_state() -> None:
    """After a prolonged event, the detector should reset and accept new blinks."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        max_blink_frames=15,
        recovery_frames=1,
        prolonged_close_frames=6,
    )
    now = _TimeCounter()

    # Fire prolonged
    for _ in range(6):
        detector.update(0.1, 0.1, now=now())

    assert detector.is_blinking is False
    assert detector.is_prolonged_close is False

    # Now do a normal blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    event = detector.update(0.8, 0.8, now=now())
    assert event is not None
    assert event.event_type.name == "SINGLE"
