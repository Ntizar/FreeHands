"""Tests for per-user EAR calibration (Eye2cursor pattern)."""
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
EARCalibration = mod.EARCalibration


class _TimeCounter:
    """Simple monotonic time counter for tests."""

    def __init__(self, start: float = 0.0, step: float = 1.0 / 30.0) -> None:
        self._t = start
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


# ── EARCalibration tests ─────────────────────────────────────────────

def test_ear_calibration_updates_average() -> None:
    """Calibration should track the running average of open-eye EAR."""
    cal = EARCalibration()
    # Feed known open-eye EAR values
    for _ in range(5):
        cal.update_open_ear(0.5)
    assert cal.open_ear_avg > 0.3  # should be close to 0.5


def test_ear_calibration_computes_adaptive_threshold() -> None:
    """After enough samples, adaptive_threshold should be computed."""
    cal = EARCalibration(calibration_required_frames=5)
    for _ in range(5):
        cal.update_open_ear(0.5)
    assert cal.calibration_complete
    assert cal.adaptive_threshold < 0.5  # threshold < average
    assert cal.adaptive_threshold > 0.1  # but not too low


def test_ear_calibration_confidence() -> None:
    """Confidence should grow from 0 to 1 as samples are collected."""
    cal = EARCalibration(calibration_required_frames=10)
    assert cal.confidence == 0.0
    for i in range(10):
        cal.update_open_ear(0.5)
        expected = min(1.0, (i + 1) / 10)
        assert abs(cal.confidence - expected) < 0.01


def test_ear_calibration_reset() -> None:
    """Reset should clear all calibration state."""
    cal = EARCalibration(calibration_required_frames=5)
    for _ in range(5):
        cal.update_open_ear(0.5)
    assert cal.calibration_complete
    cal.reset()
    assert not cal.calibration_complete
    assert cal.open_ear_avg == 0.0
    assert cal.adaptive_threshold == 0.25  # fallback


def test_ear_calibration_clamps_threshold() -> None:
    """Adaptive threshold should be clamped to min/max bounds."""
    cal = EARCalibration(
        calibration_required_frames=5,
        calibration_min_threshold=0.10,
        calibration_max_threshold=0.40,
    )
    # Feed very high EAR values — threshold should not exceed max
    for _ in range(5):
        cal.update_open_ear(0.95)
    assert cal.adaptive_threshold <= 0.40

    # Feed very low EAR values — threshold should not go below min
    cal2 = EARCalibration(
        calibration_required_frames=5,
        calibration_min_threshold=0.10,
        calibration_max_threshold=0.40,
    )
    for _ in range(5):
        cal2.update_open_ear(0.12)
    assert cal2.adaptive_threshold >= 0.10


def test_ear_calibration_tracks_min_max() -> None:
    """Should track min/max of observed open-eye EAR."""
    cal = EARCalibration()
    cal.update_open_ear(0.5)
    cal.update_open_ear(0.6)
    cal.update_open_ear(0.4)
    assert cal.open_ear_min <= 0.4
    assert cal.open_ear_max >= 0.6


# ── BlinkDetector with adaptive EAR tests ───────────────────────────

def test_adaptive_detector_uses_fixed_threshold_before_calibration() -> None:
    """Before calibration is complete, should use base threshold."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    # Feed open-eye values but not enough for calibration
    for _ in range(10):
        detector.update(0.8, 0.8, now=now())
    # Should still be using base threshold
    assert not detector.calibration.calibration_complete
    assert detector.current_threshold == 0.25


def test_adaptive_detector_learns_threshold() -> None:
    """After calibration, threshold should adapt to user's open-eye EAR."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    # Simulate user with higher-than-normal open-eye EAR
    for _ in range(40):
        detector.update(0.85, 0.85, now=now())
    assert detector.calibration.calibration_complete
    assert detector.current_threshold != 0.25  # threshold adapted


def test_adaptive_detector_detects_blink_after_calibration() -> None:
    """Blink detection should work correctly after adaptive calibration."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    # Calibrate with open eyes (force to skip warm-up period)
    for _ in range(40):
        detector.update(0.8, 0.8, now=now())
    assert detector.calibration.calibration_complete

    # Force calibration to ensure threshold is set
    detector.force_calibration_complete()

    # Now simulate a blink (EAR well below the adaptive threshold ~0.40)
    # Need 4 frames because the 2-frame history deque dilutes the first closed frame
    for _ in range(4):
        detector.update(0.05, 0.05, now=now())
    event = detector.update(0.8, 0.8, now=now())
    assert event is not None, f"Blink should be detected after calibration. Threshold={detector.current_threshold:.4f}"


def test_adaptive_detector_with_low_open_ear_user() -> None:
    """Users with naturally lower open-eye EAR should get an adapted threshold."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    # Simulate user with lower-than-normal open-eye EAR
    for _ in range(40):
        detector.update(0.45, 0.45, now=now())
    assert detector.calibration.calibration_complete
    # Threshold is computed as avg - margin = 0.45 - 0.0675 = 0.3825
    # This is actually HIGHER than 0.25 because the margin is absolute, not relative
    # The key point: the threshold adapts to the user's natural EAR
    assert detector.current_threshold != 0.25  # threshold adapted
    # And it should be reasonable (between min and max bounds)
    assert 0.10 <= detector.current_threshold <= 0.40


def test_non_adaptive_detector_ignores_calibration() -> None:
    """When use_adaptive_ear=False, calibration should be ignored."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=False,
    )
    now = _TimeCounter()
    for _ in range(40):
        detector.update(0.8, 0.8, now=now())
    # Threshold should remain at base value
    assert detector.current_threshold == 0.25


def test_calibration_info() -> None:
    """get_calibration_info should return meaningful data."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    for _ in range(40):
        detector.update(0.8, 0.8, now=now())
    info = detector.get_calibration_info()
    assert info["calibration_complete"] is True
    assert info["open_ear_samples"] == 40
    assert info["adaptive_threshold"] > 0
    assert info["confidence"] > 0


def test_force_calibration_complete() -> None:
    """force_calibration_complete should set calibration to done."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        use_adaptive_ear=True,
    )
    detector.force_calibration_complete()
    assert detector.calibration.calibration_complete
    assert detector.calibration.confidence == 1.0


def test_calibration_not_reset_on_blink() -> None:
    """Blink detection should not reset calibration state."""
    detector = BlinkDetector(
        ear_close_threshold=0.25,
        min_blink_frames=3,
        recovery_frames=1,
        use_adaptive_ear=True,
    )
    now = _TimeCounter()
    # Calibrate
    for _ in range(40):
        detector.update(0.8, 0.8, now=now())
    assert detector.calibration.calibration_complete
    samples_before = detector.calibration.open_ear_samples

    # Fire a blink
    for _ in range(3):
        detector.update(0.1, 0.1, now=now())
    detector.update(0.8, 0.8, now=now())

    # Calibration should still be intact
    assert detector.calibration.calibration_complete
    assert detector.calibration.open_ear_samples >= samples_before
