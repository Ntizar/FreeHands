"""Tests for multimodal dictation intent detection (improvement #20)."""
from __future__ import annotations

from unittest.mock import MagicMock

from freehands.voice.dictation_intent import (
    DictationIntentDetector,
    DictationIntentState,
)


# ── Helpers ─────────────────────────────────────────────────────────

def _make_fake_region(x: int = 100, y: int = 100, width: int = 200, height: int = 50):
    """Create a fake TextRegion for testing."""
    region = MagicMock()
    region.x = x
    region.y = y
    region.width = width
    region.height = height
    return region


# ── DictationIntentState tests ──────────────────────────────────────

def test_default_state() -> None:
    state = DictationIntentState()
    assert state.gazing_at_text is False
    assert state.hovered_region is None
    assert state.dwell_progress == 0.0
    assert state.ready_to_activate is False
    assert state.ready_at == 0.0


# ── DictationIntentDetector basic tests ─────────────────────────────

def test_detector_initial_state() -> None:
    detector = DictationIntentDetector()
    assert detector.gazing_at_text is False
    assert detector.ready_to_activate is False
    assert detector.hovered_region is None


def test_detector_reset_clears_state() -> None:
    detector = DictationIntentDetector()
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region()]
    detector.update_dwell(
        (200, 125), regions, 400, 300  # cursor inside region
    )
    # After one call with cursor inside region, should be gazing
    assert detector.gazing_at_text is True
    detector.reset()
    assert detector.gazing_at_text is False
    assert detector.ready_to_activate is False
    assert detector.hovered_region is None


def test_detector_no_cursor_clears_state() -> None:
    detector = DictationIntentDetector()
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region()]
    detector.update_dwell(
        (200, 125), regions, 400, 300  # cursor inside region
    )
    assert detector.gazing_at_text is True
    detector.update_dwell(None, regions, 400, 300)
    assert detector.gazing_at_text is False
    assert detector.ready_to_activate is False


def test_detector_cursor_outside_region() -> None:
    detector = DictationIntentDetector()
    regions = [_make_fake_region(x=100, y=100, width=50, height=50)]
    # Cursor at (500, 500) is outside the region
    detector.update_dwell(
        (500, 500), regions, 400, 300
    )
    assert detector.gazing_at_text is False
    assert detector.ready_to_activate is False


# ── Dwell progression tests ─────────────────────────────────────────

def test_dwell_progress_increases_over_time() -> None:
    """Dwell progress should increase with each update while gazing at region."""
    detector = DictationIntentDetector(dwell_ms=100)  # very short for testing
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region(x=100, y=100, width=200, height=50)]

    # Cursor at (200, 125) is inside the region
    for _ in range(10):
        detector.update_dwell((200, 125), regions, 400, 300)

    # After 10 steps of 16ms each = 160ms > 100ms dwell threshold
    assert detector.gazing_at_text is True
    assert detector.ready_to_activate is True


def test_dwell_resets_when_cursor_leaves() -> None:
    """Dwell progress should reset when cursor leaves the region."""
    detector = DictationIntentDetector(dwell_ms=100)
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region(x=100, y=100, width=200, height=50)]

    # Move cursor into region
    detector.update_dwell((200, 125), regions, 400, 300)
    detector.update_dwell((200, 125), regions, 400, 300)
    assert detector.gazing_at_text is True
    assert detector._state.dwell_progress > 0

    # Move cursor outside
    detector.update_dwell((999, 999), regions, 400, 300)
    assert detector.gazing_at_text is False
    assert detector._state.dwell_progress == 0.0


def test_ready_to_activate_after_dwell() -> None:
    """Should become ready_to_activate after dwell_ms of continuous gaze."""
    detector = DictationIntentDetector(dwell_ms=50)
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region(x=100, y=100, width=200, height=50)]

    # Simulate 50ms / 16ms ≈ 4 frames of dwell
    for _ in range(5):
        detector.update_dwell((200, 125), regions, 400, 300)

    assert detector.ready_to_activate is True


# ── consume_ready tests ─────────────────────────────────────────────

def test_consume_ready_returns_true_when_ready() -> None:
    detector = DictationIntentDetector(dwell_ms=50)
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region(x=100, y=100, width=200, height=50)]

    for _ in range(5):
        detector.update_dwell((200, 125), regions, 400, 300)

    assert detector.ready_to_activate is True
    assert detector.consume_ready() is True
    assert detector.ready_to_activate is False


def test_consume_ready_returns_false_when_not_ready() -> None:
    detector = DictationIntentDetector()
    regions = [_make_fake_region()]
    detector.update_dwell((999, 999), regions, 400, 300)
    assert detector.consume_ready() is False
    assert detector.ready_to_activate is False


# ── Hit-test tests ──────────────────────────────────────────────────

def test_hit_test_finds_region() -> None:
    detector = DictationIntentDetector()
    region = _make_fake_region(x=100, y=100, width=200, height=50)
    # Region at screen coords: (100,100) to (300,150)
    # Cursor at (200, 125) should be inside
    result = detector._hit_test((200, 125), 400, 300, [region])
    assert result is region


def test_hit_test_outside_region() -> None:
    detector = DictationIntentDetector()
    region = _make_fake_region(x=100, y=100, width=50, height=50)
    # Region at screen coords: (300, 250) to (350, 300)
    # Cursor at (500, 500) should be outside
    result = detector._hit_test((500, 500), 400, 300, [region])
    assert result is None


def test_hit_test_multiple_regions() -> None:
    detector = DictationIntentDetector()
    # region1: x=100, y=100, w=100, h=50 → screen: (100,100)-(200,150)
    region1 = _make_fake_region(x=100, y=100, width=100, height=50)
    # region2: x=200, y=100, w=100, h=50 → screen: (200,100)-(300,150)
    region2 = _make_fake_region(x=200, y=100, width=100, height=50)
    # Cursor at (150, 125) should hit region1
    result = detector._hit_test((150, 125), 400, 300, [region1, region2])
    assert result is region1
    # Cursor at (250, 125) should hit region2
    result = detector._hit_test((250, 125), 400, 300, [region1, region2])
    assert result is region2


# ── Gaze debounce tests ─────────────────────────────────────────────

def test_gaze_lost_debounce_resets_ready() -> None:
    """After gaze is lost, ready should be reset after debounce window."""
    detector = DictationIntentDetector(dwell_ms=50)
    # Region at x=100, y=100, w=200, h=50 → screen: (100,100)-(300,150)
    regions = [_make_fake_region(x=100, y=100, width=200, height=50)]

    # Get ready
    for _ in range(5):
        detector.update_dwell((200, 125), regions, 400, 300)
    assert detector.ready_to_activate is True

    # Lose gaze
    detector.update_dwell((999, 999), regions, 400, 300)
    # After debounce (300ms), ready should be reset
    # Since we call update_dwell with no cursor, it resets immediately
    assert detector.ready_to_activate is False


# ── Integration with main.py pattern tests ──────────────────────────

def test_detector_works_with_text_selector_regions() -> None:
    """Simulate the main.py pattern: update_dwell with gaze_text_sel regions."""
    detector = DictationIntentDetector(dwell_ms=500)
    # Simulate text regions from OCR detector
    # Region coords are relative to the overlay centre (400, 300)
    regions = [
        _make_fake_region(x=50, y=100, width=300, height=40),   # screen: (100,100)-(400,140)
        _make_fake_region(x=50, y=200, width=250, height=40),   # screen: (100,200)-(350,240)
        _make_fake_region(x=50, y=300, width=400, height=40),   # screen: (100,300)-(500,340)
    ]
    centre_x, centre_y = 400, 300

    # Cursor on first region (screen: 100-400, 100-140)
    detector.update_dwell((200, 120), regions, centre_x, centre_y)
    assert detector.gazing_at_text is True
    assert detector.hovered_region is regions[0]

    # Move to second region (screen: 100-350, 200-240)
    detector.update_dwell((200, 220), regions, centre_x, centre_y)
    assert detector.hovered_region is regions[1]

    # Move to third region (screen: 100-500, 300-340)
    detector.update_dwell((300, 320), regions, centre_x, centre_y)
    assert detector.hovered_region is regions[2]

    # Leave all regions
    detector.update_dwell((999, 999), regions, centre_x, centre_y)
    assert detector.gazing_at_text is False
