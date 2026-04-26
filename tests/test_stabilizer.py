"""Unit tests for the gesture stabiliser (anti-FP layer 2)."""
from __future__ import annotations

from freehands.gestures import GestureStabilizer


def test_emits_after_required_frames():
    s = GestureStabilizer(required_frames=4, confidence_min=0.8)
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) == "thumb_up"


def test_does_not_emit_on_mixed_gestures():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("thumb_up", 0.9)
    s.update("thumb_down", 0.9)
    assert s.update("thumb_up", 0.9) is None


def test_does_not_emit_below_confidence():
    s = GestureStabilizer(required_frames=3, confidence_min=0.95)
    s.update("thumb_up", 0.6)
    s.update("thumb_up", 0.6)
    assert s.update("thumb_up", 0.6) is None


def test_edge_trigger_no_repeat_until_change():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("thumb_up", 0.9); s.update("thumb_up", 0.9)
    assert s.update("thumb_up", 0.9) == "thumb_up"
    # Holding the same gesture should not re-emit
    assert s.update("thumb_up", 0.9) is None
    assert s.update("thumb_up", 0.9) is None


def test_none_never_emits():
    s = GestureStabilizer(required_frames=3, confidence_min=0.5)
    s.update("none", 1.0); s.update("none", 1.0)
    assert s.update("none", 1.0) is None
