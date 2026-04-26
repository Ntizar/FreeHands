"""Tests for the anti-FP state machine."""
from __future__ import annotations

import time

from freehands.fusion.state_machine import State, StateMachine


def test_starts_idle():
    sm = StateMachine()
    assert sm.state is State.IDLE


def test_activate_then_dwell_to_confirming():
    sm = StateMachine(dwell_ms=50)
    sm.activate()
    assert sm.state is State.ACTIVE
    sm.tick(gaze_stable=True)
    time.sleep(0.09)
    sm.tick(gaze_stable=True)
    assert sm.state is State.CONFIRMING


def test_cooldown_returns_to_active():
    sm = StateMachine(cooldown_ms=20)
    sm.activate()
    sm.trigger_cooldown()
    assert sm.state is State.COOLDOWN
    time.sleep(0.05)
    sm.tick(gaze_stable=False)
    assert sm.state is State.ACTIVE
