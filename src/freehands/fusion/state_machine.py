"""Anti-false-positive state machine (layer 1)."""
from __future__ import annotations

import time
from enum import Enum, auto


class State(Enum):
    IDLE       = auto()   # system paused — only listens for activation
    ACTIVE     = auto()   # listens for gestures
    CONFIRMING = auto()   # dwell satisfied, waiting for gesture confirmation
    COOLDOWN   = auto()   # post-action lockout


class StateMachine:
    def __init__(self, cooldown_ms: int = 500, dwell_ms: int = 600) -> None:
        self.state = State.IDLE
        self._cooldown_ms = cooldown_ms
        self._dwell_ms = dwell_ms
        self._state_entered_at = time.monotonic()
        self._dwell_start: float | None = None

    # ── transitions ───────────────────────────────────────────────────────
    def _set(self, new: State) -> None:
        self.state = new
        self._state_entered_at = time.monotonic()
        if new != State.CONFIRMING:
            self._dwell_start = None

    def activate(self) -> None:
        if self.state == State.IDLE:
            self._set(State.ACTIVE)

    def pause(self) -> None:
        self._set(State.IDLE)

    def trigger_cooldown(self) -> None:
        self._set(State.COOLDOWN)

    # ── per-frame tick ────────────────────────────────────────────────────
    def tick(self, gaze_stable: bool) -> None:
        now = time.monotonic()

        if self.state == State.COOLDOWN:
            if (now - self._state_entered_at) * 1000 >= self._cooldown_ms:
                self._set(State.ACTIVE)
            return

        if self.state == State.ACTIVE:
            if gaze_stable:
                self._dwell_start = self._dwell_start or now
                if (now - self._dwell_start) * 1000 >= self._dwell_ms:
                    self._set(State.CONFIRMING)
            else:
                self._dwell_start = None
            return

        if self.state == State.CONFIRMING:
            if not gaze_stable:
                # if user looked away, drop back to ACTIVE (cancels intent)
                self._set(State.ACTIVE)

    @property
    def dwell_progress(self) -> float:
        """0..1 fill ratio for the dwell ring overlay."""
        if self.state != State.ACTIVE or self._dwell_start is None:
            return 1.0 if self.state == State.CONFIRMING else 0.0
        elapsed = (time.monotonic() - self._dwell_start) * 1000
        return max(0.0, min(1.0, elapsed / self._dwell_ms))
