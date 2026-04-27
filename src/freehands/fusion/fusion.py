"""Multimodal fusion: combines gaze, gesture and (eventually) voice into actions."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..profiles import Profile
from .state_machine import State, StateMachine


DIRECT_POINTER_ACTIONS = {"click", "right_click", "double_click"}


@dataclass
class FusionResult:
    cursor_xy: tuple[int, int] | None
    state: State
    dwell_progress: float
    fired_action: str | None     # binding name (e.g. 'click', 'zoom_in', 'toggle_pause')


class GazeStabilityChecker:
    """Considers gaze 'stable' when its variance over the recent window is low."""

    def __init__(self, window: int = 8, max_std_px: float = 35.0) -> None:
        self._buf: deque[tuple[int, int]] = deque(maxlen=window)
        self._max_std = max_std_px

    def update(self, xy: tuple[int, int] | None) -> bool:
        if xy is None:
            self._buf.clear()
            return False
        self._buf.append(xy)
        if len(self._buf) < self._buf.maxlen:
            return False
        arr = np.array(self._buf)
        return float(arr.std(axis=0).max()) < self._max_std


class MultimodalFusion:
    """Owns the state machine, dwell logic and binding lookup."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.sm = StateMachine(dwell_ms=profile.dwell_time_ms)
        self.gaze_stable = GazeStabilityChecker()
        self._last_action_at = 0.0
        self._contradiction_buf: deque[tuple[float, str]] = deque(maxlen=6)

    def step(
        self,
        cursor_xy: tuple[int, int] | None,
        confirmed_gesture: str | None,
    ) -> FusionResult:
        # The pause gesture is honoured in any state — anti-FP layer 6.
        if confirmed_gesture == "fist_pause":
            if self.sm.state == State.IDLE:
                self.sm.activate()
                return FusionResult(cursor_xy, self.sm.state, 0.0, "resume")
            self.sm.pause()
            return FusionResult(cursor_xy, self.sm.state, 0.0, "toggle_pause")

        if self.sm.state == State.IDLE:
            return FusionResult(cursor_xy, self.sm.state, 0.0, None)

        gaze_is_stable = self.gaze_stable.update(cursor_xy)
        self.sm.tick(gaze_is_stable)

        action: str | None = None

        # Anti-FP layer 5: contradictory gestures within a short window
        # → trigger an extended cooldown.
        if confirmed_gesture and confirmed_gesture != "none":
            now = time.monotonic()
            self._contradiction_buf.append((now, confirmed_gesture))
            recent = [g for t, g in self._contradiction_buf if now - t < 2.0]
            if len(set(recent)) >= 3:
                self.sm.trigger_cooldown()
                self._contradiction_buf.clear()
                return FusionResult(cursor_xy, self.sm.state, 0.0, None)

            bindings = self.profile.gesture_bindings
            candidate_action = bindings.get(confirmed_gesture)
            if (
                self.profile.pointer_control_enabled
                and candidate_action in DIRECT_POINTER_ACTIONS
                and self.sm.state in {State.ACTIVE, State.CONFIRMING}
            ):
                self._last_action_at = now
                self.sm.trigger_cooldown()
                return FusionResult(cursor_xy, self.sm.state, 0.0, candidate_action)

            if self.sm.state == State.CONFIRMING:
                if candidate_action:
                    action = candidate_action
                    self._last_action_at = now
                    self.sm.trigger_cooldown()

        return FusionResult(
            cursor_xy=cursor_xy,
            state=self.sm.state,
            dwell_progress=self.sm.dwell_progress,
            fired_action=action,
        )
