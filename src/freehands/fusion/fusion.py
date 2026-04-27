"""Multimodal fusion: combines gaze, gesture and (eventually) voice into actions."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..profiles import Profile
from .state_machine import State, StateMachine


DIRECT_POINTER_ACTIONS = {"click", "right_click", "double_click", "undo"}
SIDE_BINDING_FALLBACKS = {
    "left_pointing_up": "pointing_up",
    "right_pointing_up": "pointing_up",
    "left_middle_up": "middle_up",
    "right_middle_up": "middle_up",
    "left_two_fingers_up": "two_fingers_up",
    "right_two_fingers_up": "two_fingers_up",
}


def action_for_gesture(bindings: dict[str, str], gesture: str | None) -> str | None:
    if not gesture:
        return None
    action = bindings.get(gesture)
    if action:
        return action
    fallback = SIDE_BINDING_FALLBACKS.get(gesture)
    return bindings.get(fallback) if fallback else action


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

    def step(
        self,
        cursor_xy: tuple[int, int] | None,
        confirmed_gesture: str | None,
    ) -> FusionResult:
        bindings = self.profile.gesture_bindings
        candidate_action = action_for_gesture(bindings, confirmed_gesture)

        # Any gesture explicitly mapped to toggle_pause is honoured in any state.
        if confirmed_gesture and candidate_action == "toggle_pause":
            if self.sm.state == State.IDLE:
                self.sm.activate()
                return FusionResult(cursor_xy, self.sm.state, 0.0, "resume")
            self.sm.pause()
            return FusionResult(cursor_xy, self.sm.state, 0.0, "toggle_pause")

        if self.sm.state == State.IDLE:
            return FusionResult(cursor_xy, self.sm.state, 0.0, None)

        gaze_is_stable = self.gaze_stable.update(cursor_xy)
        self.sm.tick(gaze_is_stable)

        if not candidate_action or not confirmed_gesture or confirmed_gesture == "none":
            return FusionResult(cursor_xy, self.sm.state, self.sm.dwell_progress, None)

        # Direct pointer actions: every confirmed click-family gesture fires
        # immediately, with no cooldown and no contradiction throttling.
        if (
            self.profile.pointer_control_enabled
            and candidate_action in DIRECT_POINTER_ACTIONS
        ):
            self._last_action_at = time.monotonic()
            return FusionResult(cursor_xy, self.sm.state, 0.0, candidate_action)

        # Other actions (zoom, scroll, escape) still need dwell confirmation.
        if self.sm.state == State.CONFIRMING:
            self._last_action_at = time.monotonic()
            self.sm.trigger_cooldown()
            return FusionResult(cursor_xy, self.sm.state, 0.0, candidate_action)

        return FusionResult(
            cursor_xy=cursor_xy,
            state=self.sm.state,
            dwell_progress=self.sm.dwell_progress,
            fired_action=None,
        )
