"""Multimodal fusion: combines gaze, gesture and (eventually) voice into actions."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..gaze.blink_detector import BlinkEventType
from ..gaze.head_pose import head_pose_to_screen_delta
from ..profiles import Profile
from .state_machine import State, StateMachine


DIRECT_POINTER_ACTIONS = {"click", "right_click", "double_click", "undo"}

# Blink is an instant click — no dwell needed.
BLINK_CLICK_ACTION = "click"

# Palm-scroll gestures map directly to scroll actions (no dwell needed).
PALM_SCROLL_ACTIONS = {
    "palm_scroll_up": "scroll_up",
    "palm_scroll_down": "scroll_down",
    "left_palm_scroll_up": "scroll_up",
    "left_palm_scroll_down": "scroll_down",
    "right_palm_scroll_up": "scroll_up",
    "right_palm_scroll_down": "scroll_down",
}

# Air-scroll / swipe gestures map directly to scroll actions (no dwell needed).
# Unlike palm-scroll, air-scroll works with any hand pose (pointing, fist, etc.)
# — the user simply sweeps their hand up or down in the camera frame.
AIR_SCROLL_ACTIONS = {
    "air_scroll_up": "scroll_up",
    "air_scroll_down": "scroll_down",
    "left_air_scroll_up": "scroll_up",
    "left_air_scroll_down": "scroll_down",
    "right_air_scroll_up": "scroll_up",
    "right_air_scroll_down": "scroll_down",
}

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
    blink: bool = False           # True if a blink was detected this frame
    blink_event: BlinkEventType | None = None  # Type of blink event (SINGLE, DOUBLE, PROLONGED)
    voice_action: str | None = None  # Voice command action (for AND fusion)
    gaze_confirmed: bool = False   # True when AND fusion requires gaze+voice
    head_pose_active: bool = False # True when head-pose coarse movement is active
    head_coarse_dx: float = 0.0    # Horizontal coarse displacement (screen fraction)
    head_coarse_dy: float = 0.0    # Vertical coarse displacement (screen fraction)


# Actions that benefit from gaze confirmation (AND fusion).
# When a voice command proposes one of these, it only fires if gaze
# is also present and stable — requiring the user to look at the target.
AND_FUSION_ACTIONS: frozenset[str] = frozenset({
    "click", "right_click", "double_click", "undo",
    "zoom_in", "zoom_out", "scroll_up", "scroll_down",
})

# Actions that are inherently instant (gesture-based, no dwell needed).
# Includes both palm-scroll and air-scroll gestures.
INSTANT_SCROLL_ACTIONS = frozenset({
    "scroll_up", "scroll_down",
})


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

    def peek(self) -> bool:
        """Return the last stability result without adding a new sample."""
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

    def step_and_voice(
        self,
        cursor_xy: tuple[int, int] | None,
        confirmed_gesture: str | None,
        voice_action: str | None,
        blink: bool = False,
        blink_event: BlinkEventType | None = None,
        head_pose: "HeadPose | None" = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> FusionResult:
        """Run the full fusion pipeline including voice+gaze AND fusion.

        This is the main entry point used by main.py.  It first processes
        gesture + blink logic via ``step()``, then applies the multimodal
        AND fusion: if a voice action targets a pointer/gesture action
        (click, zoom, scroll, …) the action only fires when gaze is also
        present and stable — requiring the user to look at the target.

        Returns a ``FusionResult`` with ``voice_action`` and
        ``gaze_confirmed`` populated so the caller can display the fusion
        state in the overlay.
        """
        # ── 1. Process gesture + blink through the existing pipeline ──────
        result = self.step(
            cursor_xy, confirmed_gesture,
            blink=blink, blink_event=blink_event,
            head_pose=head_pose,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        # ── 1b. Apply head-pose coarse displacement to cursor ────────────
        if head_pose is not None and head_pose.coarse_active:
            dx, dy = head_pose_to_screen_delta(
                head_pose, screen_width, screen_height
            )
            if result.cursor_xy is not None:
                result = FusionResult(
                    cursor_xy=(
                        result.cursor_xy[0] + dx,
                        result.cursor_xy[1] + dy,
                    ),
                    state=result.state,
                    dwell_progress=result.dwell_progress,
                    fired_action=result.fired_action,
                    blink=result.blink,
                    blink_event=result.blink_event,
                    voice_action=result.voice_action,
                    gaze_confirmed=result.gaze_confirmed,
                    head_pose_active=True,
                    head_coarse_dx=head_pose.coarse_dx,
                    head_coarse_dy=head_pose.coarse_dy,
                )
            else:
                result = FusionResult(
                    cursor_xy=None,
                    state=result.state,
                    dwell_progress=result.dwell_progress,
                    fired_action=result.fired_action,
                    blink=result.blink,
                    blink_event=result.blink_event,
                    voice_action=result.voice_action,
                    gaze_confirmed=result.gaze_confirmed,
                    head_pose_active=True,
                    head_coarse_dx=head_pose.coarse_dx,
                    head_coarse_dy=head_pose.coarse_dy,
                )

        # ── 2. Voice+gaze AND fusion (only for non-blink, non-gesture actions) ─
        # Blink events and gesture-mapped actions are already handled above.
        # Voice commands for pointer/gesture actions need gaze confirmation.
        if voice_action and not result.fired_action:
            if voice_action in AND_FUSION_ACTIONS:
                # AND fusion: voice alone is not enough — need gaze too.
                if cursor_xy is not None and self.sm.state != State.IDLE:
                    # Gaze present and system active — check stability.
                    gaze_ok = self.gaze_stable.peek()
                    if gaze_ok:
                        # Both voice AND gaze present → fire the action.
                        self._last_action_at = time.monotonic()
                        self.sm.trigger_cooldown()
                        return FusionResult(
                            cursor_xy=result.cursor_xy,
                            state=self.sm.state,
                            dwell_progress=0.0,
                            fired_action=voice_action,
                            blink=False,
                            blink_event=None,
                            voice_action=voice_action,
                            gaze_confirmed=True,
                        )
                    else:
                        # Gaze present but unstable → return partial result.
                        return FusionResult(
                            cursor_xy=result.cursor_xy,
                            state=self.sm.state,
                            dwell_progress=self.sm.dwell_progress,
                            fired_action=None,
                            blink=False,
                            blink_event=None,
                            voice_action=voice_action,
                            gaze_confirmed=False,
                        )
                else:
                    # No gaze or idle → voice alone cannot fire pointer actions.
                    return FusionResult(
                        cursor_xy=result.cursor_xy,
                        state=self.sm.state,
                        dwell_progress=self.sm.dwell_progress,
                        fired_action=None,
                        blink=False,
                        blink_event=None,
                        voice_action=voice_action,
                        gaze_confirmed=False,
                    )
            else:
                # Voice action is NOT in AND_FUSION_ACTIONS (e.g. drag_start).
                # Fire it directly (no gaze confirmation needed).
                self._last_action_at = time.monotonic()
                return FusionResult(
                    cursor_xy=result.cursor_xy,
                    state=self.sm.state,
                    dwell_progress=0.0,
                    fired_action=voice_action,
                    blink=False,
                    blink_event=None,
                    voice_action=voice_action,
                    gaze_confirmed=True,
                )

        # ── 3. If gesture already fired an action, propagate voice info ───
        if result.fired_action:
            result.voice_action = voice_action
            result.gaze_confirmed = True

        return result

    def step(
        self,
        cursor_xy: tuple[int, int] | None,
        confirmed_gesture: str | None,
        blink: bool = False,
        blink_event: BlinkEventType | None = None,
        head_pose: "HeadPose | None" = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> FusionResult:
        # ── Apply head-pose coarse displacement to cursor ─────────────────
        # Head-pose moves the cursor by large amounts (screen fractions).
        # This is applied before the state machine so dwell is measured on
        # the displaced position — the user looks at a target by turning
        # their head, and the fine gaze pointer handles precision.
        if head_pose is not None and head_pose.coarse_active:
            dx, dy = head_pose_to_screen_delta(
                head_pose, screen_width, screen_height
            )
            if cursor_xy is not None:
                cursor_xy = (
                    cursor_xy[0] + dx,
                    cursor_xy[1] + dy,
                )
        bindings = self.profile.gesture_bindings
        candidate_action = action_for_gesture(bindings, confirmed_gesture)

        # Blink events: type-specific actions, bypasses dwell and state machine.
        if blink and blink_event is not None:
            self._last_action_at = time.monotonic()

            if blink_event == BlinkEventType.DOUBLE:
                # Double blink → click (same as single, but distinct event type)
                return FusionResult(cursor_xy, self.sm.state, 0.0, BLINK_CLICK_ACTION, blink=True, blink_event=blink_event)

            if blink_event == BlinkEventType.PROLONGED:
                # Prolonged close → drag start
                return FusionResult(cursor_xy, self.sm.state, 0.0, "drag_start", blink=True, blink_event=blink_event)

            # Single blink → click (existing behavior)
            return FusionResult(cursor_xy, self.sm.state, 0.0, BLINK_CLICK_ACTION, blink=True, blink_event=blink_event)

        # Blink: instant click, bypasses dwell and state machine entirely.
        if blink:
            self._last_action_at = time.monotonic()
            return FusionResult(cursor_xy, self.sm.state, 0.0, BLINK_CLICK_ACTION, blink=True)

        # Palm-scroll gestures fire immediately (no dwell, no state machine).
        if confirmed_gesture and confirmed_gesture in PALM_SCROLL_ACTIONS:
            scroll_action = PALM_SCROLL_ACTIONS[confirmed_gesture]
            self._last_action_at = time.monotonic()
            return FusionResult(cursor_xy, self.sm.state, 0.0, scroll_action, blink=False)

        # Air-scroll (swipe) gestures fire immediately (no dwell, no state machine).
        # These work with any hand pose — pointing, fist, open palm, etc.
        if confirmed_gesture and confirmed_gesture in AIR_SCROLL_ACTIONS:
            scroll_action = AIR_SCROLL_ACTIONS[confirmed_gesture]
            self._last_action_at = time.monotonic()
            return FusionResult(cursor_xy, self.sm.state, 0.0, scroll_action, blink=False)

        # Any gesture explicitly mapped to toggle_pause is honoured in any state.
        if confirmed_gesture and candidate_action == "toggle_pause":
            if self.sm.state == State.IDLE:
                self.sm.activate()
                return FusionResult(cursor_xy, self.sm.state, 0.0, "resume", blink=False)
            self.sm.pause()
            return FusionResult(cursor_xy, self.sm.state, 0.0, "toggle_pause", blink=False)

        if self.sm.state == State.IDLE:
            return FusionResult(cursor_xy, self.sm.state, 0.0, None, blink=False)

        gaze_is_stable = self.gaze_stable.update(cursor_xy)
        self.sm.tick(gaze_is_stable)

        if not candidate_action or not confirmed_gesture or confirmed_gesture == "none":
            return FusionResult(cursor_xy, self.sm.state, self.sm.dwell_progress, None, blink=False)

        # Direct pointer actions: every confirmed click-family gesture fires
        # immediately, with no cooldown and no contradiction throttling.
        if (
            self.profile.pointer_control_enabled
            and candidate_action in DIRECT_POINTER_ACTIONS
        ):
            self._last_action_at = time.monotonic()
            return FusionResult(cursor_xy, self.sm.state, 0.0, candidate_action, blink=False)

        # Other actions (zoom, scroll, escape) still need dwell confirmation.
        if self.sm.state == State.CONFIRMING:
            self._last_action_at = time.monotonic()
            self.sm.trigger_cooldown()
            return FusionResult(cursor_xy, self.sm.state, 0.0, candidate_action, blink=False)

        return FusionResult(
            cursor_xy=cursor_xy,
            state=self.sm.state,
            dwell_progress=self.sm.dwell_progress,
            fired_action=None,
            blink=False,
        )
