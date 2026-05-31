"""Runtime orchestrator: capture → trackers → fusion → action + overlay."""
from __future__ import annotations

import sys
import time
from collections import deque

from PyQt6 import QtCore, QtWidgets

from .actions import ActionDispatcher
from .capture import Camera
from .config import (
    DEFAULT_GESTURE_CONFIDENCE,
    DEFAULT_STABILITY_FRAMES,
    POINTER_FINE_AIM_ALPHA,
    POINTER_FINE_AIM_HOLD_MS,
    POINTER_FINE_AIM_RADIUS_PX,
    POINTER_FINE_AIM_RELEASE_PX,
    POINTER_MOVE_INTERVAL_MS,
    POINTER_MOVE_MIN_DELTA_PX,
    TARGET_FPS,
)
from .fusion import MultimodalFusion, State, action_for_gesture, decide_channel_priority
from .gaze import GazeRegressor, GazeTracker, gaze_model_is_usable
from .gaze.dead_zones import DeadZoneClamper
from .gaze.snap_to_grid import SnapToGrid
from .gestures import GestureStabilizer, HandTracker, VolumeControl
from .gestures.face_tracker import FaceTracker, FacialObservation
from .profiles import GestureThreshold, load_profile, save_profile
from .plugins import PluginLoader
from .ui.audio_feedback import AudioFeedback
from .ui.overlay import FreeHandsControlPanel, GazeOverlay
from .ui.radial_menu import MENU_OPEN_DURATION_MS, RadialMenuWidget
from .ui.virtual_keyboard import VirtualKeyboardWidget
from .voice import VoiceListener


OPEN_PALM_HOLD_GESTURES = {"left_open_palm", "right_open_palm"}


class FineAimPointer:
    def __init__(self) -> None:
        self._hold_seconds = POINTER_FINE_AIM_HOLD_MS / 1000
        self._samples: deque[tuple[float, tuple[int, int]]] = deque(
            maxlen=TARGET_FPS * 2,
        )
        self._anchor: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self._anchor is not None

    def reset(self) -> None:
        self._samples.clear()
        self._anchor = None

    def update(self, cursor_xy: tuple[int, int]) -> tuple[int, int]:
        now = time.monotonic()
        self._samples.append((now, cursor_xy))
        while self._samples and now - self._samples[0][0] > self._hold_seconds * 2:
            self._samples.popleft()

        if self._anchor is None:
            stable_samples = self._stable_samples(now)
            if not self._has_stable_zone(now, stable_samples):
                return cursor_xy
            self._anchor = self._centroid(stable_samples)
            return self._rounded_anchor()

        if self._distance_squared(cursor_xy, self._anchor) > POINTER_FINE_AIM_RELEASE_PX ** 2:
            self.reset()
            self._samples.append((now, cursor_xy))
            return cursor_xy

        target = self._centroid(self._stable_samples(now) or list(self._samples))
        anchor_x, anchor_y = self._anchor
        target_x, target_y = target
        self._anchor = (
            anchor_x + (target_x - anchor_x) * POINTER_FINE_AIM_ALPHA,
            anchor_y + (target_y - anchor_y) * POINTER_FINE_AIM_ALPHA,
        )
        return self._rounded_anchor()

    def _stable_samples(self, now: float) -> list[tuple[float, tuple[int, int]]]:
        return [
            (sample_at, sample)
            for sample_at, sample in self._samples
            if now - sample_at <= self._hold_seconds
        ]

    def _has_stable_zone(self, now: float, samples: list[tuple[float, tuple[int, int]]]) -> bool:
        min_samples = max(5, int(TARGET_FPS * self._hold_seconds * 0.75))
        if len(samples) < min_samples or now - samples[0][0] < self._hold_seconds * 0.95:
            return False
        center = self._centroid(samples)
        radius_squared = POINTER_FINE_AIM_RADIUS_PX ** 2
        return all(self._distance_squared(sample, center) <= radius_squared for _, sample in samples)

    def _centroid(self, samples: list[tuple[float, tuple[int, int]]]) -> tuple[float, float]:
        count = len(samples)
        center_x = sum(sample[0] for _, sample in samples) / count
        center_y = sum(sample[1] for _, sample in samples) / count
        return center_x, center_y

    def _rounded_anchor(self) -> tuple[int, int]:
        if self._anchor is None:
            raise RuntimeError("fine aim anchor is not set")
        return round(self._anchor[0]), round(self._anchor[1])

    @staticmethod
    def _distance_squared(
        a: tuple[int, int] | tuple[float, float],
        b: tuple[int, int] | tuple[float, float],
    ) -> float:
        delta_x = a[0] - b[0]
        delta_y = a[1] - b[1]
        return delta_x * delta_x + delta_y * delta_y


class PauseHoldGate:
    """Leaky hold gate for whichever open palm is mapped to toggle_pause."""

    def __init__(self) -> None:
        self._score = 0.0
        self._hold_fired = False
        self._gesture: str | None = None
        self._release_frames = 0

    def update(
        self,
        observed_gesture: str,
        bindings: dict[str, str],
        thresholds: dict[str, GestureThreshold],
    ) -> tuple[str | None, float]:
        toggle_gesture = self._toggle_pause_gesture(bindings, observed_gesture)
        if toggle_gesture is not None:
            self._release_frames = 0
            if self._gesture != toggle_gesture:
                self._gesture = toggle_gesture
                self._score = 0.0
                self._hold_fired = False
            required_frames = self._required_frames(thresholds, toggle_gesture)
            if self._hold_fired:
                self._score = 0.0
            else:
                self._score = min(required_frames + 6.0, self._score + 1.0)
        elif self._hold_fired:
            required_frames = self._required_frames(thresholds, self._gesture)
            self._release_frames += 1
            self._score = 0.0
            if self._release_frames >= 2:
                self._hold_fired = False
                self._gesture = None
                self._release_frames = 0
        elif observed_gesture == "none":
            required_frames = self._required_frames(thresholds, self._gesture)
            self._score = max(0.0, self._score - 0.5)
        else:
            required_frames = self._required_frames(thresholds, self._gesture)
            self._score = max(0.0, self._score - 2.0)

        if self._score <= 0.0 and not self._hold_fired:
            self._gesture = None

        progress = min(1.0, self._score / required_frames) if required_frames > 0 else 0.0
        confirmed = None
        if self._gesture is not None and self._score >= required_frames and not self._hold_fired:
            confirmed = self._gesture
            self._hold_fired = True
            self._score = 0.0
            self._release_frames = 0
            progress = 0.0
        return confirmed, progress

    @staticmethod
    def _toggle_pause_gesture(bindings: dict[str, str], gesture: str) -> str | None:
        if gesture not in OPEN_PALM_HOLD_GESTURES:
            return None
        return gesture if action_for_gesture(bindings, gesture) == "toggle_pause" else None

    @staticmethod
    def _required_frames(thresholds: dict[str, GestureThreshold], gesture: str | None) -> int:
        threshold = thresholds.get(gesture or "right_open_palm")
        return threshold.stability_frames if threshold is not None else 60


def suppress_open_palm_toggle_without_hold(
    confirmed_gesture: str | None,
    bindings: dict[str, str],
) -> str | None:
    if confirmed_gesture not in OPEN_PALM_HOLD_GESTURES:
        return confirmed_gesture
    if action_for_gesture(bindings, confirmed_gesture) == "toggle_pause":
        return None
    return confirmed_gesture


def run_system(user_id: str, voice_enabled: bool = True) -> int:
    # ── Auto-onboarding: if no profile or no gaze model, calibrate first ──
    from .profiles.store import profile_path
    from .ui.calibration_game import run_calibration, run_gaze_calibration

    calibration_mode: str | None = None
    if not profile_path(user_id).exists():
        print(f"[FreeHands] No profile exists for '{user_id}'. Starting calibration...")
        calibration_mode = "full"
    else:
        try:
            tmp = load_profile(user_id)
            if not tmp.gaze_model.weights_x or not gaze_model_is_usable(tmp.gaze_model):
                print(f"[FreeHands] Profile '{user_id}' does not have a valid gaze model. Starting gaze recalibration...")
                calibration_mode = "gaze"
        except Exception as e:
            print(f"[FreeHands] Could not read the profile ({e}). Starting calibration...")
            calibration_mode = "full"

    if calibration_mode is not None:
        rc = run_calibration(user_id=user_id) if calibration_mode == "full" else run_gaze_calibration(user_id=user_id)
        if rc != 0:
            print("[FreeHands] Calibration was cancelled or failed. Exiting.")
            return rc

    profile = load_profile(user_id)
    if not profile.gaze_model.weights_x or not gaze_model_is_usable(profile.gaze_model):
        print(f"[FreeHands] There is still no gaze model for '{user_id}'. "
              f"Run calibration again.")
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen().geometry()

    try:
        camera = Camera(profile.camera_index).start()
    except Exception as exc:
        print(f"[FreeHands] Could not open camera {profile.camera_index}: {exc}. Using camera 0.")
        camera = Camera().start()
    gaze_tracker = GazeTracker()
    hand_tracker = HandTracker()
    hand_tracker.set_handedness_swapped(profile.swap_handedness)
    face_tracker = FaceTracker()
    face_stabilizer = GestureStabilizer(
        required_frames=DEFAULT_STABILITY_FRAMES,
        confidence_min=DEFAULT_GESTURE_CONFIDENCE,
        per_gesture=gesture_thresholds,
        rearm_frames=1,
    )
    # Volume control by hand vertical position
    volume_control = VolumeControl()
    regressor = GazeRegressor(profile.gaze_model, (screen.width(), screen.height()))
    dead_zone = DeadZoneClamper(screen.width(), screen.height())
    snap_grid = SnapToGrid()
    gesture_thresholds = {
        gesture: (threshold.stability_frames, threshold.confidence_min)
        for gesture, threshold in profile.gesture_thresholds.items()
    }
    stabilizer = GestureStabilizer(
        required_frames=DEFAULT_STABILITY_FRAMES,
        confidence_min=DEFAULT_GESTURE_CONFIDENCE,
        per_gesture=gesture_thresholds,
        rearm_frames=1,
    )
    fusion = MultimodalFusion(profile)
    dispatcher = ActionDispatcher()
    audio_feedback = AudioFeedback(enabled=profile.audio_feedback_enabled)
    voice_listener: VoiceListener | None = None
    fine_aim = FineAimPointer()
    pause_hold = PauseHoldGate()
    last_pointer_move_at = 0.0
    last_pointer_xy: tuple[int, int] | None = None

    # ── Radial menu ──────────────────────────────────────────────────────
    radial_menu = RadialMenuWidget()
    radial_menu_open_hold_frames = 0
    radial_menu_open_gesture: str | None = None

    def execute_radial_action(action_id: str, cursor_xy: tuple[int, int] | None) -> None:
        """Execute an action from the radial menu."""
        if cursor_xy is None:
            return
        overlay.flash_action(f"radial: {action_id}")
        panel.set_last_action(f"radial: {action_id}")
        dispatcher.execute(action_id, at_xy=cursor_xy)
        audio_feedback.play_gesture_confirmation()

    radial_menu.action_selected.connect(execute_radial_action)

    # ── Virtual keyboard ───────────────────────────────────────────────
    virtual_kb = VirtualKeyboardWidget(dual_layout=True)
    virtual_kb_open_hold_frames = 0
    virtual_kb_open_gesture: str | None = None
    kb_typing_buffer: list[str] = []  # chars typed this session

    def on_kb_key(char_or_action: str) -> None:
        """Handle a key press from the virtual keyboard."""
        if char_or_action == "shift":
            return  # handled internally
        if char_or_action == "backspace":
            if kb_typing_buffer:
                kb_typing_buffer.pop()
            overlay.flash_action("⌫ backspace")
            panel.set_last_action("backspace")
            return
        if char_or_action == "space":
            kb_typing_buffer.append(" ")
            overlay.flash_action("espacio")
            panel.set_last_action("espacio")
            return
        if char_or_action == "enter":
            kb_typing_buffer.append("\n")
            overlay.flash_action("enter")
            panel.set_last_action("enter")
            return
        # Regular character
        kb_typing_buffer.append(char_or_action)
        overlay.flash_action(char_or_action)
        panel.set_last_action(f"tecla: {char_or_action}")

    def on_kb_layout_changed(side: str) -> None:
        """Handle dual-layout side change — play audio feedback."""
        if side == "left":
            audio_feedback.play_gesture_confirmation()
            overlay.flash_action("teclado: lado izq")
            panel.set_last_action("teclado: lado izq")
        elif side == "right":
            audio_feedback.play_gesture_confirmation()
            overlay.flash_action("teclado: lado der")
            panel.set_last_action("teclado: lado der")
        else:
            overlay.flash_action("teclado: completo")
            panel.set_last_action("teclado: completo")

    virtual_kb.key_pressed.connect(on_kb_key)
    virtual_kb.layout_changed.connect(on_kb_layout_changed)

    overlay = GazeOverlay()
    overlay.show()

    fusion.sm.activate()  # start in ACTIVE; open palm toggles back to IDLE

    panel = FreeHandsControlPanel(user_id)

    def toggle_handedness_swap() -> None:
        profile.swap_handedness = not profile.swap_handedness
        hand_tracker.set_handedness_swapped(profile.swap_handedness)
        save_profile(profile)
        panel.set_handedness_swapped(profile.swap_handedness)
        overlay.flash_action("swap L/R on" if profile.swap_handedness else "swap L/R off")

    def update_gesture_bindings(bindings: dict[str, str]) -> None:
        profile.gesture_bindings.update(bindings)
        save_profile(profile)
        overlay.flash_action("gesture actions saved")
        panel.set_last_action("gesture actions saved")

    def activate_system() -> None:
        fusion.sm.activate()
        overlay.flash_action("FreeHands active")
        panel.set_state(fusion.sm.state)

    def pause_system() -> None:
        fusion.sm.pause()
        overlay.flash_action("FreeHands paused")
        panel.set_state(fusion.sm.state)

    panel.activate_clicked.connect(activate_system)
    panel.pause_clicked.connect(pause_system)
    panel.swap_handedness_clicked.connect(toggle_handedness_swap)
    panel.bindings_saved.connect(update_gesture_bindings)
    panel.quit_clicked.connect(app.quit)
    panel.set_state(fusion.sm.state)
    panel.set_bindings(profile.gesture_bindings)
    panel.set_handedness_swapped(profile.swap_handedness)
    panel.show()

    # ── Voice ────────────────────────────────────────────────────────────
    if voice_enabled and profile.voice_enabled:
        try:
            voice_listener = VoiceListener(
                language=profile.voice_language,
                backend=profile.voice_asr_backend,
                vosk_model_path=profile.voice_vosk_model_path or None,
                wake_words=tuple(profile.voice_wake_words),
            ).start()
            print("Voice: enabled. Try: 'FreeHands click', 'Ntizar zoom in', 'pause', 'resume'.")
        except Exception as exc:
            voice_listener = None
            print(f"Voice: disabled ({exc})")

    # ── Plugin system ────────────────────────────────────────────────────
    from pathlib import Path as _Path
    _plugins_dir = getattr(profile, 'plugins_dir', None) or str(_Path(__file__).parent / "plugins")
    plugin_loader = PluginLoader(_plugins_dir)
    plugin_loader.discover_from_directory()
    plugin_count = plugin_loader.load()
    if plugin_count > 0:
        print(f"[FreeHands] Loaded {plugin_count} plugin(s): {', '.join(p.name for p in plugin_loader.active_plugins)}")
    else:
        print("[FreeHands] No plugins loaded.")

    def handle_voice_action(action: str, cursor: tuple[int, int] | None) -> None:
        """Handle voice-only actions that bypass fusion (state transitions, system commands)."""
        if action == "toggle_pause":
            fusion.sm.pause()
            overlay.flash_action("voice: pause")
            return
        if action == "resume":
            fusion.sm.activate()
            overlay.flash_action("voice: resume")
            return
        # Keyboard commands bypass state check — handled in tick() loop.
        if action in {"teclado", "cerrar_teclado"}:
            return
        if fusion.sm.state == State.IDLE:
            overlay.flash_action("voice ignored: paused")
            return
        # System commands are always executed regardless of state (safety controls).
        if action in {"show_desktop", "screenshot", "volume_up", "volume_down", "volume_mute"}:
            if dispatcher.execute(action, at_xy=cursor):
                overlay.flash_action(f"voice: {action}")
            return
        if dispatcher.execute(action, at_xy=cursor):
            overlay.flash_action(f"voice: {action}")

    # ── per-tick processing ──────────────────────────────────────────────
    def tick() -> None:
        nonlocal last_pointer_move_at, last_pointer_xy, radial_menu_open_hold_frames, radial_menu_open_gesture, virtual_kb_open_hold_frames, virtual_kb_open_gesture, kb_typing_buffer
        frame = camera.read()
        if frame is None:
            return

        # Gaze
        feats = gaze_tracker.extract(frame.image)
        cursor = regressor.predict(feats.vector) if feats else None
        blink_detected = feats.blink if feats else False
        blink_event = feats.blink_event if feats else None
        head_pose = feats.head_pose if feats else None
        if cursor is not None and fusion.sm.state != State.IDLE:
            # Dead-zone: prevent cursor from reaching extreme screen edges
            cursor = dead_zone.clamp(cursor)
            cursor = fine_aim.update(cursor)
            # Snap-to-grid: after ~300ms of stable gaze, snap to nearest
            # grid-cell centre to make targeting UI elements easier.
            # Reuse the stability check from fusion's gaze checker (peek,
            # not update — fusion.step() will consume the sample itself).
            gaze_stable = fusion.gaze_stable.peek()
            cursor = snap_grid.update(cursor, gaze_stable)
        else:
            fine_aim.reset()

        # Hand
        hand_obs = hand_tracker.detect(frame.image)
        confirmed = stabilizer.update(hand_obs.gesture, hand_obs.confidence)
        action = action_for_gesture(profile.gesture_bindings, confirmed) or ""
        # Leaky integrator for the palm-hold pause toggle: tolerates short
        # detection gaps (briefly losing the hand, mediapipe glitches, brief
        # finger flicker) so the bar that fills on screen actually corresponds
        # to a real toggle.
        pause_confirmed, pause_progress = pause_hold.update(
            hand_obs.gesture,
            profile.gesture_bindings,
            profile.gesture_thresholds,
        )
        if pause_confirmed is not None:
            confirmed = pause_confirmed
            stabilizer.reset()  # let the next post-pause click rearm cleanly
        else:
            confirmed = suppress_open_palm_toggle_without_hold(
                confirmed,
                profile.gesture_bindings,
            )
        action = action_for_gesture(profile.gesture_bindings, confirmed) or ""
        debug = gaze_tracker.last_debug
        gaze_source = "pupil" if debug.pupil_detected else "iris" if debug.iris_detected else "no eyes"
        fine_aim_text = " fine" if fine_aim.active else ""
        cursor_text = f"{cursor[0]},{cursor[1]}{fine_aim_text}" if cursor else "-"
        hand_side = f" side={','.join(hand_obs.handedness)}" if hand_obs.handedness else ""

        # ── Face (facial expressions) ────────────────────────────────────
        face_obs = face_tracker.detect(frame.image)
        face_gesture = face_obs.primary_gesture if face_obs.primary_gesture != "none" else None
        face_confirmed = face_stabilizer.update(face_gesture, face_obs.confidence) if face_gesture else None
        face_action = action_for_gesture(profile.gesture_bindings, face_confirmed) or None
        face_labels = []
        if face_obs.smile:
            face_labels.append("sonrisa")
        if face_obs.frown:
            face_labels.append("ceño")
        if face_obs.surprise:
            face_labels.append("sorpresa")
        if face_obs.raised_eyebrows:
            face_labels.append("cejas arriba")
        if face_obs.furrowed_brows:
            face_labels.append("cejas fruncidas")
        if face_obs.mouth_open:
            face_labels.append("boca abierta")
        if face_obs.tongue_out:
            face_labels.append("lengua fuera")
        face_text = ", ".join(face_labels) if face_labels else "—"
        # Head-pose info for runtime display
        if debug.head_active:
            head_info = f" HP:y={debug.head_yaw:.2f} p={debug.head_pitch:.2f}"
        else:
            head_info = ""
        panel.set_runtime_info(
            f"Gaze: {gaze_source} conf={debug.confidence:.2f} cursor={cursor_text}{head_info}",
            f"Hand: {hand_obs.gesture}{hand_side} {hand_obs.confidence:.2f}" + (f" -> {action}" if action else ""),
        )
        panel.set_face_info(face_text)
        panel.set_camera_preview(frame.image, hand_obs.hands, hand_obs.handedness, hand_obs.gesture)
        panel.set_pause_progress(pause_progress)

        # ── Plugin pipeline ──────────────────────────────────────────────
        # Run all plugin hooks for this frame. Plugins can modify cursor,
        # gesture, action, or add metadata for overlay rendering.
        from .plugins.base import PluginContext
        plugin_ctx = PluginContext(
            frame=frame.image,
            cursor=cursor,
            gesture=confirmed,
            action=action,
            blink=blink_detected,
            blink_event=blink_event.event_type.value if blink_event else None,
            state=fusion.sm.state.name,
        )
        plugin_loader.run_all(plugin_ctx)
        # Apply any plugin modifications back to local variables
        if plugin_ctx.cursor is not None:
            cursor = plugin_ctx.cursor
        if plugin_ctx.gesture is not None and confirmed is not None:
            confirmed = plugin_ctx.gesture
        if plugin_ctx.action is not None:
            action = plugin_ctx.action

        # ── Voice commands: drain before fusion so AND fusion can use them ─
        voice_actions_this_frame: list[str] = []
        if voice_listener is not None:
            for err in voice_listener.drain_errors():
                print(f"[voice] {err}")
            for cmd in voice_listener.drain_commands():
                print(f"[voice] {cmd.text!r} -> {cmd.action}")
                handle_voice_action(cmd.action, cursor)  # state/system commands
                voice_actions_this_frame.append(cmd.action)

        # ── Facial expressions: instant-action, no dwell, no state machine ──
        # Face gestures fire immediately when a bound facial expression is
        # detected (e.g. smile → custom action, surprise → custom action).
        # This runs before fusion so facial actions take priority over
        # hand/gaze dwell-based actions.
        if face_confirmed and face_action:
            overlay.flash_action(f"rostro: {face_action}")
            panel.set_last_action(f"rostro: {face_action}")
            dispatcher.execute(face_action, at_xy=cursor)
            audio_feedback.play_gesture_confirmation()

        # ── Volume control by hand vertical position ─────────────────────
        # If a hand is detected, check its vertical position to control
        # volume. Upper half → volume up, lower half → volume down.
        # This is independent of gesture bindings — always active when
        # a hand is visible and the system is ACTIVE.
        volume_obs = volume_control.detect(
            hand_obs.hands,
            hand_obs.handedness,
            hand_obs.confidence,
        )
        if volume_obs.gesture and fusion.sm.state != State.IDLE:
            if dispatcher.execute(volume_obs.gesture, at_xy=cursor):
                side_label = f" ({volume_obs.side})" if volume_obs.side else ""
                overlay.flash_action(f"volumen: {volume_obs.gesture}{side_label}")
                panel.set_last_action(f"volumen: {volume_obs.gesture}")
                audio_feedback.play_gesture_confirmation()
            # Reset volume control when hand is lost to avoid stale state
            if not hand_obs.hands:
                volume_control.reset()

        # ── Fusion: gesture + blink + AND voice ───────────────────────────
        # Use step_and_voice which applies the multimodal AND fusion:
        # pointer/gesture voice actions only fire when gaze is also present
        # and stable — requiring the user to look at the target.
        voice_action_for_fusion = voice_actions_this_frame[-1] if voice_actions_this_frame else None
        result = fusion.step_and_voice(
            cursor,
            confirmed,
            voice_action_for_fusion,
            blink=blink_detected,
            blink_event=blink_event.event_type if blink_event else None,
            head_pose=head_pose,
            screen_width=screen.width(),
            screen_height=screen.height(),
        )

        # ── Channel priority: gesture vs voice conflict resolution ──────────
        # If the fusion step produced a gesture action and there are pending
        # voice commands for pointer/gesture actions, resolve the conflict.
        # AND fusion already handled voice actions that lack gaze confirmation.
        if voice_listener is not None:
            for cmd_text in voice_actions_this_frame:
                if cmd_text in {"toggle_pause", "resume",
                                "show_desktop", "screenshot",
                                "volume_up", "volume_down", "volume_mute"}:
                    continue  # already handled by handle_voice_action
                if result.fired_action and result.fired_action != cmd_text:
                    # Both gesture and voice propose different actions — resolve.
                    decision = decide_channel_priority(
                        result.fired_action,
                        cmd_text,
                        gesture_confidence=0.9,
                        voice_confidence=1.0,
                    )
                    if decision.action and decision.action != result.fired_action:
                        # Voice wins — override gesture action.
                        overlay.flash_action(f"voice: {decision.action} (priority)")
                        panel.set_last_action(f"voice: {decision.action}")
                        click_xy = result.cursor_xy if result.cursor_xy is not None else last_pointer_xy
                        dispatcher.execute(decision.action, at_xy=click_xy)
                        audio_feedback.play_voice_confirmation()
                    # else gesture wins — already executed above
                elif cmd_text and not result.fired_action:
                    # No gesture action and AND fusion didn't fire (no gaze).
                    # Voice action fills the gap only for non-pointer actions.
                    from .fusion import AND_FUSION_ACTIONS
                    if cmd_text not in AND_FUSION_ACTIONS:
                        overlay.flash_action(f"voice: {cmd_text}")
                        panel.set_last_action(f"voice: {cmd_text}")
                        click_xy = result.cursor_xy if result.cursor_xy is not None else last_pointer_xy
                        dispatcher.execute(cmd_text, at_xy=click_xy)
                        audio_feedback.play_voice_confirmation()

        # ── Radial menu ──────────────────────────────────────────────────
        # Detect open-palm hold to open the radial menu.
        # The gesture must be one of the open-palm gestures mapped to
        # toggle_pause (the same gesture used for pause hold), but we
        # track a longer hold (MENU_OPEN_DURATION_MS frames) to distinguish
        # menu-open from pause-toggle.
        open_palm_gesture = confirmed if confirmed in {
            "left_open_palm", "right_open_palm",
            "left_palm_scroll_up", "left_palm_scroll_down",
            "right_palm_scroll_up", "right_palm_scroll_down",
        } else None

        if radial_menu.visible:
            # Menu is open — update dwell based on cursor position
            radial_menu.update_dwell(result.cursor_xy)
            # Dismiss on state change to IDLE
            if result.state == State.IDLE:
                radial_menu.close()
        else:
            # Menu is closed — track palm hold to open it
            if open_palm_gesture is not None:
                if radial_menu_open_gesture != open_palm_gesture:
                    radial_menu_open_gesture = open_palm_gesture
                    radial_menu_open_hold_frames = 0
                radial_menu_open_hold_frames += 1
                frames_needed = int(MENU_OPEN_DURATION_MS / (1000 / 30))
                if radial_menu_open_hold_frames >= frames_needed:
                    # Open the menu at the cursor position
                    if result.cursor_xy is not None:
                        radial_menu.open_at(result.cursor_xy[0], result.cursor_xy[1])
                    radial_menu_open_hold_frames = 0
                    radial_menu_open_gesture = None
            else:
                # Gesture changed or lost — reset hold counter
                if radial_menu_open_gesture is not None:
                    radial_menu_open_gesture = None
                    radial_menu_open_hold_frames = 0

        # ── Virtual keyboard ───────────────────────────────────────────
        # Open with a dedicated voice command "teclado" or "keyboard".
        # Close with Escape key or voice command "cerrar teclado".
        kb_command = None
        if voice_listener is not None:
            for cmd_text in voice_actions_this_frame:
                if cmd_text in {"teclado", "keyboard", "open_keyboard"}:
                    kb_command = "open"
                elif cmd_text in {"cerrar_teclado", "close_keyboard", "hide_keyboard"}:
                    kb_command = "close"

        if kb_command == "open":
            if not virtual_kb.visible:
                if result.cursor_xy is not None:
                    virtual_kb.open_at(result.cursor_xy[0], result.cursor_xy[1])
                overlay.flash_action("teclado: abierto")
                panel.set_last_action("teclado abierto")
                kb_typing_buffer = []
        elif kb_command == "close":
            if virtual_kb.visible:
                # Flush typed text via pyautogui
                text = "".join(kb_typing_buffer)
                if text:
                    import pyautogui
                    pyautogui.write(text)
                    kb_typing_buffer = []
                overlay.flash_action("teclado: cerrado")
                panel.set_last_action("teclado cerrado")
                virtual_kb.close()

        if virtual_kb.visible:
            # Update dwell based on cursor position
            virtual_kb.update_dwell(result.cursor_xy)
            # Process blink events for blink-to-select mode
            if blink_detected:
                virtual_kb.process_blink(blink_detected)
            # Dismiss on state change to IDLE
            if result.state == State.IDLE:
                text = "".join(kb_typing_buffer)
                if text:
                    import pyautogui
                    pyautogui.write(text)
                    kb_typing_buffer = []
                virtual_kb.close()
        else:
            # Track open-palm hold to open keyboard (longer hold than radial menu)
            # Use a double-palm-gesture: both hands open for 2 seconds
            if confirmed in {"left_open_palm", "right_open_palm"}:
                # Single palm is radial menu; we need a different trigger
                pass
            # Use double-click gesture (blink blink) as keyboard trigger
            # This is handled by blink_detector already for clicks.
            # Alternative: use a specific gesture binding.
            # For now, keyboard opens via voice command only.

        overlay.update_view(result.cursor_xy, result.dwell_progress, result.state,
                            snap_grid.active)
        panel.set_state(result.state)

        if result.state == State.IDLE:
            fine_aim.reset()
            snap_grid.reset()
            last_pointer_xy = None
            # Also close radial menu on pause
            if radial_menu.visible:
                radial_menu.close()

        if profile.pointer_control_enabled and result.cursor_xy is not None and result.state != State.IDLE:
            now = time.monotonic()
            should_move = (now - last_pointer_move_at) * 1000 >= POINTER_MOVE_INTERVAL_MS
            if last_pointer_xy is not None:
                dx = result.cursor_xy[0] - last_pointer_xy[0]
                dy = result.cursor_xy[1] - last_pointer_xy[1]
                should_move = should_move and (dx * dx + dy * dy >= POINTER_MOVE_MIN_DELTA_PX ** 2)
            if should_move:
                dispatcher.move_pointer(result.cursor_xy)
                last_pointer_move_at = now
                last_pointer_xy = result.cursor_xy

        # Close virtual keyboard on Escape (if visible)
        if virtual_kb.visible and result.fired_action == "escape":
            text = "".join(kb_typing_buffer)
            if text:
                import pyautogui
                pyautogui.write(text)
                kb_typing_buffer = []
            overlay.flash_action("teclado: cerrado (esc)")
            panel.set_last_action("teclado cerrado (esc)")
            virtual_kb.close()

        if result.fired_action:
            label = result.fired_action
            # Append fusion indicator for AND fusion results.
            if result.voice_action and result.gaze_confirmed:
                label = f"{result.fired_action} (voz+mirada)"
            elif result.voice_action and not result.gaze_confirmed:
                label = f"{result.fired_action} (esperando mirada)"
            # Append head-pose indicator.
            if result.head_pose_active:
                label = f"{label} [cabeza]"
            overlay.flash_action(label)
            panel.set_last_action(label)
            if result.fired_action not in {"toggle_pause", "resume"}:
                # If gaze briefly dropped (cursor None) but we still have a
                # recent OS pointer position, click there instead of skipping
                # the action — clicks must always land somewhere reasonable.
                click_xy = result.cursor_xy if result.cursor_xy is not None else last_pointer_xy
                dispatcher.execute(result.fired_action, at_xy=click_xy)
                audio_feedback.play_gesture_confirmation()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(int(1000 / TARGET_FPS))

    # ── shutdown plumbing ────────────────────────────────────────────────
    def cleanup() -> None:
        timer.stop()
        camera.stop()
        gaze_tracker.close()
        hand_tracker.close()
        face_tracker.close()
        if voice_listener is not None:
            voice_listener.stop()
        plugin_loader.unload()

    app.aboutToQuit.connect(cleanup)

    print(f"FreeHands running for user '{user_id}'. "
          f"Move mouse to a screen corner (PyAutoGUI failsafe) to abort.")
    print("Face tracking: enabled (smile, frown, surprise, eyebrows, mouth).")
    if not voice_enabled or not profile.voice_enabled:
        print("Voice: disabled.")
    print(f"Loaded profile dwell={profile.dwell_time_ms}ms, "
          f"bindings={profile.gesture_bindings}")
    _ = time.monotonic()  # silence unused-import in some linters

    return app.exec()
