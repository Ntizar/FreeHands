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
from .fusion import MultimodalFusion, State
from .gaze import GazeRegressor, GazeTracker, gaze_model_is_usable
from .gestures import GestureStabilizer, HandTracker
from .profiles import load_profile
from .ui.overlay import FreeHandsControlPanel, GazeOverlay
from .voice import VoiceListener


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
    regressor = GazeRegressor(profile.gaze_model, (screen.width(), screen.height()))
    gesture_thresholds = {
        gesture: (threshold.stability_frames, threshold.confidence_min)
        for gesture, threshold in profile.gesture_thresholds.items()
    }
    stabilizer = GestureStabilizer(
        required_frames=DEFAULT_STABILITY_FRAMES,
        confidence_min=DEFAULT_GESTURE_CONFIDENCE,
        per_gesture=gesture_thresholds,
    )
    fusion = MultimodalFusion(profile)
    dispatcher = ActionDispatcher()
    voice_listener: VoiceListener | None = None
    fine_aim = FineAimPointer()
    last_pointer_move_at = 0.0
    last_pointer_xy: tuple[int, int] | None = None

    overlay = GazeOverlay()
    overlay.show()

    fusion.sm.activate()  # start in ACTIVE; open palm toggles back to IDLE

    panel = FreeHandsControlPanel(user_id)

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
    panel.quit_clicked.connect(app.quit)
    panel.set_state(fusion.sm.state)
    panel.set_bindings(profile.gesture_bindings)
    panel.show()

    if voice_enabled and profile.voice_enabled:
        try:
            voice_listener = VoiceListener(
                language=profile.voice_language,
                backend=profile.voice_asr_backend,
                wake_words=tuple(profile.voice_wake_words),
            ).start()
            print("Voice: enabled. Try: 'FreeHands click', 'Ntizar zoom in', 'pause', 'resume'.")
        except Exception as exc:
            voice_listener = None
            print(f"Voice: disabled ({exc})")

    def handle_voice_action(action: str, cursor: tuple[int, int] | None) -> None:
        if action == "toggle_pause":
            fusion.sm.pause()
            overlay.flash_action("voice: pause")
            return
        if action == "resume":
            fusion.sm.activate()
            overlay.flash_action("voice: resume")
            return
        if fusion.sm.state == State.IDLE:
            overlay.flash_action("voice ignored: paused")
            return
        if dispatcher.execute(action, at_xy=cursor):
            overlay.flash_action(f"voice: {action}")

    # ── per-tick processing ──────────────────────────────────────────────
    def tick() -> None:
        nonlocal last_pointer_move_at, last_pointer_xy
        frame = camera.read()
        if frame is None:
            return

        # Gaze
        feats = gaze_tracker.extract(frame.image)
        cursor = regressor.predict(feats.vector) if feats else None
        if cursor is not None and fusion.sm.state != State.IDLE:
            cursor = fine_aim.update(cursor)
        else:
            fine_aim.reset()

        # Hand
        hand_obs = hand_tracker.detect(frame.image)
        confirmed = stabilizer.update(hand_obs.gesture, hand_obs.confidence)
        action = profile.gesture_bindings.get(confirmed, "") if confirmed else ""
        debug = gaze_tracker.last_debug
        gaze_source = "pupil" if debug.pupil_detected else "iris" if debug.iris_detected else "no eyes"
        fine_aim_text = " fine" if fine_aim.active else ""
        cursor_text = f"{cursor[0]},{cursor[1]}{fine_aim_text}" if cursor else "-"
        hand_side = f" side={','.join(hand_obs.handedness)}" if hand_obs.handedness else ""
        panel.set_runtime_info(
            f"Gaze: {gaze_source} conf={debug.confidence:.2f} cursor={cursor_text}",
            f"Hand: {hand_obs.gesture}{hand_side} {hand_obs.confidence:.2f}" + (f" -> {action}" if action else ""),
        )

        # Fusion
        result = fusion.step(cursor, confirmed)

        overlay.update_view(result.cursor_xy, result.dwell_progress, result.state)
        panel.set_state(result.state)

        if result.state == State.IDLE:
            fine_aim.reset()
            last_pointer_xy = None

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

        if result.fired_action:
            overlay.flash_action(result.fired_action)
            if result.fired_action not in {"toggle_pause", "resume"}:
                dispatcher.execute(result.fired_action, at_xy=result.cursor_xy)

        if voice_listener is not None:
            for err in voice_listener.drain_errors():
                print(f"[voice] {err}")
            for cmd in voice_listener.drain_commands():
                print(f"[voice] {cmd.text!r} -> {cmd.action}")
                handle_voice_action(cmd.action, result.cursor_xy)

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(int(1000 / TARGET_FPS))

    # ── shutdown plumbing ────────────────────────────────────────────────
    def cleanup() -> None:
        timer.stop()
        camera.stop()
        gaze_tracker.close()
        hand_tracker.close()
        if voice_listener is not None:
            voice_listener.stop()

    app.aboutToQuit.connect(cleanup)

    print(f"FreeHands running for user '{user_id}'. "
          f"Move mouse to a screen corner (PyAutoGUI failsafe) to abort.")
    if not voice_enabled or not profile.voice_enabled:
        print("Voice: disabled.")
    print(f"Loaded profile dwell={profile.dwell_time_ms}ms, "
          f"bindings={profile.gesture_bindings}")
    _ = time.monotonic()  # silence unused-import in some linters

    return app.exec()
