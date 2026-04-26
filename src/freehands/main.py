"""Runtime orchestrator: capture → trackers → fusion → action + overlay."""
from __future__ import annotations

import sys
import time

from PyQt6 import QtCore, QtWidgets

from .actions import ActionDispatcher
from .capture import Camera
from .config import DEFAULT_GESTURE_CONFIDENCE, DEFAULT_STABILITY_FRAMES, TARGET_FPS
from .fusion import MultimodalFusion
from .gaze import GazeRegressor, GazeTracker
from .gestures import GestureStabilizer, HandTracker
from .profiles import load_profile
from .ui.overlay import GazeOverlay


def run_system(user_id: str, voice_enabled: bool = True) -> int:
    profile = load_profile(user_id)
    if not profile.gaze_model.weights_x:
        print(f"Profile '{user_id}' has no gaze model. "
              f"Run: freehands calibrate --user {user_id}")
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    screen = app.primaryScreen().geometry()

    camera = Camera().start()
    gaze_tracker = GazeTracker()
    hand_tracker = HandTracker()
    regressor = GazeRegressor(profile.gaze_model, (screen.width(), screen.height()))
    stabilizer = GestureStabilizer(
        required_frames=DEFAULT_STABILITY_FRAMES,
        confidence_min=DEFAULT_GESTURE_CONFIDENCE,
    )
    fusion = MultimodalFusion(profile)
    dispatcher = ActionDispatcher()

    overlay = GazeOverlay()
    overlay.show()

    fusion.sm.activate()  # start in ACTIVE; fist gesture toggles back to IDLE

    # ── per-tick processing ──────────────────────────────────────────────
    def tick() -> None:
        frame = camera.read()
        if frame is None:
            return

        # Gaze
        feats = gaze_tracker.extract(frame.image)
        cursor = regressor.predict(feats.vector) if feats else None

        # Hand
        hand_obs = hand_tracker.detect(frame.image)
        confirmed = stabilizer.update(hand_obs.gesture, hand_obs.confidence)

        # Fusion
        result = fusion.step(cursor, confirmed)

        overlay.update_view(result.cursor_xy, result.dwell_progress, result.state)

        if result.fired_action:
            overlay.flash_action(result.fired_action)
            if result.fired_action not in {"toggle_pause", "resume"}:
                dispatcher.execute(result.fired_action, at_xy=result.cursor_xy)

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(int(1000 / TARGET_FPS))

    # ── shutdown plumbing ────────────────────────────────────────────────
    def cleanup() -> None:
        timer.stop()
        camera.stop()
        gaze_tracker.close()
        hand_tracker.close()

    app.aboutToQuit.connect(cleanup)

    print(f"FreeHands running for user '{user_id}'. "
          f"Move mouse to a screen corner (PyAutoGUI failsafe) to abort.")
    print(f"Voice: {'enabled' if voice_enabled else 'disabled'} (Phase 3 — stub).")
    print(f"Loaded profile dwell={profile.dwell_time_ms}ms, "
          f"bindings={profile.gesture_bindings}")
    _ = time.monotonic()  # silence unused-import in some linters

    return app.exec()
