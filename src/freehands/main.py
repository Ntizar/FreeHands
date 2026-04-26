"""Runtime orchestrator: capture → trackers → fusion → action + overlay."""
from __future__ import annotations

import sys
import time

from PyQt6 import QtCore, QtWidgets

from .actions import ActionDispatcher
from .capture import Camera
from .config import DEFAULT_GESTURE_CONFIDENCE, DEFAULT_STABILITY_FRAMES, TARGET_FPS
from .fusion import MultimodalFusion, State
from .gaze import GazeRegressor, GazeTracker
from .gestures import GestureStabilizer, HandTracker
from .profiles import load_profile
from .ui.overlay import FreeHandsControlPanel, GazeOverlay
from .voice import VoiceListener


def run_system(user_id: str, voice_enabled: bool = True) -> int:
    # ── Auto-onboarding: if no profile or no gaze model, calibrate first ──
    from .profiles.store import profile_path
    from .ui.calibration_game import run_calibration

    needs_calibration = False
    if not profile_path(user_id).exists():
        print(f"[FreeHands] No existe perfil para '{user_id}'. Lanzando calibración…")
        needs_calibration = True
    else:
        try:
            tmp = load_profile(user_id)
            if not tmp.gaze_model.weights_x:
                print(f"[FreeHands] El perfil '{user_id}' no tiene modelo de mirada. Lanzando calibración…")
                needs_calibration = True
        except Exception as e:
            print(f"[FreeHands] No se pudo leer el perfil ({e}). Lanzando calibración…")
            needs_calibration = True

    if needs_calibration:
        rc = run_calibration(user_id=user_id)
        if rc != 0:
            print("[FreeHands] Calibración cancelada o fallida. Saliendo.")
            return rc

    profile = load_profile(user_id)
    if not profile.gaze_model.weights_x:
        print(f"[FreeHands] Aún no hay modelo de mirada para '{user_id}'. "
              f"Vuelve a ejecutar la calibración.")
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
    voice_listener: VoiceListener | None = None

    overlay = GazeOverlay()
    overlay.show()

    fusion.sm.activate()  # start in ACTIVE; fist gesture toggles back to IDLE

    panel = FreeHandsControlPanel(user_id)

    def activate_system() -> None:
        fusion.sm.activate()
        overlay.flash_action("FreeHands activo")
        panel.set_state(fusion.sm.state)

    def pause_system() -> None:
        fusion.sm.pause()
        overlay.flash_action("FreeHands pausado")
        panel.set_state(fusion.sm.state)

    panel.activate_clicked.connect(activate_system)
    panel.pause_clicked.connect(pause_system)
    panel.quit_clicked.connect(app.quit)
    panel.set_state(fusion.sm.state)
    panel.show()

    if voice_enabled and profile.voice_enabled:
        try:
            voice_listener = VoiceListener(
                language=profile.voice_language,
                backend=profile.voice_asr_backend,
                wake_words=tuple(profile.voice_wake_words),
            ).start()
            print("Voice: enabled. Try: 'FreeHands clic', 'Ntizar zoom mas', 'pausa', 'reanudar'.")
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
        panel.set_state(result.state)

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
