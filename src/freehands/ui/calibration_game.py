"""Aim-trainer-style calibration minigame (Phase 1: gaze only).

Usage::

    python -m freehands calibrate --user Ntizar

Workflow
--------
1. Welcome card explains what's about to happen.
2. A target appears at one of the configured normalised positions.
3. The user **looks at the target and clicks it with the mouse**.
4. We record (eye-feature vector → screen pixel) for every click.
5. After ``SAMPLES_PER_POINT`` clicks per point, we fit a ridge model and
   save it into the user's profile.

Phase 2+ will add the gesture rounds (thumb up/down, pinch, tongue).
"""
from __future__ import annotations

from collections import deque
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Literal

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..capture import Camera, list_available_cameras
from ..config import CALIBRATION_POINTS, SAMPLES_PER_POINT
from ..gaze import (
    CalibrationSample,
    GazeTracker,
    aggregate_gaze_features,
    build_gaze_design_vector,
    fit_gaze_model,
    gaze_model_has_signal,
    gaze_model_weight_norm,
)
from ..gestures import HandTracker
from ..profiles import GestureThreshold, get_or_create_profile, save_profile
from .theme import GLOBAL_STYLESHEET, PALETTE


CalibrationMode = Literal["full", "gaze", "gestures"]


# ── Welcome screen ────────────────────────────────────────────────────────
class WelcomeScreen(QtWidgets.QWidget):
    start_clicked = QtCore.pyqtSignal()

    def __init__(self, user_id: str, mode: CalibrationMode = "full") -> None:
        super().__init__()
        self.setObjectName("NtizarPage")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()

        card = QtWidgets.QFrame()
        card.setProperty("class", "NtizarCard")
        card.setFixedWidth(560)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setSpacing(14)

        brand = QtWidgets.QLabel("FreeHands")
        brand.setProperty("class", "NtizarBrand")
        title = QtWidgets.QLabel(f"Hello, {user_id}")
        title.setProperty("class", "NtizarTitle")
        if mode == "gestures":
            body = (
                "We will recalibrate hand gestures only. Gaze will not be changed.\n"
                "- The camera preview stays visible in the corner\n"
                "- Green points mean MediaPipe can see your hand\n"
                "- Hold each gesture until the ring reaches 100%\n\n"
                "If a gesture gets stuck, press ESC to skip it and save softer thresholds."
            )
            button_text = "Calibrate gestures"
        elif mode == "gaze":
            body = (
                "We will recalibrate gaze only. Gestures will not be changed.\n"
                "- Look steadily at each point\n"
                "- Click the orange target to confirm\n"
                "- Camera, eyes, landmarks and confidence stay visible\n"
                "- Press C if the wrong camera is active\n"
                "- The first 4 points are the screen corners\n\n"
                "Keep your head mostly steady and your face well lit."
            )
            button_text = "Calibrate gaze"
        else:
            body = (
                "We will calibrate gaze and gestures.\n"
                "- First: look at each orange target and click to confirm\n"
                "- Then: hold each gesture in front of the camera\n\n"
                "Gaze is saved before gestures start, so it is not lost if something fails."
            )
            button_text = "Start calibration"

        sub = QtWidgets.QLabel(body)
        sub.setProperty("class", "NtizarSubtitle")
        sub.setWordWrap(True)

        start = QtWidgets.QPushButton(button_text)
        start.setProperty("class", "NtizarPrimary")
        start.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        start.clicked.connect(self.start_clicked.emit)

        card_layout.addWidget(brand)
        card_layout.addWidget(title)
        card_layout.addWidget(sub)
        card_layout.addSpacing(8)
        card_layout.addWidget(start, alignment=QtCore.Qt.AlignmentFlag.AlignRight)

        layout.addWidget(card, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()


# ── Aim-trainer scene ─────────────────────────────────────────────────────
@dataclass
class _Plan:
    """Sequence of (px, py) targets in screen pixels."""
    points: list[tuple[int, int]]


class AimTrainer(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal(list)   # list[CalibrationSample]
    TARGET_CLICK_RADIUS_PX = 75
    TARGET_SETTLE_MS = 300

    def __init__(
        self,
        camera: Camera,
        tracker: GazeTracker,
        hand_tracker: HandTracker | None = None,
        on_camera_changed: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("NtizarPage")
        self.setMouseTracking(True)
        self._camera = camera
        self._tracker = tracker
        self._hand_tracker = hand_tracker
        self._on_camera_changed = on_camera_changed
        self._samples: list[CalibrationSample] = []
        self._target_radius = 26
        self._plan = self._build_plan()
        self._current_idx = 0
        self._latest_vector: np.ndarray | None = None
        self._latest_seen_at = 0.0
        self._status_message = "Looking for your gaze..."
        self._status_kind = "warn"
        self._preview_image: np.ndarray | None = None
        self._recent_vectors: deque[tuple[float, np.ndarray, float]] = deque(maxlen=8)
        self._available_cameras = list_available_cameras(max_index=8)
        if self._camera.index not in self._available_cameras:
            self._available_cameras.insert(0, self._camera.index)
        self._confirm_hold = 0
        self._confirm_cooldown_until = 0.0
        self._last_confirm_label = "gesture=waiting"
        self._target_changed_at = time.monotonic()
        self._last_good_gaze_at = self._target_changed_at
        self._last_camera_recovery_at = 0.0

        self._probe_timer = QtCore.QTimer(self)
        self._probe_timer.setInterval(120)
        self._probe_timer.timeout.connect(self._probe_gaze)
        self._probe_timer.start()

    def _set_status(self, message: str, kind: str = "info") -> None:
        self._status_message = message
        self._status_kind = kind
        self.update()

    def _probe_gaze(self) -> None:
        frame = self._camera.read()
        if frame is None:
            self._latest_vector = None
            self._set_status("Camera is busy or has no frames. Close other webcam apps.", "error")
            return
        self._preview_image = frame.image.copy()
        feats = self._tracker.extract(frame.image)
        debug = self._tracker.last_debug
        if feats is None:
            self._latest_vector = None
            self._maybe_recover_camera(now=time.monotonic())
            self._set_status(debug.message + ". Press C to switch camera.", "warn")
            return
        now = time.monotonic()
        self._last_good_gaze_at = now
        self._latest_vector = feats.vector
        self._latest_seen_at = now
        self._recent_vectors.append((now, feats.vector.copy(), feats.confidence))
        stable_samples = sum(1 for seen_at, _, _ in self._recent_vectors if now - seen_at <= 0.55)
        target_ready = (now - self._target_changed_at) * 1000 >= self.TARGET_SETTLE_MS
        if target_ready and stable_samples >= 3:
            self._set_status("Gaze detected and stable. Click to save this point.", "ok")
        else:
            self._set_status("Look at the new point briefly before confirming.", "ok")
        if self._hand_tracker is not None and target_ready and now >= self._confirm_cooldown_until:
            obs = self._hand_tracker.detect(frame.image)
            self._last_confirm_label = f"gesture={obs.gesture} · {obs.confidence:.2f}"
            if obs.gesture == "pointing_up" and obs.confidence >= 0.70:
                self._confirm_hold += 1
                if self._confirm_hold >= 3:
                    self._record_current_target("Index finger saved the point.")
                    self._confirm_hold = 0
                    self._confirm_cooldown_until = time.monotonic() + 0.65
            else:
                self._confirm_hold = max(0, self._confirm_hold - 1)

    def _maybe_recover_camera(self, now: float) -> None:
        if now - self._last_good_gaze_at < 2.5 or now - self._last_camera_recovery_at < 2.5:
            return
        self._last_camera_recovery_at = now
        try:
            self._camera.reopen()
            self._preview_image = None
            self._recent_vectors.clear()
            self._set_status("Reopening current camera to recover detection...", "warn")
        except Exception:
            self._switch_camera()

    def _switch_camera(self) -> None:
        cameras = self._available_cameras or [self._camera.index]
        try:
            pos = cameras.index(self._camera.index)
        except ValueError:
            pos = -1
        for step in range(1, len(cameras) + 1):
            next_index = cameras[(pos + step) % len(cameras)]
            if next_index == self._camera.index:
                continue
            try:
                self._camera.switch(next_index)
                self._preview_image = None
                self._latest_vector = None
                self._recent_vectors.clear()
                self._last_good_gaze_at = time.monotonic()
                if self._on_camera_changed is not None:
                    self._on_camera_changed(next_index)
                self._set_status(f"Switched to camera {next_index}. Waiting for eyes...", "warn")
                return
            except Exception as exc:
                self._set_status(f"Could not open camera {next_index}: {exc}", "error")
        self._set_status("No other camera is available.", "error")

    def _current_target_vector(self) -> np.ndarray | None:
        now = time.monotonic()
        if (now - self._target_changed_at) * 1000 < self.TARGET_SETTLE_MS:
            return None
        recent_vectors: list[np.ndarray] = []
        recent_weights: list[float] = []
        for seen_at, vector, confidence in self._recent_vectors:
            age = now - seen_at
            if age > 0.55:
                continue
            recency = max(0.2, 1.0 - age / 0.55)
            recent_vectors.append(vector)
            recent_weights.append(max(0.25, confidence) * recency)
        if recent_vectors:
            return aggregate_gaze_features(recent_vectors, recent_weights)
        return None

    # ── plan ──────────────────────────────────────────────────────────────
    def _build_plan(self) -> _Plan:
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        w, h = screen.width(), screen.height()
        margin = 42
        pts: list[tuple[int, int]] = []
        for nx, ny in CALIBRATION_POINTS:
            px = int(margin + nx * (w - 2 * margin))
            py = int(margin + ny * (h - 2 * margin))
            for _ in range(SAMPLES_PER_POINT):
                pts.append((px, py))
        return _Plan(points=pts)

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # liquid-glass background gradient
        grad = QtGui.QLinearGradient(0, 0, 0, rect.height())
        grad.setColorAt(0, QtGui.QColor(PALETTE.bg_grad_top))
        grad.setColorAt(1, QtGui.QColor(PALETTE.bg_grad_bottom))
        p.fillRect(rect, grad)

        # progress bar (top)
        progress = self._current_idx / max(len(self._plan.points), 1)
        bar_w = int(rect.width() * 0.35)
        bx = (rect.width() - bar_w) // 2
        p.setBrush(QtGui.QColor(255, 255, 255, 180))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(bx, 30, bar_w, 10, 5, 5)
        p.setBrush(QtGui.QColor(PALETTE.blue))
        p.drawRoundedRect(bx, 30, int(bar_w * progress), 10, 5, 5)

        info = f"{self._current_idx}/{len(self._plan.points)}  ·  Look at the point and click to confirm"
        p.setPen(QtGui.QColor(PALETTE.text_secondary))
        font = p.font()
        font.setPointSize(11)
        p.setFont(font)
        p.drawText(QtCore.QRect(0, 50, rect.width(), 30),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, info)

        status_color = {
            "ok": QtGui.QColor("#1a7f37"),
            "warn": QtGui.QColor(PALETTE.orange),
            "error": QtGui.QColor("#b42318"),
        }.get(self._status_kind, QtGui.QColor(PALETTE.text_secondary))
        p.setPen(status_color)
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(QtCore.QRect(0, 82, rect.width(), 28),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, self._status_message)

        self._draw_gaze_preview(p, rect)

        # target
        if self._current_idx < len(self._plan.points):
            tx, ty = self._plan.points[self._current_idx]
            # halo
            halo = QtGui.QColor(PALETTE.orange)
            halo.setAlpha(70)
            p.setBrush(halo)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawEllipse(QtCore.QPoint(tx, ty),
                          self._target_radius + 14, self._target_radius + 14)
            # disc
            p.setBrush(QtGui.QColor(PALETTE.orange))
            p.drawEllipse(QtCore.QPoint(tx, ty), self._target_radius, self._target_radius)
            # bullseye
            p.setBrush(QtGui.QColor("white"))
            p.drawEllipse(QtCore.QPoint(tx, ty), 6, 6)

    # ── mouse → record sample ────────────────────────────────────────────
    def _record_current_target(self, message: str) -> bool:
        if self._current_idx >= len(self._plan.points):
            return False
        target = self._plan.points[self._current_idx]
        vector = self._current_target_vector()
        if vector is None and (self._latest_vector is None or time.monotonic() - self._latest_seen_at > 1.0):
            frame = self._camera.read()
            feats = self._tracker.extract(frame.image) if frame is not None else None
            if feats is not None:
                now = time.monotonic()
                self._recent_vectors.append((now, feats.vector.copy(), feats.confidence))
                self._latest_vector = feats.vector
                self._latest_seen_at = now
                vector = self._current_target_vector()
        if vector is None:
            self._set_status("Confirmation received, but gaze needs a few more stable frames. Adjust light/face.", "error")
            return False

        self._samples.append(CalibrationSample(features=vector, target_xy=target))
        self._current_idx += 1
        self._recent_vectors.clear()
        self._latest_vector = None
        self._latest_seen_at = 0.0
        self._target_changed_at = time.monotonic()
        self._confirm_hold = 0
        self._set_status(message, "ok")
        self.update()

        if self._current_idx >= len(self._plan.points):
            self._probe_timer.stop()
            self.finished.emit(self._samples)
        return True

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._current_idx >= len(self._plan.points):
            return
        tx, ty = self._plan.points[self._current_idx]
        dx = event.position().x() - tx
        dy = event.position().y() - ty
        if dx * dx + dy * dy > self.TARGET_CLICK_RADIUS_PX * self.TARGET_CLICK_RADIUS_PX:
            self._set_status("Click the orange target to save this sample.", "warn")
            return
        self._record_current_target("Click saved the point.")

    def _draw_gaze_preview(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        preview_w = min(360, max(260, rect.width() // 4))
        preview_h = int(preview_w * 0.75)
        margin = 28
        if rect.width() < 780:
            box = QtCore.QRect((rect.width() - preview_w) // 2, 124, preview_w, preview_h)
        else:
            box = QtCore.QRect(rect.width() - preview_w - margin, 116, preview_w, preview_h)

        p.setBrush(QtGui.QColor(255, 255, 255, 205))
        p.setPen(QtGui.QPen(QtGui.QColor(PALETTE.blue), 2))
        p.drawRoundedRect(box.adjusted(-8, -30, 8, 88), 18, 18)

        debug = self._tracker.last_debug
        title = f"Camera {self._camera.index} · {debug.backend} · C switches"
        p.setPen(QtGui.QColor(PALETTE.text_primary))
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(box.adjusted(0, -25, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft, title)

        if self._preview_image is None:
            p.setBrush(QtGui.QColor(20, 24, 36, 230))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawRoundedRect(box, 12, 12)
            p.setPen(QtGui.QColor("white"))
            p.drawText(box, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
        else:
            rgb = self._preview_image[:, :, ::-1].copy()
            h, w, _ = rgb.shape
            image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
            p.drawImage(box, image)

            if debug.points:
                def map_point(name: str) -> QtCore.QPoint:
                    x, y = debug.points[name]
                    return QtCore.QPoint(box.left() + int(x / w * box.width()), box.top() + int(y / h * box.height()))

                p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                p.setPen(QtGui.QPen(QtGui.QColor(PALETTE.orange), 3))
                p.drawLine(map_point("left_outer"), map_point("left_inner"))
                p.drawLine(map_point("right_inner"), map_point("right_outer"))
                p.setBrush(QtGui.QColor(PALETTE.blue))
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                for name in ("left_iris", "right_iris", "nose"):
                    radius = 6 if name != "nose" else 4
                    p.drawEllipse(map_point(name), radius, radius)

        details = (
            f"face={'yes' if debug.face_detected else 'no'} · "
            f"landmarks={debug.landmark_count} · "
            f"iris={'yes' if debug.iris_detected else 'no'} · "
            f"pupil={'yes' if debug.pupil_detected else 'no'} · "
            f"conf={debug.confidence:.2f} · samples={len(self._recent_vectors)}"
        )
        vector_values = [] if self._latest_vector is None else self._latest_vector[:4]
        vector = "v=" + ", ".join(f"{v:.2f}" for v in vector_values)
        p.setBrush(QtGui.QColor(255, 255, 255, 230))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        badge = QtCore.QRect(box.left() + 10, box.bottom() + 10, box.width() - 20, 46)
        p.drawRoundedRect(badge, 12, 12)
        p.setPen(QtGui.QColor(PALETTE.text_primary))
        font = p.font()
        font.setPointSize(8)
        font.setBold(False)
        p.setFont(font)
        p.drawText(badge.adjusted(8, 4, -8, -24), QtCore.Qt.AlignmentFlag.AlignLeft, details)
        p.drawText(badge.adjusted(8, 23, -8, -4), QtCore.Qt.AlignmentFlag.AlignLeft,
               vector + " · " + self._last_confirm_label)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key.Key_C:
            self._switch_camera()


# ── Gesture round (Phase 2) ───────────────────────────────────────────────
GESTURE_BASE_ROUND: list[tuple[str, str, str]] = [
    # (gesture_id, label, helper text)
    ("pointing_up", "Index up", "Raise only the index finger: this confirms click."),
    ("middle_up", "Middle finger up", "Raise only the middle finger: this confirms right click."),
    ("two_fingers_up", "Index + middle", "Raise index and middle together: this confirms double click."),
    ("two_hands_together", "Hands together", "Show both hands close together: zoom in."),
    ("two_hands_apart", "Hands apart", "Show both hands apart: zoom out."),
    ("right_open_palm", "Right open palm", "Open your right hand fully for about two seconds: activates or pauses FreeHands."),
]
GESTURE_ROUND = GESTURE_BASE_ROUND + GESTURE_BASE_ROUND
HOLD_FRAMES = 12          # ~0.4 s @ 30 fps
HOLD_CONFIDENCE = 0.65    # minimum sustained confidence
TIMEOUT_MS = 9000         # per gesture
TICK_MS = 33              # ~30 fps

GESTURE_ALIASES = {
    "pointing_up": {"pointing_up", "left_pointing_up", "right_pointing_up"},
    "middle_up": {"middle_up", "left_middle_up", "right_middle_up"},
    "two_fingers_up": {"two_fingers_up", "left_two_fingers_up", "right_two_fingers_up"},
    "right_open_palm": {"right_open_palm", "left_open_palm", "open_palm"},
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class _GestureStat:
    gesture: str
    detected: bool
    median_conf: float
    frames_to_lock: int   # how many frames it took to reach HOLD_FRAMES


class GestureTrainer(QtWidgets.QWidget):
    finished = QtCore.pyqtSignal(list)   # list[_GestureStat]

    def __init__(
        self,
        camera: Camera,
        hand_tracker: HandTracker,
        on_camera_changed: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self.setObjectName("NtizarPage")
        self._camera = camera
        self._hands = hand_tracker
        self._on_camera_changed = on_camera_changed
        self._stats: list[_GestureStat] = []
        self._idx = 0
        self._held = 0
        self._frames = 0
        self._confs: list[float] = []
        self._last_obs_label = "—"
        self._last_confidence = 0.0
        self._last_hands: list[np.ndarray] = []
        self._preview_image: np.ndarray | None = None
        self._available_cameras = list_available_cameras(max_index=8)
        if self._camera.index not in self._available_cameras:
            self._available_cameras.insert(0, self._camera.index)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self._tick)

        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()

        self._timer.start()

    # ── helpers ──────────────────────────────────────────────────────────
    def _current(self) -> tuple[str, str, str] | None:
        if self._idx >= len(GESTURE_ROUND):
            return None
        return GESTURE_ROUND[self._idx]

    def _record_and_advance(self, detected: bool) -> None:
        target = GESTURE_ROUND[self._idx][0]
        median = float(np.median(self._confs)) if self._confs else 0.0
        self._stats.append(_GestureStat(
            gesture=target,
            detected=detected,
            median_conf=median,
            frames_to_lock=self._frames if detected else 0,
        ))
        self._idx += 1
        self._held = 0
        self._frames = 0
        self._confs.clear()
        self._last_obs_label = "—"
        self._elapsed.restart()

        if self._idx >= len(GESTURE_ROUND):
            self._timer.stop()
            self.finished.emit(self._stats)
        else:
            self.update()

    def _switch_camera(self) -> None:
        cameras = self._available_cameras or [self._camera.index]
        try:
            pos = cameras.index(self._camera.index)
        except ValueError:
            pos = -1
        for step in range(1, len(cameras) + 1):
            next_index = cameras[(pos + step) % len(cameras)]
            if next_index == self._camera.index:
                continue
            try:
                self._camera.switch(next_index)
                self._preview_image = None
                self._last_hands = []
                if self._on_camera_changed is not None:
                    self._on_camera_changed(next_index)
                self._last_obs_label = f"camera {next_index}"
                self.update()
                return
            except Exception:
                continue

    # ── per-frame logic ──────────────────────────────────────────────────
    def _tick(self) -> None:
        cur = self._current()
        if cur is None:
            return
        target_id, _, _ = cur

        # timeout
        if self._elapsed.elapsed() > TIMEOUT_MS:
            self._record_and_advance(detected=False)
            return

        frame = self._camera.read()
        if frame is None:
            return
        obs = self._hands.detect(frame.image)
        self._preview_image = frame.image.copy()
        self._last_hands = obs.hands
        self._last_obs_label = obs.gesture
        self._last_confidence = obs.confidence

        if obs.gesture in GESTURE_ALIASES.get(target_id, {target_id}) and obs.confidence >= HOLD_CONFIDENCE:
            self._held += 1
            self._frames += 1
            self._confs.append(obs.confidence)
            if self._held >= HOLD_FRAMES:
                self._record_and_advance(detected=True)
                return
        else:
            self._held = max(0, self._held - 1)
            self._frames += 1
        self.update()

    # ── painting ─────────────────────────────────────────────────────────
    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        grad = QtGui.QLinearGradient(0, 0, 0, rect.height())
        grad.setColorAt(0, QtGui.QColor(PALETTE.bg_grad_top))
        grad.setColorAt(1, QtGui.QColor(PALETTE.bg_grad_bottom))
        p.fillRect(rect, grad)

        cur = self._current()
        cx, cy = rect.width() // 2, rect.height() // 2

        font_title = p.font()
        font_title.setPointSize(28)
        font_title.setBold(True)
        font_sub = p.font()
        font_sub.setPointSize(13)
        font_micro = p.font()
        font_micro.setPointSize(10)

        if cur is None:
            p.setFont(font_title)
            p.setPen(QtGui.QColor(PALETTE.blue))
            p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Done")
            return

        _, label, helper = cur

        # Title
        p.setFont(font_title)
        p.setPen(QtGui.QColor(PALETTE.blue))
        p.drawText(QtCore.QRect(0, cy - 160, rect.width(), 60),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, label)

        # Helper
        p.setFont(font_sub)
        p.setPen(QtGui.QColor(PALETTE.text_secondary))
        p.drawText(QtCore.QRect(0, cy - 100, rect.width(), 30),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, helper)

        self._draw_camera_preview(p, rect)

        # Hold ring
        radius = 90
        p.setBrush(QtGui.QColor(255, 255, 255, 180))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QPoint(cx, cy + 20), radius, radius)

        ratio = self._held / HOLD_FRAMES
        path = QtGui.QPainterPath()
        path.moveTo(cx, cy + 20)
        path.arcTo(cx - radius, cy + 20 - radius, radius * 2, radius * 2,
                   90, -360 * min(1.0, ratio))
        path.closeSubpath()
        p.setBrush(QtGui.QColor(PALETTE.orange))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawPath(path)

        # Inner badge
        p.setBrush(QtGui.QColor("white"))
        p.drawEllipse(QtCore.QPoint(cx, cy + 20), radius - 18, radius - 18)
        p.setFont(font_sub)
        p.setPen(QtGui.QColor(PALETTE.text_primary))
        pct = int(min(1.0, ratio) * 100)
        p.drawText(QtCore.QRect(cx - radius, cy + 20 - 12, radius * 2, 24),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, f"{pct}%")

        # Footer
        p.setFont(font_micro)
        p.setPen(QtGui.QColor(PALETTE.text_secondary))
        rem = max(0, TIMEOUT_MS - self._elapsed.elapsed()) // 1000
        info = (f"Detecting: {self._last_obs_label}   ·   "
            f"Gesture {self._idx + 1}/{len(GESTURE_ROUND)}   ·   "
            f"{rem}s left   ·   ESC to skip")
        p.drawText(QtCore.QRect(0, rect.height() - 60, rect.width(), 30),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, info)

    def _draw_camera_preview(self, p: QtGui.QPainter, rect: QtCore.QRect) -> None:
        preview_w, preview_h = 320, 240
        margin = 28
        box = QtCore.QRect(rect.width() - preview_w - margin, 86, preview_w, preview_h)

        p.setBrush(QtGui.QColor(255, 255, 255, 190))
        p.setPen(QtGui.QPen(QtGui.QColor(PALETTE.blue), 2))
        p.drawRoundedRect(box.adjusted(-8, -28, 8, 42), 18, 18)

        p.setPen(QtGui.QColor(PALETTE.text_primary))
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(box.adjusted(0, -24, 0, 0), QtCore.Qt.AlignmentFlag.AlignLeft,
               f"Camera {self._camera.index} / hand · C switches")

        if self._preview_image is None:
            p.setBrush(QtGui.QColor(20, 24, 36, 230))
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawRoundedRect(box, 12, 12)
            p.setPen(QtGui.QColor("white"))
            p.drawText(box, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for camera...")
            return

        rgb = self._preview_image[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        p.drawImage(box, image)

        if self._last_hands:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 3)
            p.setPen(pen)
            for hand in self._last_hands:
                for a, b in HAND_CONNECTIONS:
                    ax = box.left() + int(hand[a, 0] * box.width())
                    ay = box.top() + int(hand[a, 1] * box.height())
                    bx = box.left() + int(hand[b, 0] * box.width())
                    by = box.top() + int(hand[b, 1] * box.height())
                    p.drawLine(ax, ay, bx, by)

                p.setBrush(QtGui.QColor(PALETTE.blue))
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                for x, y, _ in hand:
                    px = box.left() + int(x * box.width())
                    py = box.top() + int(y * box.height())
                    p.drawEllipse(QtCore.QPoint(px, py), 4, 4)

        status = f"{self._last_obs_label} · {self._last_confidence:.2f}"
        p.setBrush(QtGui.QColor(255, 255, 255, 220))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        badge = QtCore.QRect(box.left() + 10, box.bottom() + 10, box.width() - 20, 24)
        p.drawRoundedRect(badge, 12, 12)
        p.setPen(QtGui.QColor(PALETTE.text_primary))
        font = p.font()
        font.setPointSize(9)
        font.setBold(False)
        p.setFont(font)
        p.drawText(badge, QtCore.Qt.AlignmentFlag.AlignCenter, status)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key.Key_C:
            self._switch_camera()
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self._record_and_advance(detected=False)


# ── Orchestrator ──────────────────────────────────────────────────────────
class CalibrationWindow(QtWidgets.QMainWindow):
    def __init__(self, user_id: str, mode: CalibrationMode = "full") -> None:
        super().__init__()
        self.user_id = user_id
        self.mode = mode
        self.setWindowTitle("FreeHands · Calibration")
        self.showFullScreen()

        self._tracker: GazeTracker | None = None
        self._hands: HandTracker | None = None
        self._profile = get_or_create_profile(self.user_id)
        try:
            self._camera = Camera(self._profile.camera_index).start()
        except Exception:
            self._camera = Camera().start()
            self._save_camera_index(self._camera.index)
        self._gaze_rms: float | None = None

        self._stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self._stack)

        self.welcome = WelcomeScreen(user_id, mode)
        self.welcome.start_clicked.connect(self._start_selected_mode)
        self._stack.addWidget(self.welcome)

    def _save_camera_index(self, index: int) -> None:
        self._profile.camera_index = index
        save_profile(self._profile)

    def _start_selected_mode(self) -> None:
        if self.mode == "gestures":
            self._start_gestures()
        else:
            self._start_aim()

    def _start_aim(self) -> None:
        try:
            self._tracker = GazeTracker()
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self,
                "Calibration",
                "Could not start gaze tracking:\n" + str(exc) +
                "\n\nRun FreeHands.bat doctor to inspect or repair dependencies.",
            )
            self.close()
            return
        # Keep gaze calibration light: hand tracking runs in the separate gesture phase.
        self.aim = AimTrainer(self._camera, self._tracker, None, self._save_camera_index)
        self.aim.finished.connect(self._after_aim)
        self._stack.addWidget(self.aim)
        self._stack.setCurrentWidget(self.aim)

    def _after_aim(self, samples: list[CalibrationSample]) -> None:
        # Persist the gaze model first, then move to gesture round.
        if len(samples) < 4:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Not enough samples.")
            self.close()
            return

        model = fit_gaze_model(samples)
        X = np.stack([
            build_gaze_design_vector(s.features, model.feature_version)
            for s in samples
        ])
        y = np.array([s.target_xy for s in samples])
        wx = np.array(model.weights_x)
        wy = np.array(model.weights_y)
        pred = np.stack([X @ wx + model.bias_x, X @ wy + model.bias_y], axis=1)
        self._gaze_rms = float(np.sqrt(np.mean(np.sum((pred - y) ** 2, axis=1))))

        pred_range_x = float(np.ptp(pred[:, 0])) if len(pred) else 0.0
        pred_range_y = float(np.ptp(pred[:, 1])) if len(pred) else 0.0
        target_range_x = float(np.ptp(y[:, 0])) if len(y) else 0.0
        target_range_y = float(np.ptp(y[:, 1])) if len(y) else 0.0
        weak_x = target_range_x > 0 and pred_range_x < target_range_x * 0.35
        weak_y = target_range_y > 0 and pred_range_y < target_range_y * 0.35
        if not gaze_model_has_signal(model) or weak_x or weak_y:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Calibration did not produce a useful gaze signal.\n\n"
                f"weight={gaze_model_weight_norm(model):.3e} · rangeX={pred_range_x:.1f}/{target_range_x:.1f}px · rangeY={pred_range_y:.1f}/{target_range_y:.1f}px\n\n"
                "Repeat calibration and look at each point before confirming.",
            )
            self.close()
            return

        self._profile.gaze_model = model
        self._profile.gaze_calibrated_at = datetime.now().isoformat(timespec="seconds")
        save_profile(self._profile)

        if self.mode == "gaze":
            path = save_profile(self._profile)
            QtWidgets.QMessageBox.information(
                self,
                "Gaze calibrated",
                f"Profile saved at:\n{path}\n\nRMS error: {self._gaze_rms:.0f} px",
            )
            self.close()
            return

        self._start_gestures()

    def _start_gestures(self) -> None:
        try:
            if self._hands is None:
                self._hands = HandTracker()
        except Exception as exc:
            QtWidgets.QMessageBox.information(
                self, "Calibration",
                "Could not start the gesture round:\n" + str(exc),
            )
            self.close()
            return

        self.gestures = GestureTrainer(self._camera, self._hands, self._save_camera_index)
        self.gestures.finished.connect(self._finish)
        self._stack.addWidget(self.gestures)
        self._stack.setCurrentWidget(self.gestures)

    def _finish(self, stats: list[_GestureStat]) -> None:
        # Adapt per-gesture thresholds based on observed performance
        lines: list[str] = []
        self._profile.gesture_calibration_results = {}
        grouped: dict[str, list[_GestureStat]] = {}
        for stat in stats:
            grouped.setdefault(stat.gesture, []).append(stat)

        for gesture, _, _ in GESTURE_BASE_ROUND:
            gesture_stats = grouped.get(gesture, [])
            cur = self._profile.gesture_thresholds.get(gesture, GestureThreshold())
            detected_stats = [stat for stat in gesture_stats if stat.detected]
            detected_count = len(detected_stats)
            detected = detected_count > 0
            medians = [stat.median_conf for stat in detected_stats if stat.median_conf > 0]
            lock_frames = [stat.frames_to_lock for stat in detected_stats if stat.frames_to_lock > 0]
            median_conf = float(np.median(medians)) if medians else 0.0
            frames_to_lock = int(np.median(lock_frames)) if lock_frames else 0
            total_repetitions = max(len(gesture_stats), 1)
            self._profile.gesture_calibration_results[gesture] = {
                "detected": detected,
                "detected_repetitions": detected_count,
                "total_repetitions": total_repetitions,
                "median_conf": round(median_conf, 3),
                "frames_to_lock": frames_to_lock,
            }
            if detected:
                # Lower the bar slightly toward the user's actual confidence,
                # but never below 0.55 and never above 0.95.
                target_conf = max(0.55, min(0.95, median_conf - 0.05))
                # Required frames: average of default and observed lock time, clamped.
                target_frames = max(4, min(20, int((cur.stability_frames + frames_to_lock) / 2)))
                self._profile.gesture_thresholds[gesture] = GestureThreshold(
                    confidence_min=round(target_conf, 2),
                    stability_frames=target_frames,
                )
                lines.append(f"  {gesture}: ok {detected_count}/{total_repetitions} conf>={target_conf:.2f}, "
                             f"frames>={target_frames}")
            else:
                # Loosen slightly so the user isn't locked out
                target_conf = max(0.55, cur.confidence_min - 0.05)
                target_frames = max(4, cur.stability_frames - 1)
                self._profile.gesture_thresholds[gesture] = GestureThreshold(
                    confidence_min=round(target_conf, 2),
                    stability_frames=target_frames,
                )
                lines.append(f"  {gesture}: not detected, relaxed to "
                             f"conf>={target_conf:.2f}, frames>={target_frames}")

        self._profile.gesture_calibrated_at = datetime.now().isoformat(timespec="seconds")
        path = save_profile(self._profile)
        gaze_line = (
            f"Gaze - RMS error: {self._gaze_rms:.0f} px\n"
            if self._gaze_rms is not None else
            "Gaze - existing calibration kept.\n"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Calibration complete",
            "Profile saved at:\n" + str(path) +
            "\n\n" + gaze_line +
            "Gestures:\n" + "\n".join(lines),
        )
        self.close()

    def closeEvent(self, event):  # noqa: N802
        try:
            self._camera.stop()
            if self._tracker is not None:
                self._tracker.close()
            if self._hands is not None:
                self._hands.close()
        except Exception:
            pass
        super().closeEvent(event)


def run_calibration(user_id: str, mode: CalibrationMode = "full") -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    win = CalibrationWindow(user_id, mode=mode)
    win.show()
    return app.exec()


def run_gaze_calibration(user_id: str) -> int:
    return run_calibration(user_id, mode="gaze")


def run_gesture_calibration(user_id: str) -> int:
    return run_calibration(user_id, mode="gestures")
