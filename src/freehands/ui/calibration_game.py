"""Aim-trainer-style calibration minigame (Phase 1: gaze only).

Usage::

    python -m freehands calibrate --user luis

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

import random
import sys
from dataclasses import dataclass

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..capture import Camera
from ..config import CALIBRATION_POINTS, SAMPLES_PER_POINT
from ..gaze import CalibrationSample, GazeTracker, fit_gaze_model
from ..profiles import get_or_create_profile, save_profile
from .theme import GLOBAL_STYLESHEET, PALETTE


# ── Welcome screen ────────────────────────────────────────────────────────
class WelcomeScreen(QtWidgets.QWidget):
    start_clicked = QtCore.pyqtSignal()

    def __init__(self, user_id: str) -> None:
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
        title = QtWidgets.QLabel(f"Hola, {user_id} 👋")
        title.setProperty("class", "NtizarTitle")
        sub = QtWidgets.QLabel(
            "Vamos a calibrar tu mirada. Aparecerán puntos en la pantalla:\n"
            "• Mira fijamente cada punto\n"
            "• Haz clic encima con el ratón\n"
            "• Repite ~30-40 veces (3 por posición)\n\n"
            "Mantén la cabeza relativamente quieta y la cara bien iluminada."
        )
        sub.setProperty("class", "NtizarSubtitle")
        sub.setWordWrap(True)

        start = QtWidgets.QPushButton("Empezar calibración")
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

    def __init__(self, camera: Camera, tracker: GazeTracker) -> None:
        super().__init__()
        self.setObjectName("NtizarPage")
        self.setMouseTracking(True)
        self._camera = camera
        self._tracker = tracker
        self._samples: list[CalibrationSample] = []
        self._target_radius = 26
        self._plan = self._build_plan()
        self._current_idx = 0

    # ── plan ──────────────────────────────────────────────────────────────
    def _build_plan(self) -> _Plan:
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        w, h = screen.width(), screen.height()
        margin = 80
        pts: list[tuple[int, int]] = []
        for nx, ny in CALIBRATION_POINTS:
            px = int(margin + nx * (w - 2 * margin))
            py = int(margin + ny * (h - 2 * margin))
            for _ in range(SAMPLES_PER_POINT):
                pts.append((px, py))
        random.shuffle(pts)
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

        info = f"{self._current_idx}/{len(self._plan.points)}  ·  Mira el punto naranja y haz clic"
        p.setPen(QtGui.QColor(PALETTE.text_secondary))
        font = p.font(); font.setPointSize(11); p.setFont(font)
        p.drawText(QtCore.QRect(0, 50, rect.width(), 30),
                   QtCore.Qt.AlignmentFlag.AlignHCenter, info)

        # target
        if self._current_idx < len(self._plan.points):
            tx, ty = self._plan.points[self._current_idx]
            # halo
            halo = QtGui.QColor(PALETTE.orange); halo.setAlpha(70)
            p.setBrush(halo); p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawEllipse(QtCore.QPoint(tx, ty),
                          self._target_radius + 14, self._target_radius + 14)
            # disc
            p.setBrush(QtGui.QColor(PALETTE.orange))
            p.drawEllipse(QtCore.QPoint(tx, ty), self._target_radius, self._target_radius)
            # bullseye
            p.setBrush(QtGui.QColor("white"))
            p.drawEllipse(QtCore.QPoint(tx, ty), 6, 6)

    # ── mouse → record sample ────────────────────────────────────────────
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self._current_idx >= len(self._plan.points):
            return
        target = self._plan.points[self._current_idx]
        click = (event.position().x(), event.position().y())
        # Only accept clicks reasonably close to the target (anti-cheat)
        if (click[0] - target[0]) ** 2 + (click[1] - target[1]) ** 2 > (self._target_radius * 2.5) ** 2:
            return

        frame = self._camera.read()
        if frame is None:
            return
        feats = self._tracker.extract(frame.image)
        if feats is None:
            return  # face not detected — skip silently

        self._samples.append(CalibrationSample(features=feats.vector, target_xy=target))
        self._current_idx += 1
        self.update()

        if self._current_idx >= len(self._plan.points):
            self.finished.emit(self._samples)


# ── Orchestrator ──────────────────────────────────────────────────────────
class CalibrationWindow(QtWidgets.QMainWindow):
    def __init__(self, user_id: str) -> None:
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("FreeHands · Calibración")
        self.showFullScreen()

        self._camera = Camera().start()
        self._tracker = GazeTracker()

        self._stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self._stack)

        self.welcome = WelcomeScreen(user_id)
        self.welcome.start_clicked.connect(self._start_aim)
        self._stack.addWidget(self.welcome)

    def _start_aim(self) -> None:
        self.aim = AimTrainer(self._camera, self._tracker)
        self.aim.finished.connect(self._finish)
        self._stack.addWidget(self.aim)
        self._stack.setCurrentWidget(self.aim)

    def _finish(self, samples: list[CalibrationSample]) -> None:
        if len(samples) < 4:
            QtWidgets.QMessageBox.warning(self, "Calibración", "Muestras insuficientes.")
            self.close(); return

        model = fit_gaze_model(samples)
        # quick training error report
        X = np.stack([s.features for s in samples])
        y = np.array([s.target_xy for s in samples])
        wx = np.array(model.weights_x); wy = np.array(model.weights_y)
        pred = np.stack([X @ wx + model.bias_x, X @ wy + model.bias_y], axis=1)
        rms = float(np.sqrt(np.mean(np.sum((pred - y) ** 2, axis=1))))

        profile = get_or_create_profile(self.user_id)
        profile.gaze_model = model
        path = save_profile(profile)

        QtWidgets.QMessageBox.information(
            self,
            "Calibración completada ✅",
            f"Perfil guardado en:\n{path}\n\nError RMS de entrenamiento: {rms:.0f} px",
        )
        self.close()

    def closeEvent(self, event):  # noqa: N802
        try:
            self._camera.stop()
            self._tracker.close()
        except Exception:
            pass
        super().closeEvent(event)


def run_calibration(user_id: str) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    win = CalibrationWindow(user_id)
    win.show()
    return app.exec()
