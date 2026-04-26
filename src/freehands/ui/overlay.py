"""Transparent always-on-top overlay: gaze cursor + dwell ring + state badge."""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..fusion import State
from .theme import PALETTE


class GazeOverlay(QtWidgets.QWidget):
    """Frameless, click-through overlay drawn over the entire primary screen."""

    def __init__(self) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self._cursor: tuple[int, int] | None = None
        self._dwell_progress = 0.0
        self._state = State.IDLE
        self._action_flash: str | None = None

        self._flash_timer = QtCore.QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ── public API ────────────────────────────────────────────────────────
    def update_view(self, cursor: tuple[int, int] | None,
                    dwell_progress: float, state: State) -> None:
        self._cursor = cursor
        self._dwell_progress = dwell_progress
        self._state = state
        self.update()

    def flash_action(self, action: str) -> None:
        self._action_flash = action
        self._flash_timer.start(700)
        self.update()

    def _clear_flash(self) -> None:
        self._action_flash = None
        self.update()

    # ── paint ─────────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        self._draw_state_badge(p)

        if self._cursor is None:
            return
        x, y = self._cursor

        # Ntizar liquid-glass cursor
        outer = QtGui.QColor(PALETTE.blue)
        outer.setAlpha(60)
        p.setBrush(outer)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QPoint(x, y), 28, 28)

        inner = QtGui.QColor(PALETTE.blue)
        inner.setAlpha(220)
        p.setBrush(inner)
        p.drawEllipse(QtCore.QPoint(x, y), 6, 6)

        # Dwell ring (orange fill clockwise)
        if self._state in (State.ACTIVE, State.CONFIRMING) and self._dwell_progress > 0:
            pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 4)
            p.setPen(pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            rect = QtCore.QRect(x - 22, y - 22, 44, 44)
            p.drawArc(rect, 90 * 16, int(-360 * 16 * self._dwell_progress))

        if self._action_flash:
            p.setPen(QtGui.QColor(PALETTE.text_primary))
            font = p.font(); font.setPointSize(11); font.setBold(True); p.setFont(font)
            p.drawText(x + 32, y + 6, f"⟶ {self._action_flash}")

    def _draw_state_badge(self, p: QtGui.QPainter) -> None:
        text = {
            State.IDLE: "PAUSADO",
            State.ACTIVE: "ACTIVO",
            State.CONFIRMING: "CONFIRMAR",
            State.COOLDOWN: "COOLDOWN",
        }[self._state]
        color = {
            State.IDLE: PALETTE.text_muted,
            State.ACTIVE: PALETTE.blue,
            State.CONFIRMING: PALETTE.orange,
            State.COOLDOWN: PALETTE.warning,
        }[self._state]

        bg = QtGui.QColor(255, 255, 255, 200)
        p.setBrush(bg)
        p.setPen(QtGui.QPen(QtGui.QColor(color), 2))
        rect = QtCore.QRect(20, 20, 150, 36)
        p.drawRoundedRect(rect, 18, 18)

        p.setPen(QtGui.QColor(color))
        font = p.font(); font.setPointSize(10); font.setBold(True); p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, f"FreeHands · {text}")


class FreeHandsControlPanel(QtWidgets.QWidget):
    """Small control surface for turning PC control on/off."""

    activate_clicked = QtCore.pyqtSignal()
    pause_clicked = QtCore.pyqtSignal()
    quit_clicked = QtCore.pyqtSignal()

    def __init__(self, user_id: str) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.WindowStaysOnTopHint | QtCore.Qt.WindowType.Tool,
        )
        self.setWindowTitle("FreeHands")
        self.setFixedWidth(310)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(f"FreeHands · {user_id}")
        title.setObjectName("fhTitle")
        self._status = QtWidgets.QLabel("PAUSADO")
        self._status.setObjectName("fhStatus")

        self._activate = QtWidgets.QPushButton("Activar")
        self._pause = QtWidgets.QPushButton("Desactivar")
        quit_btn = QtWidgets.QPushButton("Cerrar")

        self._activate.clicked.connect(self.activate_clicked.emit)
        self._pause.clicked.connect(self.pause_clicked.emit)
        quit_btn.clicked.connect(self.quit_clicked.emit)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._activate)
        row.addWidget(self._pause)
        row.addWidget(quit_btn)

        hint = QtWidgets.QLabel("Gesto: puño cerrado alterna activar/desactivar.")
        hint.setWordWrap(True)
        hint.setObjectName("fhHint")

        layout.addWidget(title)
        layout.addWidget(self._status)
        layout.addLayout(row)
        layout.addWidget(hint)

        self.setStyleSheet(f"""
            QWidget {{
                background: rgba(255, 255, 255, 235);
                color: {PALETTE.text_primary};
                border-radius: 14px;
                font-family: Segoe UI, Arial, sans-serif;
            }}
            QLabel#fhTitle {{ font-size: 13px; font-weight: 700; color: {PALETTE.blue}; }}
            QLabel#fhStatus {{ font-size: 22px; font-weight: 800; padding: 4px 0; }}
            QLabel#fhHint {{ color: {PALETTE.text_secondary}; font-size: 11px; }}
            QPushButton {{
                border: 1px solid rgba(30, 91, 255, 0.25);
                border-radius: 8px;
                padding: 8px 10px;
                background: white;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: rgba(30, 91, 255, 0.08); }}
        """)

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.move(screen.right() - self.width() - 24, screen.top() + 72)

    def set_state(self, state: State) -> None:
        active = state != State.IDLE
        self._status.setText("ACTIVO" if active else "PAUSADO")
        self._status.setStyleSheet(
            f"color: {PALETTE.blue};" if active else f"color: {PALETTE.text_muted};"
        )
        self._activate.setEnabled(not active)
        self._pause.setEnabled(active)
