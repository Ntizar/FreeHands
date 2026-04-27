"""Transparent always-on-top overlay: gaze cursor + dwell ring + state badge."""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..fusion import State
from .theme import PALETTE


GESTURE_LABELS = {
    "thumb_up": "Thumb up",
    "thumb_down": "Thumb down",
    "pointing_up": "Index",
    "middle_up": "Middle",
    "two_fingers_up": "Index+middle",
    "left_pointing_up": "Left index",
    "right_pointing_up": "Right index",
    "left_middle_up": "Left middle",
    "right_middle_up": "Right middle",
    "left_two_fingers_up": "Left index+middle",
    "right_two_fingers_up": "Right index+middle",
    "two_hands_together": "Hands together",
    "two_hands_apart": "Hands apart",
    "pinch_open": "Pinch open",
    "pinch_close": "Pinch close",
    "left_open_palm": "Left palm",
    "right_open_palm": "Right palm",
    "fist_pause": "Closed fist",
}

ACTION_OPTIONS = {
    "": "off",
    "click": "click",
    "right_click": "right click",
    "double_click": "double click",
    "scroll_up": "scroll up",
    "scroll_down": "scroll down",
    "zoom_in": "zoom +",
    "zoom_out": "zoom -",
    "escape": "escape",
    "undo": "undo",
    "toggle_pause": "active/pause",
}


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
            font = p.font()
            font.setPointSize(11)
            font.setBold(True)
            p.setFont(font)
            p.drawText(x + 32, y + 6, f"-> {self._action_flash}")

    def _draw_state_badge(self, p: QtGui.QPainter) -> None:
        text = {
            State.IDLE: "PAUSED",
            State.ACTIVE: "ACTIVE",
            State.CONFIRMING: "CONFIRM",
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
        font = p.font()
        font.setPointSize(10)
        font.setBold(True)
        p.setFont(font)
        p.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, f"FreeHands · {text}")


class FreeHandsControlPanel(QtWidgets.QWidget):
    """Small control surface for turning PC control on/off."""

    activate_clicked = QtCore.pyqtSignal()
    pause_clicked = QtCore.pyqtSignal()
    swap_handedness_clicked = QtCore.pyqtSignal()
    binding_changed = QtCore.pyqtSignal(str, str)
    quit_clicked = QtCore.pyqtSignal()

    def __init__(self, user_id: str) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.WindowStaysOnTopHint | QtCore.Qt.WindowType.Tool,
        )
        self.setWindowTitle("FreeHands")
        self.setFixedWidth(420)
        self._binding_combos: dict[str, QtWidgets.QComboBox] = {}
        self._updating_bindings = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel(f"FreeHands · {user_id}")
        title.setObjectName("fhTitle")
        self._status = QtWidgets.QLabel("PAUSED")
        self._status.setObjectName("fhStatus")

        self._activate = QtWidgets.QPushButton("Activate")
        self._pause = QtWidgets.QPushButton("Pause")
        self._swap_handedness = QtWidgets.QPushButton("Swap L/R: off")
        quit_btn = QtWidgets.QPushButton("Close")

        self._activate.clicked.connect(self.activate_clicked.emit)
        self._pause.clicked.connect(self.pause_clicked.emit)
        self._swap_handedness.clicked.connect(self.swap_handedness_clicked.emit)
        quit_btn.clicked.connect(self.quit_clicked.emit)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._activate)
        row.addWidget(self._pause)
        row.addWidget(quit_btn)

        self._camera_preview = QtWidgets.QLabel("Camera preview")
        self._camera_preview.setObjectName("fhCamera")
        self._camera_preview.setFixedSize(392, 220)
        self._camera_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        hint = QtWidgets.QLabel("Gesture: hold right open palm for 2s to toggle active/paused.")
        hint.setWordWrap(True)
        hint.setObjectName("fhHint")
        self._last_action = QtWidgets.QLabel("Last action: -")
        self._last_action.setObjectName("fhAction")
        self._pause_progress = QtWidgets.QProgressBar()
        self._pause_progress.setObjectName("fhPauseProgress")
        self._pause_progress.setRange(0, 100)
        self._pause_progress.setValue(0)
        self._pause_progress.setFormat("Pause hold %p%")
        self._gaze = QtWidgets.QLabel("Gaze: waiting")
        self._gaze.setObjectName("fhRuntime")
        self._gaze.setWordWrap(True)
        self._gesture = QtWidgets.QLabel("Hand: waiting")
        self._gesture.setObjectName("fhRuntime")
        self._gesture.setWordWrap(True)
        bindings_scroll = QtWidgets.QScrollArea()
        bindings_scroll.setObjectName("fhBindingsScroll")
        bindings_scroll.setWidgetResizable(True)
        bindings_scroll.setFixedHeight(250)
        bindings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bindings_box = QtWidgets.QGroupBox("Gesture actions")
        bindings_box.setObjectName("fhBindingsBox")
        bindings_grid = QtWidgets.QGridLayout(bindings_box)
        bindings_grid.setContentsMargins(8, 8, 8, 8)
        bindings_grid.setHorizontalSpacing(8)
        bindings_grid.setVerticalSpacing(4)
        for row_index, (gesture, label) in enumerate(GESTURE_LABELS.items()):
            gesture_label = QtWidgets.QLabel(label)
            gesture_label.setObjectName("fhBindingLabel")
            combo = QtWidgets.QComboBox()
            combo.setObjectName("fhBindingCombo")
            for action, action_label in ACTION_OPTIONS.items():
                combo.addItem(action_label, action)
            combo.currentIndexChanged.connect(lambda _index, g=gesture: self._emit_binding_changed(g))
            self._binding_combos[gesture] = combo
            bindings_grid.addWidget(gesture_label, row_index, 0)
            bindings_grid.addWidget(combo, row_index, 1)
        bindings_scroll.setWidget(bindings_box)

        layout.addWidget(title)
        layout.addWidget(self._status)
        layout.addWidget(self._camera_preview)
        layout.addLayout(row)
        layout.addWidget(self._swap_handedness)
        layout.addWidget(self._gaze)
        layout.addWidget(self._gesture)
        layout.addWidget(self._last_action)
        layout.addWidget(self._pause_progress)
        layout.addWidget(bindings_scroll)
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
            QLabel#fhCamera {{
                background: rgba(8, 18, 38, 0.08);
                border: 1px solid rgba(30, 91, 255, 0.18);
                border-radius: 8px;
                color: {PALETTE.text_secondary};
                font-size: 11px;
                font-weight: 700;
            }}
            QLabel#fhHint {{ color: {PALETTE.text_secondary}; font-size: 11px; }}
            QLabel#fhRuntime {{ color: {PALETTE.text_primary}; font-size: 11px; font-weight: 700; }}
            QLabel#fhAction {{ color: {PALETTE.orange}; font-size: 12px; font-weight: 800; }}
            QGroupBox#fhBindingsBox {{
                border: 1px solid rgba(30, 91, 255, 0.16);
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
                font-size: 11px;
                font-weight: 800;
                color: {PALETTE.blue};
            }}
            QScrollArea#fhBindingsScroll {{ border: none; background: transparent; }}
            QLabel#fhBindingLabel {{ color: {PALETTE.text_primary}; font-size: 10px; font-weight: 700; }}
            QComboBox#fhBindingCombo {{
                border: 1px solid rgba(30, 91, 255, 0.18);
                border-radius: 6px;
                background: white;
                padding: 3px 6px;
                font-size: 10px;
                min-height: 20px;
            }}
            QProgressBar {{
                border: 1px solid rgba(30, 91, 255, 0.18);
                border-radius: 6px;
                height: 14px;
                text-align: center;
                color: {PALETTE.text_primary};
                background: rgba(30, 91, 255, 0.06);
                font-size: 9px;
                font-weight: 700;
            }}
            QProgressBar::chunk {{ background: {PALETTE.orange}; border-radius: 5px; }}
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

    def _emit_binding_changed(self, gesture: str) -> None:
        if self._updating_bindings:
            return
        combo = self._binding_combos[gesture]
        action = combo.currentData() or ""
        self.binding_changed.emit(gesture, action)

    def set_state(self, state: State) -> None:
        active = state != State.IDLE
        self._status.setText("ACTIVE" if active else "PAUSED")
        self._status.setStyleSheet(
            f"color: {PALETTE.blue};" if active else f"color: {PALETTE.text_muted};"
        )
        self._activate.setEnabled(not active)
        self._pause.setEnabled(active)

    def set_runtime_info(self, gaze: str, gesture: str) -> None:
        self._gaze.setText(gaze)
        self._gesture.setText(gesture)

    def set_camera_preview(
        self,
        frame_bgr,
        hands: list,
        handedness: list[str],
        gesture: str,
    ) -> None:
        if frame_bgr is None:
            return
        height, width = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1].copy()
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            width * 3,
            QtGui.QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            self._camera_preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        scale_x = pixmap.width()
        scale_y = pixmap.height()
        pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(PALETTE.orange))
        for index, hand in enumerate(hands):
            for point in hand:
                painter.drawEllipse(QtCore.QPointF(float(point[0]) * scale_x, float(point[1]) * scale_y), 2.2, 2.2)
            label = handedness[index] if index < len(handedness) else "?"
            wrist = hand[0]
            painter.drawText(
                int(float(wrist[0]) * scale_x) + 6,
                int(float(wrist[1]) * scale_y) - 6,
                label,
            )
        painter.setPen(QtGui.QColor(PALETTE.blue))
        painter.drawText(8, 18, f"{gesture or 'none'} · mirrored")
        painter.end()
        self._camera_preview.setPixmap(pixmap)

    def set_pause_progress(self, progress: float) -> None:
        self._pause_progress.setValue(round(max(0.0, min(1.0, progress)) * 100))

    def set_last_action(self, action: str) -> None:
        self._last_action.setText(f"Last action: {action}")

    def set_handedness_swapped(self, enabled: bool) -> None:
        self._swap_handedness.setText("Swap L/R: on" if enabled else "Swap L/R: off")

    def set_bindings(self, bindings: dict[str, str]) -> None:
        self._updating_bindings = True
        try:
            for gesture, combo in self._binding_combos.items():
                action = bindings.get(gesture, "")
                index = combo.findData(action)
                combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._updating_bindings = False
