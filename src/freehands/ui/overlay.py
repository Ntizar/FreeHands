"""Transparent always-on-top overlay: gaze cursor + dwell ring + state badge."""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from ..fusion import State
from .theme import PALETTE


GESTURE_LABELS = {
    "pointing_up": "☝️ Index",
    "middle_up": "🖕 Middle",
    "two_fingers_up": "✌️ Index+middle",
    "left_pointing_up": "👈☝️ Left index",
    "right_pointing_up": "👉☝️ Right index",
    "left_middle_up": "👈🖕 Left middle",
    "right_middle_up": "👉🖕 Right middle",
    "left_two_fingers_up": "👈✌️ Left index+middle",
    "right_two_fingers_up": "👉✌️ Right index+middle",
    "right_open_palm": "👉🖐 Right palm",
    "left_open_palm": "👈🖐 Left palm",
    "two_hands_together": "🤲 Hands together",
    "two_hands_apart": "🙌 Hands apart",
    "thumb_down": "👎 Thumb down",
    "thumb_up": "👍 Thumb up",
    "pinch_open": "🤏↔ Pinch open",
    "pinch_close": "🤏 Pinch close",
    "fist_pause": "✊ Closed fist",
    # Palm-scroll gestures
    "palm_scroll_up": "🖐↓ Palm scroll up",
    "palm_scroll_down": "🖐↑ Palm scroll down",
    "left_palm_scroll_up": "👈🖐↓ Left palm scroll up",
    "left_palm_scroll_down": "👈🖐↑ Left palm scroll down",
    "right_palm_scroll_up": "👉🖐↓ Right palm scroll up",
    "right_palm_scroll_down": "👉🖐↑ Right palm scroll down",
}

ACTION_OPTIONS = {
    "": "— No action",
    "click": "🖱 click",
    "right_click": "🖱 right click",
    "double_click": "🖱 double click",
    "scroll_up": "↟ scroll up",
    "scroll_down": "↡ scroll down",
    "zoom_in": "＋ zoom +",
    "zoom_out": "－ zoom -",
    "escape": "Esc escape",
    "undo": "↶ undo",
    "toggle_pause": "⏯ active/pause",
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
        self._snap_active = False

        self._flash_timer = QtCore.QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_flash)

    # ── public API ────────────────────────────────────────────────────────
    def update_view(self, cursor: tuple[int, int] | None,
                    dwell_progress: float, state: State,
                    snap_active: bool = False) -> None:
        self._cursor = cursor
        self._dwell_progress = dwell_progress
        self._state = state
        self._snap_active = snap_active
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

        # Snap-to-grid indicator: green dashed ring when snap is active.
        if self._snap_active:
            snap_pen = QtGui.QPen(QtGui.QColor(PALETTE.success), 2)
            snap_pen.setDashPattern([4, 4])
            p.setPen(snap_pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            snap_rect = QtCore.QRect(x - 16, y - 16, 32, 32)
            p.drawEllipse(snap_rect)

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
    bindings_saved = QtCore.pyqtSignal(dict)
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
        self._deduping_bindings = False
        self._saved_bindings: dict[str, str] = {}
        self._minimized = False
        self._user_id = user_id

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self._title = QtWidgets.QLabel(f"FreeHands · {user_id}")
        self._title.setObjectName("fhTitle")
        self._minimize = QtWidgets.QPushButton("−")
        self._minimize.setObjectName("fhMiniButton")
        self._minimize.setFixedSize(30, 28)
        self._minimize.clicked.connect(self._toggle_minimized)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._minimize)
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

        self._button_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._button_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._activate)
        row.addWidget(self._pause)
        row.addWidget(quit_btn)

        self._camera_preview = QtWidgets.QLabel("Camera preview")
        self._camera_preview.setObjectName("fhCamera")
        self._camera_preview.setFixedSize(392, 220)
        self._camera_preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self._hint = QtWidgets.QLabel("Gesture: hold right open palm for 2s to toggle active/paused.")
        self._hint.setWordWrap(True)
        self._hint.setObjectName("fhHint")
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
        self._bindings_scroll = bindings_scroll
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

        layout.addLayout(header)
        layout.addWidget(self._status)
        layout.addWidget(self._camera_preview)
        layout.addWidget(self._button_row)
        layout.addWidget(self._swap_handedness)
        layout.addWidget(self._gaze)
        layout.addWidget(self._gesture)
        layout.addWidget(self._last_action)
        layout.addWidget(self._pause_progress)
        layout.addWidget(bindings_scroll)
        layout.addWidget(self._hint)

        self._detail_widgets = [
            self._camera_preview,
            self._button_row,
            self._swap_handedness,
            self._gaze,
            self._gesture,
            self._last_action,
            self._pause_progress,
            self._status,
            self._bindings_scroll,
            self._hint,
        ]

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
            QPushButton#fhMiniButton {{ padding: 0; font-size: 18px; }}
        """)

        self._place_window()

    def _emit_binding_changed(self, gesture: str) -> None:
        if self._updating_bindings or self._deduping_bindings:
            return
        combo = self._binding_combos[gesture]
        action = combo.currentData() or ""
        if action:
            self._clear_duplicate_action(gesture, action)
        self._saved_bindings = self._selected_bindings()
        self._hint.setText(self._pause_hint(self._saved_bindings))
        self.bindings_saved.emit(dict(self._saved_bindings))

    def _clear_duplicate_action(self, source_gesture: str, action: str) -> None:
        self._deduping_bindings = True
        try:
            for gesture, combo in self._binding_combos.items():
                if gesture != source_gesture and combo.currentData() == action:
                    combo.setCurrentIndex(0)
        finally:
            self._deduping_bindings = False

    def _selected_bindings(self) -> dict[str, str]:
        return {
            gesture: combo.currentData() or ""
            for gesture, combo in self._binding_combos.items()
        }

    def _toggle_minimized(self) -> None:
        self._minimized = not self._minimized
        for widget in self._detail_widgets:
            widget.setVisible(not self._minimized)
        self._minimize.setText("+" if self._minimized else "−")
        if self._minimized:
            self._title.setText(f"FreeHands · {self._status.text()}")
            self.setFixedSize(260, 58)
        else:
            self._title.setText(f"FreeHands · {self._user_id}")
            self.setMinimumHeight(0)
            self.setMaximumHeight(16777215)
            self.setFixedWidth(420)
        self.adjustSize()
        self._place_window()

    def _place_window(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        if self._minimized:
            self.move(screen.right() - self.width() - 24, screen.bottom() - self.height() - 24)
        else:
            self.move(screen.right() - self.width() - 24, screen.top() + 72)

    def set_state(self, state: State) -> None:
        active = state != State.IDLE
        self._status.setText("ACTIVE" if active else "PAUSED")
        if self._minimized:
            self._title.setText(f"FreeHands · {self._status.text()}")
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
        self._saved_bindings = dict(bindings)
        self._hint.setText(self._pause_hint(bindings))
        self._updating_bindings = True
        try:
            for gesture, combo in self._binding_combos.items():
                action = bindings.get(gesture, "")
                index = combo.findData(action)
                combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self._updating_bindings = False

    @staticmethod
    def _pause_hint(bindings: dict[str, str]) -> str:
        for gesture, label in GESTURE_LABELS.items():
            if bindings.get(gesture) == "toggle_pause":
                return f"Gesture: hold {label} for 2s to toggle active/paused."
        return "Gesture: no pause gesture is currently assigned."
