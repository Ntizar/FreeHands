"""Camera picker with live gaze/hand diagnostics."""
from __future__ import annotations

import sys

from PyQt6 import QtCore, QtGui, QtWidgets

from ..capture import Camera, list_available_cameras
from ..gaze import GazeTracker
from ..gestures import HandTracker
from ..profiles import get_or_create_profile, save_profile
from .theme import GLOBAL_STYLESHEET, PALETTE


class CameraSelector(QtWidgets.QWidget):
    def __init__(self, user_id: str) -> None:
        super().__init__()
        self.user_id = user_id
        self.profile = get_or_create_profile(user_id)
        self.camera: Camera | None = None
        self.gaze = GazeTracker()
        self.hands = HandTracker()

        self.setWindowTitle("FreeHands · Camera")
        self.setMinimumSize(880, 560)

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(18)

        self.preview = QtWidgets.QLabel("Opening camera...")
        self.preview.setMinimumSize(640, 480)
        self.preview.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#101426;color:white;border-radius:16px;")

        side = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel(f"Camera for {user_id}")
        title.setStyleSheet(f"font-size:24px;font-weight:800;color:{PALETTE.blue};")
        self.combo = QtWidgets.QComboBox()
        self.combo.currentIndexChanged.connect(self._combo_changed)
        self.status = QtWidgets.QLabel("Choose the camera that sees your face and eyes.")
        self.status.setWordWrap(True)
        self.gaze_info = QtWidgets.QLabel("Gaze: -")
        self.gaze_info.setWordWrap(True)
        self.hand_info = QtWidgets.QLabel("Hands: -")
        self.hand_info.setWordWrap(True)

        save_btn = QtWidgets.QPushButton("Save this camera")
        save_btn.clicked.connect(self._save)
        next_btn = QtWidgets.QPushButton("Next camera")
        next_btn.clicked.connect(self._next_camera)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(QtWidgets.QApplication.instance().quit)

        side.addWidget(title)
        side.addWidget(self.combo)
        side.addWidget(self.status)
        side.addSpacing(8)
        side.addWidget(self.gaze_info)
        side.addWidget(self.hand_info)
        side.addStretch()
        side.addWidget(save_btn)
        side.addWidget(next_btn)
        side.addWidget(close_btn)

        root.addWidget(self.preview, stretch=1)
        root.addLayout(side)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(90)
        self.timer.timeout.connect(self._tick)

        self._populate()
        self.timer.start()

    def _populate(self) -> None:
        found = set(list_available_cameras(max_index=8))
        indices = sorted(found | {self.profile.camera_index}) or list(range(8))
        for index in indices:
            suffix = "detected" if index in found else "try"
            self.combo.addItem(f"Camera {index} · {suffix}", index)
        current = self.combo.findData(self.profile.camera_index)
        self.combo.setCurrentIndex(max(current, 0))
        self._open_selected()

    def _combo_changed(self) -> None:
        self._open_selected()

    def _open_selected(self) -> None:
        index = int(self.combo.currentData())
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        try:
            self.camera = Camera(index).start()
            self.status.setText(f"Testing camera {index}. If your eyes are highlighted, save this camera.")
        except Exception as exc:
            self.status.setText(f"Could not open camera {index}: {exc}")
            self.preview.setText("No image")

    def _next_camera(self) -> None:
        if self.combo.count() == 0:
            return
        self.combo.setCurrentIndex((self.combo.currentIndex() + 1) % self.combo.count())

    def _save(self) -> None:
        index = int(self.combo.currentData())
        self.profile.camera_index = index
        save_profile(self.profile)
        self.status.setText(f"Saved camera {index}. Calibration and FreeHands will use this camera.")

    def _tick(self) -> None:
        if self.camera is None:
            return
        frame = self.camera.read()
        if frame is None:
            return
        image = frame.image.copy()
        gaze_features = self.gaze.extract(image)
        hand_obs = self.hands.detect(image)
        debug = self.gaze.last_debug

        rgb = image[:, :, ::-1].copy()
        h, w, _ = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        scale_x = pix.width() / w
        scale_y = pix.height() / h
        if debug.points:
            def point(name: str) -> QtCore.QPoint:
                x, y = debug.points[name]
                return QtCore.QPoint(int(x * scale_x), int(y * scale_y))
            painter.setPen(QtGui.QPen(QtGui.QColor(PALETTE.orange), 3))
            painter.drawLine(point("left_outer"), point("left_inner"))
            painter.drawLine(point("right_inner"), point("right_outer"))
            painter.setBrush(QtGui.QColor(PALETTE.blue))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for name in ("left_iris", "right_iris", "nose"):
                painter.drawEllipse(point(name), 6, 6)
        if hand_obs.hands:
            painter.setPen(QtGui.QPen(QtGui.QColor("#1a7f37"), 2))
            painter.drawText(14, 24, f"hand: {hand_obs.gesture} {hand_obs.confidence:.2f}")
        painter.end()
        self.preview.setPixmap(pix)

        self.gaze_info.setText(
            f"Gaze: {debug.message}\n"
            f"face={debug.face_detected} landmarks={debug.landmark_count} "
            f"iris={debug.iris_detected} pupil={debug.pupil_detected} conf={debug.confidence:.2f}"
        )
        self.hand_info.setText(f"Hands: {hand_obs.gesture} · {hand_obs.confidence:.2f} · hands={len(hand_obs.hands)}")
        if gaze_features is not None:
            self.status.setText("This camera can see your eyes. You can save it.")

    def closeEvent(self, event):  # noqa: N802
        self.timer.stop()
        if self.camera is not None:
            self.camera.stop()
        self.gaze.close()
        self.hands.close()
        super().closeEvent(event)


def run_camera_selector(user_id: str) -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_STYLESHEET)
    win = CameraSelector(user_id)
    win.show()
    return app.exec()