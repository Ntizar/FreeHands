"""Magnifier overlay — zoom lens over the gaze cursor area.

Inspired by VocalIris-OS. Renders a circular magnifying glass
that enlarges the desktop region around the cursor so the user
can read small text more easily.

Uses a Qt widget with a frameless, click-through window that
captures the screen region, scales it up, and paints it as a
circle with a subtle border ring.
"""
from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import PALETTE


class MagnifierWidget(QtWidgets.QWidget):
    """Circular magnifier overlay drawn over the desktop.

    The widget sits on top of everything else (WindowStaysOnTopHint),
    is transparent to mouse events (WA_TransparentForMouseEvents),
    and paints a scaled-up view of the screen region around the
    current gaze cursor position.

    Parameters
    ----------
    zoom_factor : float
        Magnification factor (1.5 – 4.0 recommended).
    radius : int
        Radius of the magnifier circle in pixels.
    offset_x / offset_y : int
        Offset from the cursor position (default: cursor is at
        the bottom of the magnifier, so offset_y is positive).
    """

    def __init__(
        self,
        zoom_factor: float = 2.0,
        radius: int = 100,
        offset_x: int = 0,
        offset_y: int = -140,
    ) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setWindowOpacity(0.95)

        self._zoom_factor = max(1.1, min(4.0, zoom_factor))
        self._radius = max(40, radius)
        self._offset_x = offset_x
        self._offset_y = offset_y

        self._cursor: tuple[int, int] | None = None
        self._screen: QtWidgets.QScreen | None = None
        self._cached_pixmap: QtGui.QPixmap | None = None
        self._last_screen_rect: QtCore.QRect | None = None

        # Timer for periodic refresh (avoids stale screen capture)
        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setInterval(80)  # ~12 Hz refresh
        self._refresh_timer.timeout.connect(self._refresh_if_needed)
        self._refresh_timer.start()

        # Track screen changes
        self._init_screen()

    def _init_screen(self) -> None:
        """Initialize or re-initialize the screen reference."""
        self._screen = QtWidgets.QApplication.primaryScreen()
        if self._screen is not None:
            self._last_screen_rect = self._screen.geometry()

    def _refresh_if_needed(self) -> None:
        """Re-capture the screen if the cursor moved or screen changed."""
        if self._cursor is None:
            return
        # Check if screen geometry changed
        if self._screen is not None:
            current_rect = self._screen.geometry()
            if self._last_screen_rect != current_rect:
                self._last_screen_rect = current_rect
                self._cached_pixmap = None

        self.update()

    def update_cursor(self, cursor: tuple[int, int] | None) -> None:
        """Update the gaze cursor position.

        Parameters
        ----------
        cursor : tuple[int, int] | None
            (x, y) screen coordinates of the gaze cursor, or None
            to hide the magnifier.
        """
        self._cursor = cursor
        self._cached_pixmap = None  # force re-capture
        if cursor is not None:
            self.show()
            self.update()
        else:
            self.hide()

    def set_zoom_factor(self, factor: float) -> None:
        """Change the magnification factor at runtime."""
        self._zoom_factor = max(1.1, min(4.0, factor))
        self._cached_pixmap = None
        self.update()

    def set_radius(self, radius: int) -> None:
        """Change the magnifier radius at runtime."""
        self._radius = max(40, radius)
        self._cached_pixmap = None
        self.update()

    def _capture_screen_region(
        self, center: tuple[int, int]
    ) -> QtGui.QPixmap | None:
        """Capture the screen region around the given center point.

        Returns a QPixmap of the unscaled region, or None if capture
        fails (e.g. screen not available).
        """
        if self._screen is None:
            return None

        cx, cy = center
        # The region to capture is the magnifier radius divided by zoom factor
        # — this gives us the source area that will fill the magnifier circle
        source_radius = int(self._radius / self._zoom_factor)
        source_radius = max(10, source_radius)

        rect = QtCore.QRect(
            cx - source_radius,
            cy - source_radius,
            source_radius * 2,
            source_radius * 2,
        )

        # Clamp to screen bounds
        screen_rect = self._screen.geometry()
        rect = rect.intersected(screen_rect)

        if rect.isEmpty():
            return None

        pixmap = self._screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
        return pixmap

    # ── paint ───────────────────────────────────────────────────────────────
    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if self._cursor is None:
            return

        cx, cy = self._cursor
        mx = cx + self._offset_x
        my = cy + self._offset_y
        diameter = self._radius * 2

        # Resize widget to fit the magnifier circle
        self.setGeometry(mx - self._radius, my - self._radius, diameter, diameter)

        # Capture screen region
        screen_pixmap = self._capture_screen_region((cx, cy))
        if screen_pixmap is None or screen_pixmap.isNull():
            return

        # Scale up the captured region
        scaled = screen_pixmap.scaled(
            diameter, diameter,
            QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        # Clip to circle
        p.setClipRegion(QtGui.QRegion(QtCore.QPoint(self._radius, self._radius), self._radius))

        # Draw the magnified content
        p.drawPixmap(0, 0, scaled)

        # Glass highlight (top-left gradient)
        highlight = QtGui.QRadialGradient(
            self._radius * 0.6, self._radius * 0.4, self._radius * 0.8
        )
        highlight.setColorAt(0, QtGui.QColor(255, 255, 255, 80))
        highlight.setColorAt(1, QtGui.QColor(255, 255, 255, 0))
        p.setBrush(QtGui.QBrush(highlight))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QRect(0, 0, diameter, diameter))

        # Border ring
        border_pen = QtGui.QPen(
            QtGui.QColor(PALETTE.blue).lighter(110), 2
        )
        border_pen.setCosmetic(True)
        p.setPen(border_pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(QtCore.QRect(1, 1, diameter - 2, diameter - 2))

        # Outer glow
        glow_pen = QtGui.QPen(
            QtGui.QColor(PALETTE.blue).lighter(130), 1
        )
        glow_pen.setCosmetic(True)
        p.setPen(glow_pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(QtCore.QRect(0, 0, diameter, diameter))

        p.end()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._refresh_timer.stop()
        super().closeEvent(event)
