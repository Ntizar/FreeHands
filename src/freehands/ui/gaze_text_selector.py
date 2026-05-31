"""Gaze text selector widget for OCR-based typing.

When the user activates gaze typing mode, the widget captures the screen,
detects text regions, and highlights them with a gaze-following cursor.
Dwell on a text region selects it; the user can then type by looking at
keys in an overlay keyboard, or the system can inject text directly.

This widget follows the same patterns as VirtualKeyboardWidget and
EmojiOverlayWidget: frameless overlay, translucent background, dwell
selection, blink-to-select alternative.

Design rules
------------
* Light-mode liquid-glass aesthetic matching the Ntizar palette.
* Semi-transparent background so the desktop is visible.
* Text regions highlighted with a colored border (blue #1E5BFF).
* Selected region gets an orange (#FF7A1A) border.
* Dwell time configurable (default 1200 ms — longer than keyboard for accuracy).
* Widget dismisses on Escape or a dedicated voice command.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

from PyQt6 import QtCore, QtGui, QtWidgets

from ..ocr.text_region_detector import TextRegion, TextRegionDetector
from .theme import PALETTE

# ── Constants ────────────────────────────────────────────────────────────────

TEXT_SELECTOR_WIDTH: Final[int] = 800
TEXT_SELECTOR_HEIGHT: Final[int] = 600
TEXT_SELECTOR_DWELL_MS: Final[int] = 1200  # longer dwell for text regions
TEXT_SELECTOR_OPEN_DURATION_MS: Final[int] = 200
TEXT_SELECTOR_CLOSE_DURATION_MS: Final[int] = 200
TEXT_SELECTOR_ANIM_STEP: Final[int] = 16  # ms per animation frame (~60fps)

# Highlight border width for selected region
SELECT_BORDER_WIDTH: Final[int] = 3
# Highlight border width for hovered region
HOVER_BORDER_WIDTH: Final[int] = 2

# Pulse animation for dwell ring
DWELL_RING_RADIUS: Final[int] = 25


class SelectorMode(Enum):
    """Current mode of the text selector."""
    SCANNING = auto()     # scanning for text regions
    SELECTING = auto()    # user is selecting a text region to type into
    TYPING = auto()       # typing in the selected region


@dataclass
class SelectorState:
    """Tracks the current state of the gaze text selector."""
    visible: bool = False
    open_progress: float = 0.0
    centre_x: int = 0
    centre_y: int = 0
    regions: list[TextRegion] = field(default_factory=list)
    selected_region: TextRegion | None = None
    hovered_region: TextRegion | None = None
    dwell_progress: float = 0.0
    mode: SelectorMode = SelectorMode.SCANNING
    opening: bool = False
    closing: bool = False
    scan_count: int = 0  # number of scans performed


class GazeTextSelectorWidget(QtWidgets.QWidget):
    """Overlay widget for gaze-based text region selection and typing.

    When activated, the widget:
    1. Captures the screen and detects text regions
    2. Highlights each region with a blue border
    3. Tracks gaze cursor position
    4. When dwell completes on a region, selects it (orange border)
    5. User can then type via the virtual keyboard or voice dictation

    The widget can be opened with the voice command "gaze typing" or
    "escribir texto". Closed with "cerrar escritura" or Escape.
    """

    region_selected = QtCore.pyqtSignal(TextRegion)  # user selected a text region
    text_selector_closed = QtCore.pyqtSignal()
    mode_changed = QtCore.pyqtSignal(str)  # "scanning", "selecting", "typing"

    def __init__(self, dwell_ms: int = TEXT_SELECTOR_DWELL_MS) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self.hide()

        self._dwell_ms = dwell_ms
        self._state = SelectorState()
        self._detector = TextRegionDetector()
        self._scan_timer = QtCore.QTimer(self)
        self._scan_timer.setSingleShot(False)
        self._scan_timer.timeout.connect(self._scan_regions)
        self._scan_timer.start(2000)  # scan every 2 seconds

        # Animation timer
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._advance_animation)

        # Dwell timer for region selection
        self._dwell_timer = QtCore.QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._confirm_region)

    # ── public API ───────────────────────────────────────────────────────

    @property
    def visible(self) -> bool:
        return self._state.visible

    @property
    def mode(self) -> SelectorMode:
        return self._state.mode

    @property
    def selected_region(self) -> TextRegion | None:
        return self._state.selected_region

    @property
    def region_count(self) -> int:
        return len(self._state.regions)

    def open_at(self, x: int, y: int) -> None:
        """Start opening the text selector centred at the given position."""
        screen_w = max(1920, x * 2)
        centre_x = min(max(x, TEXT_SELECTOR_WIDTH // 2 + 20),
                       screen_w - TEXT_SELECTOR_WIDTH // 2 - 20)
        centre_y = min(max(y, TEXT_SELECTOR_HEIGHT // 2 + 20),
                       1080 - TEXT_SELECTOR_HEIGHT // 2 - 20)

        self._state = SelectorState(
            visible=True,
            centre_x=centre_x,
            centre_y=centre_y,
            regions=list(self._detector.regions),
            opening=True,
            closing=False,
            mode=SelectorMode.SCANNING,
            scan_count=0,
        )
        self._start_animation(TEXT_SELECTOR_OPEN_DURATION_MS)
        self.show()
        self.mode_changed.emit("scanning")

    def close(self) -> None:
        """Start closing the text selector."""
        if self._state.closing:
            return
        self._state.closing = True
        self._state.opening = False
        self._dwell_timer.stop()
        self._start_animation(TEXT_SELECTOR_CLOSE_DURATION_MS)

    def update_dwell(self, cursor_xy: tuple[int, int] | None) -> None:
        """Call each frame with the current cursor position.

        Determines which text region the cursor is in and updates dwell progress.
        """
        if not self._state.visible or self._state.closing:
            return

        cx, cy = self._state.centre_x, self._state.centre_y

        if cursor_xy is None:
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.hovered_region = None
            self.update()
            return

        # Check if cursor is over any text region
        hovered = self._hit_test(cursor_xy, cx, cy)

        if hovered is not None:
            self._state.hovered_region = hovered
            dwell_steps = self._dwell_ms / TEXT_SELECTOR_ANIM_STEP
            self._state.dwell_progress = min(
                1.0,
                self._state.dwell_progress + 1.0 / dwell_steps,
            )
            if self._state.dwell_progress >= 1.0:
                self._confirm_region()
            else:
                self._dwell_timer.start(TEXT_SELECTOR_ANIM_STEP)
        else:
            # Cursor left the region area — cancel dwell
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.hovered_region = None
            self.update()

    def process_blink(self, blink: bool) -> None:
        """Process a blink event for blink-to-select mode.

        A blink can confirm the hovered region selection.
        """
        if blink and self._state.hovered_region is not None:
            # Blink confirms current hover
            self._confirm_region()

    # ── scanning ─────────────────────────────────────────────────────────

    def _scan_regions(self) -> None:
        """Scan the screen for text regions."""
        if not self._state.visible:
            return
        regions = self._detector.detect()
        self._state.regions = regions
        self._state.scan_count += 1
        self.update()

    # ── selection ────────────────────────────────────────────────────────

    def _hit_test(
        self,
        cursor_xy: tuple[int, int],
        cx: int,
        cy: int,
    ) -> TextRegion | None:
        """Determine which text region the cursor is in, or None."""
        for region in self._state.regions:
            screen_x = cx - TEXT_SELECTOR_WIDTH // 2 + region.x
            screen_y = cy - TEXT_SELECTOR_HEIGHT // 2 + region.y
            if (screen_x <= cursor_xy[0] < screen_x + region.width and
                    screen_y <= cursor_xy[1] < screen_y + region.height):
                return region
        return None

    def _confirm_region(self) -> None:
        """Region dwell completed — select the hovered region."""
        region = self._state.hovered_region
        if region is None:
            return

        self._state.selected_region = region
        self._state.hovered_region = None
        self._state.dwell_progress = 0.0
        self._dwell_timer.stop()

        self.region_selected.emit(region)
        self.update()

    # ── animation ────────────────────────────────────────────────────────

    def _start_animation(self, duration_ms: int) -> None:
        self._anim_timer.start(duration_ms)

    def _advance_animation(self) -> None:
        if self._state.opening:
            self._state.open_progress = 1.0
            self._state.opening = False
        elif self._state.closing:
            self._state.open_progress = 0.0
            self._state.closing = False
            self._state.visible = False
            self.hide()
            self.text_selector_closed.emit()
        self.update()

    # ── painting ─────────────────────────────────────────────────────────

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Paint the text selector overlay with highlighted regions."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Semi-transparent background
        alpha = int(200 * self._state.open_progress)
        painter.fillRect(self.rect(), QtGui.QColor(255, 255, 255, alpha))

        if not self._state.visible or not self._state.regions:
            painter.end()
            return

        cx, cy = self._state.centre_x, self._state.centre_y

        # Draw text region highlights
        for region in self._state.regions:
            screen_x = cx - TEXT_SELECTOR_WIDTH // 2 + region.x
            screen_y = cy - TEXT_SELECTOR_HEIGHT // 2 + region.y

            # Check if this is the selected region
            is_selected = (self._state.selected_region is not None and
                           self._state.selected_region.x == region.x and
                           self._state.selected_region.y == region.y)

            # Check if this is the hovered region
            is_hovered = (self._state.hovered_region is not None and
                          self._state.hovered_region.x == region.x and
                          self._state.hovered_region.y == region.y)

            if is_selected:
                # Orange border for selected
                pen = QtGui.QPen(PALETTE['accent'], SELECT_BORDER_WIDTH)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor(PALETTE['accent'].red(),
                                              PALETTE['accent'].green(),
                                              PALETTE['accent'].blue(), 40))
                painter.drawRect(screen_x, screen_y, region.width, region.height)
            elif is_hovered:
                # Blue border with dwell ring for hovered
                pen = QtGui.QPen(PALETTE['primary'], HOVER_BORDER_WIDTH)
                painter.setPen(pen)
                painter.setBrush(QtGui.QColor(PALETTE['primary'].red(),
                                              PALETTE['primary'].green(),
                                              PALETTE['primary'].blue(), 30))
                painter.drawRect(screen_x, screen_y, region.width, region.height)

                # Draw dwell progress ring around the region
                centre_x = screen_x + region.width // 2
                centre_y = screen_y + region.height // 2
                radius = max(region.width, region.height) // 2 + DWELL_RING_RADIUS
                start_angle = -90 * 16  # start from top
                span_angle = int(360 * 16 * self._state.dwell_progress)
                painter.drawArc(
                    centre_x - radius, centre_y - radius,
                    radius * 2, radius * 2,
                    start_angle, span_angle,
                )
            else:
                # Faint blue border for unselected regions
                pen = QtGui.QPen(PALETTE['primary'], 1)
                pen.setAlpha(80)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.GlobalColor.transparent)
                painter.drawRect(screen_x, screen_y, region.width, region.height)

        # Draw info text
        font = QtGui.QFont("sans-serif", 10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(80, 80, 80))
        info_y = 20
        painter.drawText(10, info_y,
                         f"Regiones de texto detectadas: {self._state.region_count}")
        if self._state.selected_region:
            painter.drawText(10, info_y + 20,
                             f"Seleccionado: ({self._state.selected_region.x}, "
                             f"{self._state.selected_region.y}) "
                             f"{self._state.selected_region.width}x"
                             f"{self._state.selected_region.height}")

        painter.end()

    # ── cleanup ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the widget and reset state."""
        super().close()
        self._detector.clear()
