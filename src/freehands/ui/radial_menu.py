"""Radial OSD menu — open-palm hold triggers a circular action picker.

When the user holds an open palm (the gesture mapped to ``toggle_pause`` or
any explicitly-configured gesture) for a configurable dwell time, a translucent
circular menu appears centred on the palm position.  The menu shows frequently-
used actions as icons arranged radially.  The user selects an action by moving
their gaze (cursor) to the desired sector and holding for a short dwell.

Design rules
------------
* Light-mode liquid-glass aesthetic (Ntizar palette).
* Semi-transparent background so the desktop is visible through the menu.
* Maximum 8 actions visible at once (full circle = 360° / 8 = 45° per sector).
* Actions are configurable via the user's profile bindings.
* Menu dismisses on state change to IDLE, on a second palm-hold, or on Escape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import PALETTE


# ── Constants ────────────────────────────────────────────────────────────────

MENU_RADIUS: Final[int] = 140          # outer radius of the menu ring
MENU_ITEM_RADIUS: Final[int] = 36      # radius of each action icon circle
MENU_ITEM_GAP: Final[int] = 6          # gap between icon circles
MENU_DWELL_MS: Final[int] = 400        # dwell to confirm a menu action
MENU_OPEN_DURATION_MS: Final[int] = 1500  # hold open palm to open menu
MENU_ANIM_DURATION_MS: Final[int] = 250  # open/close animation duration
MENU_MAX_ACTIONS: Final[int] = 8       # max actions shown in the radial menu

# Sector hit-test tolerance (degrees)
SECTOR_TOLERANCE: Final[int] = 22      # ±22° per sector (45° total)


# ── Action definitions ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RadialAction:
    """A single action displayed in the radial menu."""
    action_id: str                        # matches dispatcher ACTIONS
    label: str                            # visible label (emoji + text)
    icon: str                             # single-character icon
    color: str = PALETTE.blue             # accent colour

    @property
    def radius(self) -> int:
        return MENU_ITEM_RADIUS


# Default actions shown in the radial menu, in priority order.
DEFAULT_RADIAL_ACTIONS: list[RadialAction] = [
    RadialAction("click",       "Click",        "🖱", PALETTE.blue),
    RadialAction("right_click", "Right click",  "🖲", PALETTE.orange),
    RadialAction("double_click","Double click",  "🖱🖱", PALETTE.blue),
    RadialAction("scroll_up",   "Scroll up",    "↟", PALETTE.success),
    RadialAction("scroll_down", "Scroll down",  "↡", PALETTE.warning),
    RadialAction("zoom_in",     "Zoom in",      "＋", PALETTE.blue_soft),
    RadialAction("zoom_out",    "Zoom out",     "－", PALETTE.orange_soft),
    RadialAction("escape",      "Escape",       "Esc", PALETTE.danger),
]


# ── Menu state machine ─────────────────────────────────────────────────────

@dataclass
class MenuState:
    """Tracks the current state of the radial menu."""
    visible: bool = False
    open_progress: float = 0.0        # 0.0 → 1.0 animation
    selected_action: RadialAction | None = None
    dwell_progress: float = 0.0       # 0.0 → 1.0 for dwell confirmation
    centre_x: int = 0
    centre_y: int = 0
    actions: list[RadialAction] = field(default_factory=list)
    opening: bool = False
    closing: bool = False
    confirmed: bool = False           # action was confirmed this session


# ── Radial menu widget ──────────────────────────────────────────────────────

class RadialMenuWidget(QtWidgets.QWidget):
    """Translucent radial menu drawn over the desktop."""

    action_selected = QtCore.pyqtSignal(str)  # action_id of selected action

    def __init__(self) -> None:
        super().__init__(
            None,
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool,
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # Hide by default
        self.hide()

        self._state = MenuState()
        self._actions = list(DEFAULT_RADIAL_ACTIONS)

        # Animation timer
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._advance_animation)

        # Dwell timer for menu action selection
        self._dwell_timer = QtCore.QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._confirm_dwell)

    # ── public API ───────────────────────────────────────────────────────

    def set_actions(self, actions: list[RadialAction]) -> None:
        """Replace the default action list."""
        self._actions = actions

    def open_at(self, x: int, y: int) -> None:
        """Start opening the menu at the given screen position."""
        self._state = MenuState(
            visible=True,
            centre_x=x,
            centre_y=y,
            actions=self._actions,
            opening=True,
            closing=False,
            confirmed=False,
        )
        self._start_animation(MENU_ANIM_DURATION_MS)
        self.show()

    def close(self) -> None:
        """Start closing the menu."""
        if self._state.closing:
            return
        self._state.closing = True
        self._state.opening = False
        self._dwell_timer.stop()
        self._start_animation(MENU_ANIM_DURATION_MS)

    @property
    def visible(self) -> bool:
        return self._state.visible

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
        self.update()

    def update_dwell(self, cursor_xy: tuple[int, int] | None) -> None:
        """Call each frame with the current cursor position.

        Determines which sector the cursor is in and updates dwell progress.
        """
        if not self._state.visible or self._state.closing:
            return

        cx, cy = self._state.centre_x, self._state.centre_y
        actions = self._state.actions

        if cursor_xy is None:
            # No cursor — reset dwell
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_action = None
            self.update()
            return

        # Find which sector the cursor is in
        selected = self._hit_test(cursor_xy, cx, cy, actions)

        if selected is not None:
            self._state.selected_action = selected
            self._state.dwell_progress = min(1.0,
                self._state.dwell_progress + 1.0 / (MENU_DWELL_MS / 33))
            if self._state.dwell_progress >= 1.0:
                # Dwell complete — confirm!
                self._confirm_dwell()
            else:
                self._dwell_timer.start(33)  # ~30fps dwell updates
        else:
            # Cursor left the menu area — cancel dwell
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_action = None
        self.update()

    def _hit_test(
        self,
        cursor_xy: tuple[int, int],
        cx: int,
        cy: int,
        actions: list[RadialAction],
    ) -> RadialAction | None:
        """Determine which sector the cursor is in, or None."""
        if not actions:
            return None

        dx = cursor_xy[0] - cx
        dy = cursor_xy[1] - cy  # positive = below centre (image coords)
        dist = (dx * dx + dy * dy) ** 0.5

        # Must be within the menu ring area
        if dist < MENU_RADIUS - MENU_ITEM_RADIUS - 10:
            return None
        if dist > MENU_RADIUS + MENU_ITEM_RADIUS + 10:
            return None

        # Calculate angle (0° = top, clockwise)
        angle = math.degrees(math.atan2(dx, -dy))
        if angle < 0:
            angle += 360

        # Which sector?
        n = len(actions)
        sector_size = 360 / n
        sector = int(angle / sector_size) % n
        sector_center = sector * sector_size + sector_size / 2

        # Tolerance check
        diff = abs(angle - sector_center)
        if diff > 180:
            diff = 360 - diff
        if diff <= SECTOR_TOLERANCE:
            return actions[sector]

        return None

    def _confirm_dwell(self) -> None:
        """Dwell completed — emit the selected action."""
        action = self._state.selected_action
        if action is not None and not self._state.confirmed:
            self._state.confirmed = True
            self.action_selected.emit(action.action_id)
            self.close()

    # ── paint ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if not self._state.visible:
            return

        cx, cy = self._state.centre_x, self._state.centre_y
        alpha = int(255 * self._state.open_progress)

        # ── Background ring ──────────────────────────────────────────────
        bg = QtGui.QColor(PALETTE.text_primary)
        bg.setAlpha(alpha // 3)
        p.setBrush(bg)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawEllipse(QtCore.QPoint(cx, cy), MENU_RADIUS, MENU_RADIUS)

        # Inner transparent circle
        inner = QtGui.QColor(255, 255, 255, alpha // 4)
        p.setBrush(inner)
        p.drawEllipse(QtCore.QPoint(cx, cy), MENU_RADIUS - 8, MENU_RADIUS - 8)

        # ── Action sectors ───────────────────────────────────────────────
        actions = self._state.actions
        n = len(actions)
        if n == 0:
            return

        sector_size = 360.0 / n

        for i, action in enumerate(actions):
            angle = i * sector_size + sector_size / 2
            rad = math.radians(angle)

            # Position of icon centre
            ix = cx + int(MENU_RADIUS * math.sin(rad))
            iy = cy - int(MENU_RADIUS * math.cos(rad))

            is_selected = (self._state.selected_action is not None
                           and self._state.selected_action.action_id == action.action_id)

            # Draw sector background
            sector_pen = QtGui.QPen(QtGui.QColor(action.color), 1)
            sector_pen.setAlpha(alpha // 3)
            p.setPen(sector_pen)
            sector_brush = QtGui.QColor(action.color)
            sector_brush.setAlpha(alpha // 8)
            p.setBrush(sector_brush)

            # Draw arc for this sector
            start_angle = int((angle - sector_size / 2) * 16)
            sweep_angle = int(sector_size * 16)
            rect = QtCore.QRect(cx - MENU_RADIUS, cy - MENU_RADIUS,
                                MENU_RADIUS * 2, MENU_RADIUS * 2)
            p.drawArc(rect, start_angle, sweep_angle)

            # Draw icon circle
            icon_color = QtGui.QColor(action.color)
            icon_color.setAlpha(alpha)
            p.setBrush(icon_color)
            p.setPen(QtCore.Qt.PenStyle.NoPen)

            # Highlight selected sector
            if is_selected:
                highlight = QtGui.QColor(PALETTE.orange)
                highlight.setAlpha(int(alpha * 0.6))
                p.setBrush(highlight)
                p.drawEllipse(QtCore.QPoint(ix, iy),
                              MENU_ITEM_RADIUS + 4, MENU_ITEM_RADIUS + 4)

            p.setBrush(icon_color)
            p.drawEllipse(QtCore.QPoint(ix, iy),
                          MENU_ITEM_RADIUS, MENU_ITEM_RADIUS)

            # Draw icon text
            icon_color.setAlpha(alpha)
            p.setPen(QtGui.QPen(icon_color, 1))
            font = p.font()
            font.setPointSize(10)
            font.setBold(True)
            p.setFont(font)
            p.drawText(QtCore.QRect(ix - MENU_ITEM_RADIUS, iy - MENU_ITEM_RADIUS,
                                    MENU_ITEM_RADIUS * 2, MENU_ITEM_RADIUS * 2),
                       QtCore.Qt.AlignmentFlag.AlignCenter, action.icon)

            # Draw label below icon
            label_color = QtGui.QColor(PALETTE.text_primary)
            label_color.setAlpha(alpha)
            p.setPen(QtGui.QPen(label_color, 1))
            font.setPointSize(8)
            p.setFont(font)
            p.drawText(QtCore.QRect(ix - 40, iy + MENU_ITEM_RADIUS + 4,
                                    80, 16),
                       QtCore.Qt.AlignmentFlag.AlignCenter, action.label)

        # ── Dwell progress ring (around selected action) ─────────────────
        if self._state.selected_action is not None:
            sel = self._state.selected_action
            sel_idx = actions.index(sel) if sel in actions else 0
            sel_angle = sel_idx * sector_size + sector_size / 2
            sel_rad = math.radians(sel_angle)
            sel_ix = cx + int(MENU_RADIUS * math.sin(sel_rad))
            sel_iy = cy - int(MENU_RADIUS * math.cos(sel_rad))

            dwell_pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 4)
            dwell_pen.setAlpha(int(255 * self._state.dwell_progress))
            p.setPen(dwell_pen)
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            dwell_rect = QtCore.QRect(sel_ix - 22, sel_iy - 22, 44, 44)
            p.drawArc(dwell_rect, 90 * 16,
                      int(-360 * 16 * self._state.dwell_progress))

    def closeEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        self._anim_timer.stop()
        self._dwell_timer.stop()
