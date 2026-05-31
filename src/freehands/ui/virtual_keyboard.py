"""Virtual keyboard with gaze-based dwell selection.

When the virtual keyboard is active, the user types by looking at keys.
A dwell ring fills around the key under the gaze cursor; when full, the
key is "pressed" and its character is typed into the currently focused
text field via pyautogui.

Design rules
------------
* Light-mode liquid-glass aesthetic matching the Ntizar palette.
* Semi-transparent background so the desktop is visible through the keyboard.
* Standard QWERTY layout with shift, space, enter, backspace.
* Dwell time configurable (default 800 ms).
* Keyboard dismisses on Escape or a dedicated "close keyboard" dwell.
* All text typed via pyautogui hotkey/press so it works in any focused field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import PALETTE

# ── Constants ────────────────────────────────────────────────────────────────

KEYBOARD_WIDTH: Final[int] = 680
KEYBOARD_HEIGHT: Final[int] = 280
KEY_RADIUS: Final[int] = 22               # half-width of a key cell
KEY_GAP: Final[int] = 4                   # gap between keys
KEYBOARD_DWELL_MS: Final[int] = 800       # dwell to confirm a key
KEYBOARD_OPEN_DURATION_MS: Final[int] = 200  # open animation duration
KEYBOARD_CLOSE_DURATION_MS: Final[int] = 200  # close animation duration
KEYBOARD_ANIM_STEP: Final[int] = 16       # ms per animation frame (~60fps)

# Key dimensions
KEY_WIDTH: Final[int] = 52
KEY_HEIGHT: Final[int] = 44

# ── Keyboard layout ─────────────────────────────────────────────────────────

# Each row is a list of (label, action_or_char).
# action_or_char: a single character to type, or a special action name.
KEYBOARD_ROWS: Final[list[list[tuple[str, str]]]] = [
    [
        ("Q", "q"), ("W", "w"), ("E", "e"), ("R", "r"), ("T", "t"),
        ("Y", "y"), ("U", "u"), ("I", "i"), ("O", "o"), ("P", "p"),
    ],
    [
        ("A", "a"), ("S", "s"), ("D", "d"), ("F", "f"), ("G", "g"),
        ("H", "h"), ("J", "j"), ("K", "k"), ("L", "l"), ("Ñ", "ñ"),
    ],
    [
        ("Z", "z"), ("X", "x"), ("C", "c"), ("V", "v"), ("B", "b"),
        ("N", "n"), ("M", "m"), ("", "backspace"),
    ],
    [
        ("⇧", "shift"), ("", "space"), ("⏎", "enter"), ("⌫", "backspace"),
    ],
]


@dataclass(frozen=True)
class KeyDefinition:
    """A single key on the virtual keyboard."""
    label: str                     # visible label
    char: str                      # character to type (or action name)
    x: int                         # screen x position (center)
    y: int                         # screen y position (center)
    width: int = KEY_WIDTH
    height: int = KEY_HEIGHT
    is_special: bool = False       # True for shift/space/backspace/enter

    @property
    def rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self.x - self.width // 2,
            self.y - self.height // 2,
            self.width,
            self.height,
        )

    def contains_point(self, px: int, py: int) -> bool:
        return self.rect.contains(QtCore.QPoint(px, py))


# ── Keyboard state ──────────────────────────────────────────────────────────

@dataclass
class KeyboardState:
    """Tracks the current state of the virtual keyboard."""
    visible: bool = False
    open_progress: float = 0.0    # 0.0 → 1.0 animation
    centre_x: int = 0
    centre_y: int = 0
    keys: list[KeyDefinition] = field(default_factory=list)
    selected_key: KeyDefinition | None = None
    dwell_progress: float = 0.0   # 0.0 → 1.0 for dwell confirmation
    shift_active: bool = False    # uppercase mode
    opening: bool = False
    closing: bool = False
    confirmed_keys: list[str] = field(default_factory=list)  # last confirmed this frame


# ── Virtual keyboard widget ─────────────────────────────────────────────────

class VirtualKeyboardWidget(QtWidgets.QWidget):
    """Translucent virtual keyboard drawn over the desktop.

    The keyboard appears centred on screen. The user navigates with gaze
    (the cursor highlight follows the key under gaze). Dwell confirms
    the key. Shift toggles uppercase. Space types a space. Enter sends
    Enter. Backspace deletes the last character.
    """

    key_pressed = QtCore.pyqtSignal(str)  # character or action name
    keyboard_closed = QtCore.pyqtSignal()

    def __init__(self, dwell_ms: int = KEYBOARD_DWELL_MS) -> None:
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
        self._state = KeyboardState()
        self._shift_active = False
        self._typing_buffer: list[str] = []  # characters typed this session

        # Animation timer
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._advance_animation)

        # Dwell timer for key selection
        self._dwell_timer = QtCore.QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._confirm_dwell)

        # Build key definitions
        self._build_keys()

    # ── public API ───────────────────────────────────────────────────────

    def _build_keys(self) -> None:
        """Build the key grid from KEYBOARD_ROWS."""
        self._keys: list[KeyDefinition] = []
        row_y_start = 50  # top padding

        for row_idx, row in enumerate(KEYBOARD_ROWS):
            row_y = row_y_start + row_idx * (KEY_HEIGHT + KEY_GAP)
            # Calculate total width of this row
            row_width = len(row) * KEY_WIDTH + (len(row) - 1) * KEY_GAP
            x_start = (KEYBOARD_WIDTH - row_width) // 2

            for col_idx, (label, char) in enumerate(row):
                x = x_start + col_idx * (KEY_WIDTH + KEY_GAP) + KEY_WIDTH // 2
                is_special = char in {"shift", "space", "enter", "backspace"}
                key = KeyDefinition(
                    label=label,
                    char=char,
                    x=x,
                    y=row_y,
                    width=KEY_WIDTH,
                    height=KEY_HEIGHT,
                    is_special=is_special,
                )
                self._keys.append(key)

    def open_at(self, x: int, y: int) -> None:
        """Start opening the keyboard centred at the given position."""
        screen_w = max(1920, x * 2)  # reasonable default
        centre_x = min(max(x, KEYBOARD_WIDTH // 2 + 20), screen_w - KEYBOARD_WIDTH // 2 - 20)
        centre_y = min(max(y, KEYBOARD_HEIGHT // 2 + 20), 1080 - KEYBOARD_HEIGHT // 2 - 20)

        self._state = KeyboardState(
            visible=True,
            centre_x=centre_x,
            centre_y=centre_y,
            keys=list(self._keys),
            opening=True,
            closing=False,
            confirmed_keys=[],
        )
        self._start_animation(KEYBOARD_OPEN_DURATION_MS)
        self.show()
        self._typing_buffer = []

    def close(self) -> None:
        """Start closing the keyboard."""
        if self._state.closing:
            return
        self._state.closing = True
        self._state.opening = False
        self._dwell_timer.stop()
        self._start_animation(KEYBOARD_CLOSE_DURATION_MS)

    @property
    def visible(self) -> bool:
        return self._state.visible

    @property
    def shift_active(self) -> bool:
        return self._shift_active

    def set_shift(self, active: bool) -> None:
        self._shift_active = active
        self.update()

    @property
    def typing_buffer(self) -> str:
        return "".join(self._typing_buffer)

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
            self.keyboard_closed.emit()
        self.update()

    def update_dwell(self, cursor_xy: tuple[int, int] | None) -> None:
        """Call each frame with the current cursor position.

        Determines which key the cursor is in and updates dwell progress.
        """
        if not self._state.visible or self._state.closing:
            return

        cx, cy = self._state.centre_x, self._state.centre_y
        keys = self._state.keys

        if cursor_xy is None:
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
            self.update()
            return

        # Find which key the cursor is in
        selected = self._hit_test(cursor_xy, cx, cy, keys)

        if selected is not None:
            self._state.selected_key = selected
            dwell_steps = self._dwell_ms / KEYBOARD_ANIM_STEP
            self._state.dwell_progress = min(
                1.0,
                self._state.dwell_progress + 1.0 / dwell_steps,
            )
            if self._state.dwell_progress >= 1.0:
                self._confirm_dwell()
            else:
                self._dwell_timer.start(KEYBOARD_ANIM_STEP)
        else:
            # Cursor left the keyboard area — cancel dwell
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
        self.update()

    def _hit_test(
        self,
        cursor_xy: tuple[int, int],
        cx: int,
        cy: int,
        keys: list[KeyDefinition],
    ) -> KeyDefinition | None:
        """Determine which key the cursor is in, or None."""
        for key in keys:
            # Transform key position from local to screen coords
            screen_x = cx - KEYBOARD_WIDTH // 2 + key.x
            screen_y = cy - KEYBOARD_HEIGHT // 2 + key.y
            key_rect = QtCore.QRect(
                screen_x - key.width // 2,
                screen_y - key.height // 2,
                key.width,
                key.height,
            )
            if key_rect.contains(QtCore.QPoint(cursor_xy[0], cursor_xy[1])):
                return key
        return None

    def _confirm_dwell(self) -> None:
        """Dwell completed — process the selected key."""
        key = self._state.selected_key
        if key is None:
            return

        char_to_type = key.char

        # Handle shift: toggle uppercase
        if char_to_type == "shift":
            self._shift_active = not self._shift_active
            self.key_pressed.emit("shift")
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
            self.update()
            return

        # Handle backspace
        if char_to_type == "backspace":
            if self._typing_buffer:
                self._typing_buffer.pop()
            self.key_pressed.emit("backspace")
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
            self.update()
            return

        # Handle space
        if char_to_type == "space":
            self._typing_buffer.append(" ")
            self.key_pressed.emit("space")
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
            self.update()
            return

        # Handle enter
        if char_to_type == "enter":
            self._typing_buffer.append("\n")
            self.key_pressed.emit("enter")
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_key = None
            self.update()
            return

        # Regular character
        if len(char_to_type) == 1:
            if self._shift_active:
                char_to_type = char_to_type.upper()
                self._shift_active = False  # auto-reset shift
            self._typing_buffer.append(char_to_type)
            self.key_pressed.emit(char_to_type)

        self._dwell_timer.stop()
        self._state.dwell_progress = 0.0
        self._state.selected_key = None
        self.update()

    # ── paint ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        if not self._state.visible:
            return

        alpha = int(255 * self._state.open_progress)
        cx, cy = self._state.centre_x, self._state.centre_y

        # ── Background panel ─────────────────────────────────────────────
        bg_x = cx - KEYBOARD_WIDTH // 2
        bg_y = cy - KEYBOARD_HEIGHT // 2

        # Semi-transparent background
        bg_color = QtGui.QColor(255, 255, 255)
        bg_color.setAlpha(alpha // 3)
        p.setBrush(bg_color)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(
            bg_x - 16, bg_y - 16,
            KEYBOARD_WIDTH + 32, KEYBOARD_HEIGHT + 32,
            16, 16,
        )

        # Subtle border
        border_color = QtGui.QColor(PALETTE.blue)
        border_color.setAlpha(alpha // 4)
        p.setPen(QtGui.QPen(border_color, 2))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(
            bg_x - 16, bg_y - 16,
            KEYBOARD_WIDTH + 32, KEYBOARD_HEIGHT + 32,
            16, 16,
        )

        # Title bar hint
        title_color = QtGui.QColor(PALETTE.blue)
        title_color.setAlpha(alpha)
        p.setPen(QtGui.QPen(title_color, 1))
        font = p.font()
        font.setPointSize(9)
        font.setBold(True)
        p.setFont(font)
        p.drawText(
            bg_x + 16, bg_y - 6,
            KEYBOARD_WIDTH - 32, 16,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "Teclado virtual — mira una tecla para escribir",
        )

        # ── Keys ─────────────────────────────────────────────────────────
        keys = self._state.keys

        for key in keys:
            screen_x = cx - KEYBOARD_WIDTH // 2 + key.x
            screen_y = cy - KEYBOARD_HEIGHT // 2 + key.y

            is_selected = (
                self._state.selected_key is not None
                and self._state.selected_key.char == key.char
                and self._state.selected_key.label == key.label
            )

            # Key background
            key_color = QtGui.QColor(255, 255, 255)
            key_color.setAlpha(alpha // 2)

            if is_selected:
                # Highlight: orange glow
                highlight = QtGui.QColor(PALETTE.orange)
                highlight.setAlpha(int(alpha * 0.5))
                p.setBrush(highlight)
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.drawRoundedRect(
                    screen_x - key.width // 2 - 2,
                    screen_y - key.height // 2 - 2,
                    key.width + 4,
                    key.height + 4,
                    8, 8,
                )

            p.setBrush(key_color)
            p.setPen(QtGui.QPen(
                QtGui.QColor(PALETTE.blue),
                1,
            ))
            p.setPen(QtGui.QPen(
                QtGui.QColor(PALETTE.blue).lighter(140),
                1,
            ))
            p.drawRoundedRect(
                screen_x - key.width // 2,
                screen_y - key.height // 2,
                key.width,
                key.height,
                8, 8,
            )

            # Key text
            text_color = QtGui.QColor(PALETTE.text_primary)
            text_color.setAlpha(alpha)
            p.setPen(QtGui.QPen(text_color, 1))

            font = p.font()
            font.setPointSize(12 if len(key.label) <= 2 else 10)
            font.setBold(is_selected)
            p.setFont(font)

            # Show shifted character if shift is active
            display_label = key.label
            if self._shift_active and len(key.char) == 1 and key.char.isalpha():
                display_label = key.char.upper()

            p.drawText(
                screen_x - key.width // 2 + 2,
                screen_y - key.height // 2 + 2,
                key.width - 4,
                key.height - 4,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                display_label,
            )

            # ── Dwell ring around selected key ───────────────────────────
            if is_selected and self._state.dwell_progress > 0:
                dwell_pen = QtGui.QPen(
                    QtGui.QColor(PALETTE.orange),
                    4,
                )
                dwell_pen.setAlpha(int(255 * self._state.dwell_progress))
                p.setPen(dwell_pen)
                p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                dwell_rect = QtCore.QRect(
                    screen_x - 22, screen_y - 22,
                    44, 44,
                )
                p.drawArc(
                    dwell_rect,
                    90 * 16,
                    int(-360 * 16 * self._state.dwell_progress),
                )

        # ── Typing buffer preview ────────────────────────────────────────
        if self._typing_buffer:
            preview_text = "".join(self._typing_buffer[-30:])
            preview_color = QtGui.QColor(PALETTE.text_primary)
            preview_color.setAlpha(alpha)
            p.setPen(QtGui.QPen(preview_color, 1))
            font = p.font()
            font.setPointSize(10)
            p.setFont(font)
            p.drawText(
                bg_x + 16, bg_y + KEYBOARD_HEIGHT - 16,
                KEYBOARD_WIDTH - 32, 16,
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"Texto: {preview_text}",
            )

    def closeEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        self._anim_timer.stop()
        self._dwell_timer.stop()


# ── Helper: build keyboard at screen centre ──────────────────────────────────

def build_keyboard_at_screen_center(screen_geometry: QtCore.QRect) -> VirtualKeyboardWidget:
    """Create a keyboard widget centred on the given screen geometry."""
    kb = VirtualKeyboardWidget()
    cx = screen_geometry.left() + screen_geometry.width() // 2
    cy = screen_geometry.top() + screen_geometry.height() // 2
    kb.open_at(cx, cy)
    return kb
