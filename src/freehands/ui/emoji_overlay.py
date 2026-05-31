"""Emoji overlay — gaze-navigable emoji picker with voice confirmation.

When the emoji overlay is active, the user browses emoji categories and
selects emojis by looking at them (dwell) or double-blinking.  A voice
command ("emojis", "abrir emojis") opens the overlay; "cerrar emojis"
closes it.

Design rules
------------
* Light-mode liquid-glass aesthetic matching the Ntizar palette.
* Semi-transparent background so the desktop is visible through the overlay.
* Categories shown as rows; emojis shown as large clickable tiles.
* Dwell time configurable (default 600 ms — shorter than keyboard for speed).
* Blink-to-select as alternative to dwell.
* Overlay dismisses on Escape, voice command, or state change to IDLE.
* Emojis are typed into the focused field via pyautogui.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Final

from PyQt6 import QtCore, QtGui, QtWidgets

from .theme import PALETTE

# ── Constants ────────────────────────────────────────────────────────────────

EMOJI_OVERLAY_WIDTH: Final[int] = 720
EMOJI_OVERLAY_HEIGHT: Final[int] = 480
EMOJI_TILE_SIZE: Final[int] = 52       # px per emoji tile
EMOJI_TILE_GAP: Final[int] = 4         # gap between tiles
EMOJI_OVERLAY_DWELL_MS: Final[int] = 600  # dwell to confirm an emoji
EMOJI_OPEN_DURATION_MS: Final[int] = 200
EMOJI_CLOSE_DURATION_MS: Final[int] = 200
EMOJI_ANIM_STEP: Final[int] = 16       # ms per animation frame (~60fps)
BLINK_SELECT_DWELL_MS: Final[int] = 400  # shorter dwell for blink-to-select

# ── Emoji data ───────────────────────────────────────────────────────────────

# Each category: (category_name, list_of_emoji_tuples)
# emoji_tuple: (display_emoji, unicode_char)
EMOJI_CATEGORIES: Final[list[tuple[str, list[tuple[str, str]]]]] = [
    (
        "Caras",
        [
            ("😀", "😀"), ("😂", "😂"), ("🥰", "🥰"), ("😍", "😍"),
            ("🤩", "🤩"), ("😎", "😎"), ("🤔", "🤔"), ("😢", "😢"),
            ("😡", "😡"), ("🤯", "🤯"), ("😱", "😱"), ("🥺", "🥺"),
            ("😴", "😴"), ("🤮", "🤮"), ("👻", "👻"), ("💀", "💀"),
            ("🤖", "🤖"), ("👽", "👽"), ("😈", "😈"), ("🫠", "🫠"),
        ],
    ),
    (
        "Gestos",
        [
            ("👍", "👍"), ("👎", "👎"), ("👏", "👏"), ("🙌", "🙌"),
            ("🤝", "🤝"), ("✌️", "✌️"), ("🤞", "🤞"), ("🤟", "🤟"),
            ("🤘", "🤘"), ("👌", "👌"), ("🫰", "🫰"), ("🫶", "🫶"),
            ("💪", "💪"), ("🙏", "🙏"), ("✋", "✋"), ("👋", "👋"),
            ("🤚", "🤚"), ("🖐️", "🖐️"), ("👆", "👆"), ("👇", "👇"),
        ],
    ),
    (
        "Corazones",
        [
            ("❤️", "❤️"), ("🧡", "🧡"), ("💛", "💛"), ("💚", "💚"),
            ("💙", "💙"), ("💜", "💜"), ("🖤", "🖤"), ("🤍", "🤍"),
            ("💔", "💔"), ("❣️", "❣️"), ("💕", "💕"), ("💞", "💞"),
            ("💓", "💓"), ("💗", "💗"), ("💖", "💖"), ("💘", "💘"),
            ("💝", "💝"), ("💟", "💟"), ("♥️", "♥️"), ("🫶", "🫶"),
        ],
    ),
    (
        "Flechas",
        [
            ("⬆️", "⬆️"), ("⬇️", "⬇️"), ("➡️", "➡️"), ("⬅️", "⬅️"),
            ("↗️", "↗️"), ("↘️", "↘️"), ("↙️", "↙️"), ("↖️", "↖️"),
            ("↔️", "↔️"), ("↕️", "↕️"), ("⇆", "⇆"), ("⇄", "⇄"),
            ("⇄", "⇄"), ("▶️", "▶️"), ("⏩", "⏩"), ("⏪", "⏪"),
            ("⏭️", "⏭️"), ("⏮️", "⏮️"), ("🔼", "🔼"), ("🔽", "🔽"),
        ],
    ),
    (
        "Símbolos",
        [
            ("✅", "✅"), ("❌", "❌"), ("⭕", "⭕"), ("❗", "❗"),
            ("❓", "❓"), ("‼️", "‼️"), ("⁉️", "⁉️"), ("🔴", "🔴"),
            ("🟢", "🟢"), ("🔵", "🔵"), ("🟡", "🟡"), ("🟠", "🟠"),
            ("⚫", "⚫"), ("⚪", "⚪"), ("🔶", "🔶"), ("🔷", "🔷"),
            ("⭐", "⭐"), ("🌟", "🌟"), ("💫", "💫"), ("✨", "✨"),
        ],
    ),
    (
        "Objetos",
        [
            ("🔥", "🔥"), ("💧", "💧"), ("☀️", "☀️"), ("🌙", "🌙"),
            ("⚡", "⚡"), ("🌈", "🌈"), ("🎉", "🎉"), ("🎊", "🎊"),
            ("🎈", "🎈"), ("🎁", "🎁"), ("🏆", "🏆"), ("🥇", "🥇"),
            ("💻", "💻"), ("📱", "📱"), ("🖥️", "🖥️"), ("⌨️", "⌨️"),
            ("🖱️", "🖱️"), ("📷", "📷"), ("🎵", "🎵"), ("🎶", "🎶"),
        ],
    ),
]


# ── Enums & dataclasses ──────────────────────────────────────────────────────

class EmojiOverlayState(Enum):
    HIDDEN = auto()
    OPENING = auto()
    VISIBLE = auto()
    CLOSING = auto()


@dataclass(frozen=True)
class EmojiTile:
    """A single emoji tile on the overlay."""
    emoji: str           # display character
    char: str            # unicode char to type
    x: int               # center x in overlay coords
    y: int               # center y in overlay coords
    row: int             # category row index
    col: int             # column index within row
    category: str        # category name

    @property
    def rect(self) -> QtCore.QRect:
        half = EMOJI_TILE_SIZE // 2
        return QtCore.QRect(
            self.x - half, self.y - half, EMOJI_TILE_SIZE, EMOJI_TILE_SIZE,
        )

    def contains_point(self, px: int, py: int) -> bool:
        return self.rect.contains(QtCore.QPoint(px, py))


@dataclass
class OverlayState:
    """Tracks the current state of the emoji overlay."""
    state: EmojiOverlayState = EmojiOverlayState.HIDDEN
    open_progress: float = 0.0
    centre_x: int = 0
    centre_y: int = 0
    selected_tile: EmojiTile | None = None
    dwell_progress: float = 0.0
    tiles: list[EmojiTile] = field(default_factory=list)
    active_category: int = 0   # which category row is focused
    blink_select_mode: bool = False
    blink_timestamp: float = 0.0
    blink_count: int = 0


# ── Emoji overlay widget ─────────────────────────────────────────────────────

class EmojiOverlayWidget(QtWidgets.QWidget):
    """Translucent emoji picker drawn over the desktop.

    The overlay appears centred on screen.  Categories are shown as rows
    of emoji tiles.  The user navigates with gaze (the cursor highlight
    follows the tile under gaze).  Dwell confirms the emoji.  Blink-to-
    select is also available as an alternative to dwell.
    """

    emoji_selected = QtCore.pyqtSignal(str)  # unicode emoji character
    overlay_closed = QtCore.pyqtSignal()

    def __init__(self, dwell_ms: int = EMOJI_OVERLAY_DWELL_MS) -> None:
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
        self._state = OverlayState()
        self._screen_width = 1920
        self._screen_height = 1080

        # Animation timer
        self._anim_timer = QtCore.QTimer(self)
        self._anim_timer.setSingleShot(True)
        self._anim_timer.timeout.connect(self._advance_animation)

        # Dwell timer for emoji selection
        self._dwell_timer = QtCore.QTimer(self)
        self._dwell_timer.setSingleShot(True)
        self._dwell_timer.timeout.connect(self._confirm_dwell)

        # Build emoji tiles
        self._build_tiles()

    # ── public API ───────────────────────────────────────────────────────

    def _build_tiles(self) -> None:
        """Build the emoji tile grid from EMOJI_CATEGORIES."""
        self._tiles: list[EmojiTile] = []
        padding_x = 40
        padding_y = 80  # extra space for category labels
        row_height = EMOJI_TILE_SIZE + EMOJI_TILE_GAP

        for cat_idx, (cat_name, emojis) in enumerate(EMOJI_CATEGORIES):
            row_width = len(emojis) * EMOJI_TILE_SIZE + (len(emojis) - 1) * EMOJI_TILE_GAP
            x_start = (EMOJI_OVERLAY_WIDTH - row_width) // 2
            y = padding_y + cat_idx * (EMOJI_TILE_SIZE + EMOJI_TILE_GAP + 20)

            for col_idx, (display, char) in enumerate(emojis):
                x = x_start + col_idx * (EMOJI_TILE_SIZE + EMOJI_TILE_GAP) + EMOJI_TILE_SIZE // 2
                tile = EmojiTile(
                    emoji=display,
                    char=char,
                    x=x,
                    y=y,
                    row=cat_idx,
                    col=col_idx,
                    category=cat_name,
                )
                self._tiles.append(tile)

    def open_at(self, x: int, y: int) -> None:
        """Start opening the overlay centred at the given position."""
        centre_x = min(
            max(x, EMOJI_OVERLAY_WIDTH // 2 + 20),
            max(1920, x * 2) - EMOJI_OVERLAY_WIDTH // 2 - 20,
        )
        centre_y = min(
            max(y, EMOJI_OVERLAY_HEIGHT // 2 + 20),
            1080 - EMOJI_OVERLAY_HEIGHT // 2 - 20,
        )

        self._state = OverlayState(
            state=EmojiOverlayState.OPENING,
            open_progress=0.0,
            centre_x=centre_x,
            centre_y=centre_y,
            tiles=list(self._tiles),
            selected_tile=None,
            dwell_progress=0.0,
        )
        self._screen_width = max(1920, x * 2)
        self._screen_height = 1080
        self._start_animation(EMOJI_OPEN_DURATION_MS)
        self.show()
        self.update()

    def close(self) -> None:
        """Start closing the overlay."""
        if self._state.state == EmojiOverlayState.HIDDEN:
            return
        self._state.state = EmojiOverlayState.CLOSING
        self._dwell_timer.stop()
        self._start_animation(EMOJI_CLOSE_DURATION_MS)

    @property
    def visible(self) -> bool:
        return self._state.state in (
            EmojiOverlayState.VISIBLE,
            EmojiOverlayState.OPENING,
            EmojiOverlayState.CLOSING,
        )

    def set_blink_select_mode(self, enabled: bool) -> None:
        """Enable blink-to-select mode (faster than dwell)."""
        self._state.blink_select_mode = enabled
        if enabled:
            self._state.dwell_progress = 0.0
            self._dwell_timer.stop()

    # ── blink-to-select ──────────────────────────────────────────────────

    def process_blink(self, blink: bool) -> None:
        """Process a blink event for blink-to-select mode."""
        if not self._state.blink_select_mode or not self.visible:
            return

        if blink:
            now = QtCore.QDateTime.currentDateTime().toMSecsSinceEpoch() / 1000.0
            time_since_last = now - self._state.blink_timestamp

            if time_since_last < 0.5:  # second blink within 500ms
                if self._state.selected_tile is not None:
                    self._confirm_dwell()
            else:
                self._state.blink_count += 1
                self._state.blink_timestamp = now
                if self._state.selected_tile is not None:
                    dwell_steps = BLINK_SELECT_DWELL_MS / EMOJI_ANIM_STEP
                    self._state.dwell_progress = min(
                        1.0,
                        self._state.dwell_progress + 1.0 / dwell_steps,
                    )
                    if self._state.dwell_progress >= 1.0:
                        self._confirm_dwell()
                    else:
                        self._dwell_timer.start(EMOJI_ANIM_STEP)
        else:
            self._state.blink_count = 0

    # ── animation ────────────────────────────────────────────────────────

    def _start_animation(self, duration_ms: int) -> None:
        self._anim_timer.start(duration_ms)

    def _advance_animation(self) -> None:
        if self._state.state == EmojiOverlayState.OPENING:
            self._state.state = EmojiOverlayState.VISIBLE
            self._state.open_progress = 1.0
        elif self._state.state == EmojiOverlayState.CLOSING:
            self._state.open_progress = 0.0
            self._state.state = EmojiOverlayState.HIDDEN
            self.hide()
            self.overlay_closed.emit()
        self.update()

    # ── dwell update ─────────────────────────────────────────────────────

    def update_dwell(self, cursor_xy: tuple[int, int] | None) -> None:
        """Call each frame with the current cursor position."""
        if not self.visible or self._state.state == EmojiOverlayState.CLOSING:
            return

        cx, cy = self._state.centre_x, self._state.centre_y

        if cursor_xy is None:
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_tile = None
            self.update()
            return

        selected = self._hit_test(cursor_xy, cx, cy)

        if selected is not None:
            self._state.selected_tile = selected
            dwell_steps = self._dwell_ms / EMOJI_ANIM_STEP
            self._state.dwell_progress = min(
                1.0,
                self._state.dwell_progress + 1.0 / dwell_steps,
            )
            if self._state.dwell_progress >= 1.0:
                self._confirm_dwell()
            else:
                self._dwell_timer.start(EMOJI_ANIM_STEP)
        else:
            self._dwell_timer.stop()
            self._state.dwell_progress = 0.0
            self._state.selected_tile = None
        self.update()

    def _hit_test(
        self,
        cursor_xy: tuple[int, int],
        cx: int,
        cy: int,
    ) -> EmojiTile | None:
        """Determine which tile the cursor is in, or None."""
        for tile in self._tiles:
            screen_x = cx - EMOJI_OVERLAY_WIDTH // 2 + tile.x
            screen_y = cy - EMOJI_OVERLAY_HEIGHT // 2 + tile.y
            half = EMOJI_TILE_SIZE // 2
            tile_rect = QtCore.QRect(
                screen_x - half, screen_y - half, EMOJI_TILE_SIZE, EMOJI_TILE_SIZE,
            )
            if tile_rect.contains(QtCore.QPoint(cursor_xy[0], cursor_xy[1])):
                return tile
        return None

    def _confirm_dwell(self) -> None:
        """Dwell completed — emit the selected emoji."""
        tile = self._state.selected_tile
        if tile is None:
            return

        self.emoji_selected.emit(tile.char)
        self._dwell_timer.stop()
        self._state.dwell_progress = 0.0
        self._state.selected_tile = None
        self.update()

    # ── paint ────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        # Background — translucent glass
        alpha = int(255 * self._state.open_progress * 0.85)
        bg = QtGui.QColor(255, 255, 255, alpha)
        p.setBrush(bg)
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(
            QtCore.QRect(
                self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2,
                self._state.centre_y - EMOJI_OVERLAY_HEIGHT // 2,
                EMOJI_OVERLAY_WIDTH,
                EMOJI_OVERLAY_HEIGHT,
            ),
            20, 20,
        )

        # Glass border
        border_pen = QtGui.QPen(QtGui.QColor(PALETTE.blue_soft), 2)
        border_pen.setCosmetic(True)
        p.setPen(border_pen)
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(
            QtCore.QRect(
                self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2,
                self._state.centre_y - EMOJI_OVERLAY_HEIGHT // 2,
                EMOJI_OVERLAY_WIDTH,
                EMOJI_OVERLAY_HEIGHT,
            ),
            20, 20,
        )

        # Title
        title_font = p.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QtGui.QColor(PALETTE.blue))
        title_rect = QtCore.QRect(
            self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2,
            self._state.centre_y - EMOJI_OVERLAY_HEIGHT // 2 + 8,
            EMOJI_OVERLAY_WIDTH, 30,
        )
        p.drawText(title_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "😊 Emojis")

        # Draw category labels and emoji tiles
        padding_y = 80
        for cat_idx, (cat_name, emojis) in enumerate(EMOJI_CATEGORIES):
            row_width = len(emojis) * EMOJI_TILE_SIZE + (len(emojis) - 1) * EMOJI_TILE_GAP
            x_start = (EMOJI_OVERLAY_WIDTH - row_width) // 2
            y = padding_y + cat_idx * (EMOJI_TILE_SIZE + EMOJI_TILE_GAP + 20)

            # Category label
            cat_font = p.font()
            cat_font.setPointSize(10)
            cat_font.setBold(True)
            p.setFont(cat_font)
            p.setPen(QtGui.QColor(PALETTE.text_secondary))
            label_rect = QtCore.QRect(
                self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2 + 10,
                y - EMOJI_TILE_SIZE // 2 - 14,
                120, 18,
            )
            p.drawText(label_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, cat_name)

            # Draw each emoji tile
            for col_idx, (display, char) in enumerate(emojis):
                tx = x_start + col_idx * (EMOJI_TILE_SIZE + EMOJI_TILE_GAP) + EMOJI_TILE_SIZE // 2
                ty = y

                # Find if this is the selected tile
                screen_x = self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2 + tx
                screen_y = self._state.centre_y - EMOJI_OVERLAY_HEIGHT // 2 + ty
                tile_rect = QtCore.QRect(
                    screen_x - EMOJI_TILE_SIZE // 2,
                    screen_y - EMOJI_TILE_SIZE // 2,
                    EMOJI_TILE_SIZE, EMOJI_TILE_SIZE,
                )

                # Check if selected
                is_selected = (
                    self._state.selected_tile is not None
                    and self._state.selected_tile.row == cat_idx
                    and self._state.selected_tile.col == col_idx
                )

                if is_selected:
                    # Highlight ring
                    ring_pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 3)
                    ring_pen.setCosmetic(True)
                    p.setPen(ring_pen)
                    p.setBrush(QtGui.QColor(PALETTE.orange_soft))
                    p.drawRoundedRect(tile_rect.adjusted(0, 0, 0, 0), 8, 8)

                    # Dwell arc
                    if self._state.dwell_progress > 0:
                        arc_pen = QtGui.QPen(QtGui.QColor(PALETTE.orange), 4)
                        arc_pen.setCosmetic(True)
                        p.setPen(arc_pen)
                        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                        arc_rect = QtCore.QRect(
                            screen_x - 22, screen_y - 22, 44, 44,
                        )
                        p.drawArc(
                            arc_rect,
                            90 * 16,
                            int(-360 * 16 * self._state.dwell_progress),
                        )

                # Draw emoji
                emoji_font = p.font()
                emoji_font.setPointSize(22)
                p.setFont(emoji_font)
                p.setPen(QtGui.QColor(PALETTE.text_primary))
                p.drawText(
                    tile_rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    display,
                )

        # Close hint at bottom
        hint_font = p.font()
        hint_font.setPointSize(9)
        p.setFont(hint_font)
        p.setPen(QtGui.QColor(PALETTE.text_muted))
        hint_rect = QtCore.QRect(
            self._state.centre_x - EMOJI_OVERLAY_WIDTH // 2 + 10,
            self._state.centre_y + EMOJI_OVERLAY_HEIGHT // 2 - 24,
            EMOJI_OVERLAY_WIDTH - 20, 18,
        )
        p.drawText(
            hint_rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            "Mirar un emoji para escribir · Esc para cerrar",
        )
