"""Tests for the virtual keyboard widget."""
import os
import pytest

# Skip tests that require a display (Qt widgets) in headless CI
has_display = os.environ.get("DISPLAY") is not None

from freehands.ui.virtual_keyboard import (
    KeyDefinition,
    KeyboardState,
    KEYBOARD_DWELL_MS,
    KEYBOARD_ROWS,
    KEY_WIDTH,
    KEY_HEIGHT,
    KEYBOARD_WIDTH,
    KEYBOARD_HEIGHT,
)


class TestKeyDefinition:
    """Test KeyDefinition dataclass."""

    def test_key_contains_point(self):
        key = KeyDefinition(
            label="A", char="a", x=100, y=100,
            width=KEY_WIDTH, height=KEY_HEIGHT,
        )
        assert key.contains_point(100, 100)
        assert not key.contains_point(0, 0)

    def test_key_rect(self):
        key = KeyDefinition(
            label="A", char="a", x=100, y=100,
            width=KEY_WIDTH, height=KEY_HEIGHT,
        )
        rect = key.rect
        assert rect.left() == 100 - KEY_WIDTH // 2
        assert rect.top() == 100 - KEY_HEIGHT // 2
        assert rect.width() == KEY_WIDTH
        assert rect.height() == KEY_HEIGHT


@pytest.mark.skipif(not has_display, reason="requires display")
class TestVirtualKeyboardWidget:
    """Test VirtualKeyboardWidget initialization (requires display)."""

    def test_widget_creation(self):
        from freehands.ui.virtual_keyboard import VirtualKeyboardWidget
        widget = VirtualKeyboardWidget()
        assert widget is not None
        assert not widget.visible
        assert not widget.shift_active
        assert widget.typing_buffer == ""

    def test_default_dwell_time(self):
        from freehands.ui.virtual_keyboard import VirtualKeyboardWidget
        widget = VirtualKeyboardWidget()
        assert widget._dwell_ms == KEYBOARD_DWELL_MS

    def test_custom_dwell_time(self):
        from freehands.ui.virtual_keyboard import VirtualKeyboardWidget
        widget = VirtualKeyboardWidget(dwell_ms=1200)
        assert widget._dwell_ms == 1200

    def test_keys_built(self):
        from freehands.ui.virtual_keyboard import VirtualKeyboardWidget
        widget = VirtualKeyboardWidget()
        assert len(widget._keys) > 0
        # Should have keys for QWERTY layout
        char_labels = {k.char for k in widget._keys}
        assert "q" in char_labels
        assert "a" in char_labels
        assert "z" in char_labels
        assert "space" in char_labels
        assert "backspace" in char_labels
        assert "enter" in char_labels
        assert "shift" in char_labels

    def test_shift_toggle(self):
        from freehands.ui.virtual_keyboard import VirtualKeyboardWidget
        widget = VirtualKeyboardWidget()
        assert not widget.shift_active
        widget.set_shift(True)
        assert widget.shift_active
        widget.set_shift(False)
        assert not widget.shift_active


class TestKeyboardState:
    """Test KeyboardState dataclass."""

    def test_default_state(self):
        state = KeyboardState()
        assert not state.visible
        assert state.open_progress == 0.0
        assert state.selected_key is None
        assert state.dwell_progress == 0.0
        assert not state.opening
        assert not state.closing

    def test_opening_state(self):
        state = KeyboardState(visible=True, opening=True, open_progress=0.5)
        assert state.visible
        assert state.opening
        assert state.open_progress == 0.5


class TestKeyboardLayout:
    """Test keyboard layout constants."""

    def test_keyboard_dimensions(self):
        assert KEYBOARD_WIDTH == 680
        assert KEYBOARD_HEIGHT == 280

    def test_key_dimensions(self):
        assert KEY_WIDTH == 52
        assert KEY_HEIGHT == 44

    def test_dwell_time(self):
        assert KEYBOARD_DWELL_MS == 800

    def test_rows_count(self):
        assert len(KEYBOARD_ROWS) == 4

    def test_row_lengths(self):
        assert len(KEYBOARD_ROWS[0]) == 10  # QWERTY row
        assert len(KEYBOARD_ROWS[1]) == 10  # ASDF row
        assert len(KEYBOARD_ROWS[2]) == 8   # ZXCV row
        assert len(KEYBOARD_ROWS[3]) == 4   # Shift/Space/Enter/Backspace
