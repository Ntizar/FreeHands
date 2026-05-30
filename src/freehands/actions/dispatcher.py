"""Translates abstract action names into OS-level events via pyautogui."""
from __future__ import annotations

import sys

import pyautogui

pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort
pyautogui.PAUSE = 0

ACTIONS = {"click", "right_click", "double_click", "escape",
           "zoom_in", "zoom_out", "scroll_up", "scroll_down",
           "undo", "toggle_pause", "resume",
           "show_desktop", "screenshot",
           "volume_up", "volume_down", "volume_mute",
           "drag_start", "drag_end"}


class ActionDispatcher:
    """Stateless dispatcher. Returns ``True`` if the action was executed."""

    @staticmethod
    def _safe_xy(at_xy: tuple[int, int]) -> tuple[int, int]:
        x, y = int(at_xy[0]), int(at_xy[1])
        return max(2, x), max(2, y)

    def move_pointer(self, at_xy: tuple[int, int]) -> bool:
        x, y = self._safe_xy(at_xy)
        pyautogui.moveTo(x, y, duration=0)
        return True

    def execute(self, action: str, at_xy: tuple[int, int] | None = None) -> bool:
        if action not in ACTIONS:
            return False

        click_xy: tuple[int, int] | None = None
        if at_xy is not None and action in {"click", "right_click", "double_click"}:
            x, y = self._safe_xy(at_xy)
            click_xy = (x, y)

        if action == "click":
            pyautogui.click(*click_xy) if click_xy is not None else pyautogui.click()
        elif action == "right_click":
            pyautogui.rightClick(*click_xy) if click_xy is not None else pyautogui.rightClick()
        elif action == "double_click":
            # Use Windows-native double click semantics so apps (Explorer,
            # desktop, native dialogs) actually register it as a double click.
            if click_xy is not None:
                pyautogui.doubleClick(*click_xy)
            else:
                pyautogui.doubleClick()
        elif action == "escape":
            pyautogui.press("escape")
        elif action == "zoom_in":
            pyautogui.hotkey("ctrl", "+")
        elif action == "zoom_out":
            pyautogui.hotkey("ctrl", "-")
        elif action == "scroll_up":
            pyautogui.scroll(3)
        elif action == "scroll_down":
            pyautogui.scroll(-3)
        elif action == "undo":
            pyautogui.hotkey("ctrl", "z")
        # ── System commands ────────────────────────────────────────────────
        elif action == "show_desktop":
            # Windows: Show desktop (Win+D). Cross-platform fallback: Escape.
            if sys.platform == "win32":
                pyautogui.hotkey("win", "d")
            else:
                pyautogui.press("escape")
        elif action == "screenshot":
            # Windows: PrintScreen. Cross-platform fallback: no-op with print.
            if sys.platform == "win32":
                pyautogui.hotkey("win", "shift", "s")
            else:
                print("[freehands] screenshot: no screenshot shortcut on this platform")
        elif action == "volume_up":
            pyautogui.press("volumeup")
        elif action == "volume_down":
            pyautogui.press("volumedown")
        elif action == "volume_mute":
            pyautogui.press("volumemute")
        # ── Drag gestures (prolonged blink) ────────────────────────────────
        elif action == "drag_start":
            # Press and hold left mouse button to begin a drag operation.
            pyautogui.mouseDown()
        elif action == "drag_end":
            # Release the left mouse button to end the drag operation.
            pyautogui.mouseUp()
        # toggle_pause / resume are handled by the UI overlay, not pyautogui.
        return True
