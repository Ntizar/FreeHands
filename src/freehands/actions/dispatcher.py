"""Translates abstract action names into OS-level events via pyautogui."""
from __future__ import annotations

import pyautogui

pyautogui.FAILSAFE = True   # move mouse to a screen corner to abort
pyautogui.PAUSE = 0

ACTIONS = {"click", "right_click", "double_click", "escape",
           "zoom_in", "zoom_out", "scroll_up", "scroll_down",
           "undo", "toggle_pause", "resume"}


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
            if click_xy is not None:
                pyautogui.click(*click_xy, clicks=2, interval=0.10)
            else:
                pyautogui.click(clicks=2, interval=0.10)
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
        # toggle_pause / resume are handled by the UI overlay, not pyautogui.
        return True
