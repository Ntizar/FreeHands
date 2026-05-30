"""Example plugin: cursor trail renderer.

Draws a fading trail behind the cursor to help users track
cursor movement. Demonstrates the on_gaze and on_overlay hooks.

Usage:
    from freehands.plugins.example_cursor_trail import CursorTrailPlugin

    loader = PluginLoader()
    loader.register(CursorTrailPlugin(trail_length=15))
    loader.load()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from ..plugins.base import FreeHandsPlugin, PluginContext


class CursorTrailPlugin(FreeHandsPlugin):
    """Renders a fading trail behind the cursor.

    Tracks recent cursor positions and exposes them via the
    plugin context so the overlay can draw the trail.

    Attributes:
        trail_length: Number of positions to keep in the trail.
        trail_speed: How fast the trail fades (seconds for full fade).
    """

    name = "cursor_trail"
    version = "1.0.0"
    description = "Draws a fading trail behind the cursor"
    enabled_by_default = False
    priority = 50  # Early in pipeline

    def __init__(self, trail_length: int = 15, trail_speed: float = 0.5) -> None:
        super().__init__()
        self.trail_length = trail_length
        self.trail_speed = trail_speed
        self._trail: deque[tuple[float, tuple[int, int]]] = deque(maxlen=trail_length)

    def on_gaze(self, cursor: tuple[int, int] | None, ctx: PluginContext) -> tuple[int, int] | None:
        """Record cursor position in trail."""
        if cursor is not None:
            now = time.monotonic()
            self._trail.append((now, cursor))
            # Clean up old entries beyond trail_speed
            cutoff = now - self.trail_speed
            while self._trail and self._trail[0][0] < cutoff:
                self._trail.popleft()
            # Expose trail via context for overlay rendering
            ctx.metadata["cursor_trail"] = [pos for _, pos in self._trail]
        return cursor

    def on_overlay(self, ctx: PluginContext) -> None:
        """Log trail status for debugging."""
        trail = ctx.metadata.get("cursor_trail", [])
        if trail:
            ctx.metadata["cursor_trail_length"] = len(trail)
