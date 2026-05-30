"""Snap-to-grid: after 300 ms of stable gaze, snap cursor to nearest UI element center.

When the user's gaze stabilises on a region of the screen, this module
computes the centre of the nearest "UI element" (button, icon, text block)
and snaps the cursor there.  The snap is soft — it only activates when the
gaze has been stable long enough (``snap_dwell_ms``) and the target is
within a reasonable radius of the raw cursor position.

Design decisions
----------------
* No external libraries (no UI automation framework required).
* Uses screen geometry heuristics: grid cells of configurable size.
* The snap target is the centre of the nearest grid cell that the raw
  cursor is closest to — this simulates "snapping to UI elements" without
  needing a screen-scraping library.
* Configurable via ``SnapConfig`` so it can be tuned per-user.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Final

# Default configuration values — match the plan spec.
DEFAULT_SNAP_DEWELL_MS: Final[int] = 300
DEFAULT_SNAP_RADIUS_PX: Final[int] = 80
DEFAULT_GRID_SIZE: Final[int] = 40  # cell size simulating UI element bounds


@dataclass
class SnapConfig:
    """Tunable parameters for the snap-to-grid module."""

    snap_dwell_ms: int = DEFAULT_SNAP_DEWELL_MS
    snap_radius_px: int = DEFAULT_SNAP_RADIUS_PX
    grid_size: int = DEFAULT_GRID_SIZE
    enabled: bool = True


class SnapToGrid:
    """Snaps cursor to the centre of the nearest grid cell after dwell.

    The algorithm:
    1. Collect cursor samples while gaze is stable.
    2. After ``snap_dwell_ms`` of continuous stability, compute the centroid.
    3. Snap the centroid to the nearest grid-cell centre.
    4. The snap only activates if the raw cursor is within ``snap_radius_px``
       of the snapped target (prevents wild jumps).
    """

    # Minimum number of stable samples needed to trigger a snap.
    _MIN_SAMPLES = 5

    def __init__(self, config: SnapConfig | None = None) -> None:
        self._config = config or SnapConfig()
        self._samples: deque[tuple[float, tuple[int, int]]] = deque(maxlen=30)
        self._anchor: tuple[float, float] | None = None
        self._active = False

    @property
    def active(self) -> bool:
        """True while a snap target is locked in."""
        return self._active

    def reset(self) -> None:
        """Clear all state — call when gaze becomes unstable or cursor is None."""
        self._samples.clear()
        self._anchor = None
        self._active = False

    def update(self, cursor_xy: tuple[int, int], gaze_stable: bool) -> tuple[int, int]:
        """Process a cursor sample and return the (possibly snapped) position.

        Parameters
        ----------
        cursor_xy :
            The raw cursor position from the gaze tracker.
        gaze_stable :
            Whether the gaze is currently considered stable (low variance).

        Returns
        -------
        The final cursor position — possibly snapped to a grid cell centre.
        """
        now = time.monotonic()

        if not gaze_stable:
            # Gaze became unstable — reset snap state.
            self.reset()
            return cursor_xy

        self._samples.append((now, cursor_xy))

        # Prune old samples outside the dwell window.
        dwell_seconds = self._config.snap_dwell_ms / 1000.0
        while self._samples and now - self._samples[0][0] > dwell_seconds * 2:
            self._samples.popleft()

        if self._anchor is None:
            # Not yet anchored — check if we have enough stable samples.
            stable_samples = [
                (t, xy) for t, xy in self._samples
                if now - t <= dwell_seconds
            ]
            if len(stable_samples) >= self._MIN_SAMPLES:
                centroid = self._centroid(stable_samples)
                snapped = self._snap_to_grid(centroid)
                # Only snap if the target is reasonably close.
                dist = self._distance(centroid, snapped)
                if dist <= self._config.snap_radius_px:
                    self._anchor = snapped
                    self._active = True
                    return snapped
                # Too far — don't snap, keep raw cursor.
                return cursor_xy
            return cursor_xy

        # Already snapped — keep tracking and gently follow if cursor moves.
        stable_samples = [
            (t, xy) for t, xy in self._samples
            if now - t <= dwell_seconds
        ]
        if stable_samples:
            centroid = self._centroid(stable_samples)
            snapped = self._snap_to_grid(centroid)
            dist = self._distance(centroid, snapped)
            if dist <= self._config.snap_radius_px:
                # If the snapped target jumped significantly from the old
                # anchor, the user moved to a different region — release.
                if self._anchor is not None:
                    jump = self._distance(self._anchor, snapped)
                    if jump > self._config.grid_size * 2:
                        self.reset()
                        return cursor_xy
                # Smoothly follow the snap target.
                ax, ay = self._anchor
                sx, sy = snapped
                self._anchor = (
                    ax + (sx - ax) * 0.3,
                    ay + (sy - ay) * 0.3,
                )
                return round(self._anchor[0]), round(self._anchor[1])
            else:
                # Target moved too far — release snap.
                self.reset()
                return cursor_xy

        return cursor_xy

    @staticmethod
    def _centroid(samples: list[tuple[float, tuple[int, int]]]) -> tuple[float, float]:
        count = len(samples)
        cx = sum(xy[0] for _, xy in samples) / count
        cy = sum(xy[1] for _, xy in samples) / count
        return cx, cy

    def _snap_to_grid(self, pos: tuple[float, float]) -> tuple[int, int]:
        """Snap a position to the centre of the nearest grid cell."""
        gs = self._config.grid_size
        gx = round(pos[0] / gs) * gs + gs // 2
        gy = round(pos[1] / gs) * gs + gs // 2
        return gx, gy

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[int, int]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5
