"""Dead-zone clamping for screen-edge cursor coordinates.

Prevents the gaze cursor from reaching the extreme edges of the screen,
where accidental detections (head tilt, camera noise) could move the
pointer off-screen or into OS hotspots (Windows taskbar, macOS menu bar).

The dead zone is expressed as a percentage of the screen dimension.
A value of 0.05 means the cursor will never go below 5 % or above 95 %
of the screen width / height.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeadZoneConfig:
    """Configuration for screen-edge dead zones.

    Parameters
    ----------
    edge_margin_pct : float
        Percentage of screen dimension reserved as dead zone on each side.
        ``0.05`` → 5 % on left + 5 % on right = 10 % total dead zone.
        Must be in ``(0, 0.5)``.
    """

    edge_margin_pct: float = 0.05  # 5 % on each side

    def __post_init__(self) -> None:
        if not (0.0 < self.edge_margin_pct < 0.5):
            raise ValueError(
                f"edge_margin_pct must be in (0, 0.5), got {self.edge_margin_pct}"
            )

    @property
    def margin_px(self) -> int:
        """Default margin in pixels (used when screen size is unknown)."""
        return 40  # at least 40 px even on tiny screens


class DeadZoneClamper:
    """Clamps cursor coordinates to stay within dead-zone margins.

    Usage
    -----
    >>> clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    >>> clamper.clamp((0, 0))
    (96, 54)
    >>> clamper.clamp((1920, 1080))
    (1824, 1026)
    """

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        config: DeadZoneConfig | None = None,
    ) -> None:
        self._config = config or DeadZoneConfig()
        self._screen_w = max(screen_width, 1)
        self._screen_h = max(screen_height, 1)
        self._min_x: int = max(0, int(self._screen_w * self._config.edge_margin_pct))
        self._max_x: int = min(
            self._screen_w - 1,
            int(self._screen_w * (1.0 - self._config.edge_margin_pct)),
        )
        self._min_y: int = max(0, int(self._screen_h * self._config.edge_margin_pct))
        self._max_y: int = min(
            self._screen_h - 1,
            int(self._screen_h * (1.0 - self._config.edge_margin_pct)),
        )
        # Ensure minimum margin in pixels
        min_margin = self._config.margin_px
        self._min_x = max(self._min_x, min_margin)
        self._max_x = min(self._max_x, self._screen_w - 1 - min_margin)
        self._min_y = max(self._min_y, min_margin)
        self._max_y = min(self._max_y, self._screen_h - 1 - min_margin)

    def update_screen(self, width: int, height: int) -> None:
        """Recompute bounds when the screen geometry changes (multi-monitor)."""
        self._screen_w = max(width, 1)
        self._screen_h = max(height, 1)
        self._min_x = max(0, int(self._screen_w * self._config.edge_margin_pct))
        self._max_x = min(
            self._screen_w - 1,
            int(self._screen_w * (1.0 - self._config.edge_margin_pct)),
        )
        self._min_y = max(0, int(self._screen_h * self._config.edge_margin_pct))
        self._max_y = min(
            self._screen_h - 1,
            int(self._screen_h * (1.0 - self._config.edge_margin_pct)),
        )
        min_margin = self._config.margin_px
        self._min_x = max(self._min_x, min_margin)
        self._max_x = min(self._max_x, self._screen_w - 1 - min_margin)
        self._min_y = max(self._min_y, min_margin)
        self._max_y = min(self._max_y, self._screen_h - 1 - min_margin)

    def clamp(self, xy: tuple[int, int]) -> tuple[int, int]:
        """Return *xy* clamped to the valid region."""
        x = max(self._min_x, min(xy[0], self._max_x))
        y = max(self._min_y, min(xy[1], self._max_y))
        return (x, y)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return ``(min_x, min_y, max_x, max_y)`` for debugging."""
        return (self._min_x, self._min_y, self._max_x, self._max_y)
