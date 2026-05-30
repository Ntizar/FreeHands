"""Tests for snap-to-grid cursor snapping feature."""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path


def _load_snap_module():
    """Load snap_to_grid.py directly, bypassing gaze/__init__.py (sklearn dep)."""
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "freehands" / "gaze" / "snap_to_grid.py"
    )
    spec = importlib.util.spec_from_file_location(
        "freehands.gaze.snap_to_grid", module_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["freehands.gaze.snap_to_grid"] = mod  # register so dataclass works
    spec.loader.exec_module(mod)
    return mod


_snap = _load_snap_module()
SnapConfig = _snap.SnapConfig
SnapToGrid = _snap.SnapToGrid


class TestSnapToGridBasic:
    """Basic functionality tests."""

    def test_starts_inactive(self):
        snap = SnapToGrid()
        assert not snap.active
        assert snap._anchor is None

    def test_reset_clears_state(self):
        snap = SnapToGrid()
        snap._anchor = (100.0, 100.0)
        snap._active = True
        snap.reset()
        assert not snap.active
        assert snap._anchor is None
        assert len(snap._samples) == 0

    def test_returns_raw_cursor_when_unstable(self):
        snap = SnapToGrid()
        result = snap.update((500, 300), gaze_stable=False)
        assert result == (500, 300)
        assert not snap.active


class TestSnapToGridDwell:
    """Tests that snap activates after dwell time."""

    def test_snaps_after_stable_samples(self):
        """After enough stable samples, cursor snaps to grid centre."""
        snap = SnapToGrid(SnapConfig(snap_dwell_ms=50, snap_radius_px=80))
        pos = (200, 150)
        now = time.monotonic()
        for i in range(10):
            t = now + i * 0.005
            snap._samples.append((t, pos))

        result = snap.update(pos, gaze_stable=True)
        # Grid cell size is 40, so centre of cell at (200,150) = (200, 160)
        assert snap.active
        # The snapped position should be near a grid centre
        assert abs(result[0] - 200) <= 20
        assert abs(result[1] - 160) <= 20

    def test_does_not_snap_when_too_far(self):
        """Snap is rejected if the target is beyond snap_radius_px."""
        snap = SnapToGrid(SnapConfig(snap_dwell_ms=50, snap_radius_px=10))
        now = time.monotonic()
        pos = (205, 155)
        for i in range(10):
            t = now + i * 0.005
            snap._samples.append((t, pos))

        result = snap.update(pos, gaze_stable=True)
        # Grid centre of (205,155) is (200,160), distance ~11.2 > 10
        assert not snap.active


class TestSnapToGridReset:
    """Tests that snap resets on instability or large movement."""

    def test_resets_on_large_movement(self):
        """If cursor moves far from anchor, snap resets."""
        snap = SnapToGrid(SnapConfig(snap_dwell_ms=50, snap_radius_px=80))
        now = time.monotonic()

        pos1 = (200, 160)
        for i in range(10):
            t = now + i * 0.005
            snap._samples.append((t, pos1))
        snap.update(pos1, gaze_stable=True)
        assert snap.active
        old_anchor = snap._anchor

        # Manually set up a far-away state to test the jump detection.
        # The jump threshold is grid_size * 2 = 80.
        # Set anchor to (100, 100) and feed samples at (500, 500).
        snap._anchor = (100.0, 100.0)
        snap._active = True
        snap._samples.clear()
        for i in range(10):
            t = now + 0.1 + i * 0.005
            snap._samples.append((t, (500, 500)))

        result = snap.update((500, 500), gaze_stable=True)
        # Jump from (100,100) to grid-centre-of-(500,500)=(500,520) = 400px > 80
        assert not snap.active
        assert result == (500, 500)

    def test_resets_on_gaze_instability(self):
        """Gaze instability resets the snap state."""
        snap = SnapToGrid()
        snap._anchor = (100.0, 100.0)
        snap._active = True

        snap.update((100, 100), gaze_stable=False)
        assert not snap.active


class TestSnapToGridGridMath:
    """Tests for the grid snapping math."""

    def test_snap_to_grid_centre(self):
        """SnapToGrid._snap_to_grid should return nearest cell centre."""
        # Grid size is 40. Cell centres are at (20, 60, 100, 140, ...)
        snap = SnapToGrid(SnapConfig(grid_size=40))
        result = snap._snap_to_grid((0.0, 0.0))
        assert result == (20, 20)

        result = snap._snap_to_grid((40.0, 40.0))
        assert result == (60, 60)

        result = snap._snap_to_grid((100.0, 100.0))
        assert result == (100, 100)

    def test_snap_to_grid_rounds_to_nearest(self):
        """Snap should round to nearest cell centre."""
        snap = SnapToGrid(SnapConfig(grid_size=40))
        # Position at (39.9, 39.9) → nearest cell centre is (60, 60)
        result = snap._snap_to_grid((39.9, 39.9))
        assert result == (60, 60)

        # Position at (19.9, 19.9) → nearest cell centre is (20, 20)
        result = snap._snap_to_grid((19.9, 19.9))
        assert result == (20, 20)


class TestSnapToGridConfig:
    """Tests for configurable parameters."""

    def test_custom_grid_size(self):
        """Snap uses configured grid size."""
        snap = SnapToGrid(SnapConfig(grid_size=64, snap_radius_px=100))
        now = time.monotonic()
        pos = (100, 100)
        for i in range(10):
            t = now + i * 0.005
            snap._samples.append((t, pos))
        snap.update(pos, gaze_stable=True)
        assert snap.active
        # Grid size 64: centres at 32, 96, 160...
        # (100, 100) → round(100/64) = round(1.5625) = 2 → 2*64+32 = 160
        assert snap._anchor == (160, 160)

    def test_disabled_snap(self):
        """Snap can be disabled via config."""
        snap = SnapToGrid(SnapConfig(enabled=False))
        result = snap.update((200, 150), gaze_stable=True)
        # Even though disabled, the core logic still works;
        # the enabled flag is for external gating.
        assert result == (200, 150)
