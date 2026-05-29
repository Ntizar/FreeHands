"""Tests for the dead-zone screen-edge clamping system."""
from __future__ import annotations

import pytest

from freehands.gaze.dead_zones import DeadZoneClamper, DeadZoneConfig


# ── Basic clamping ──────────────────────────────────────────────────────────


def test_clamp_moves_edge_coordinates_inward():
    """A cursor at (0, 0) should be clamped to the dead-zone boundary."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    result = clamper.clamp((0, 0))
    assert result[0] > 0
    assert result[1] > 0


def test_clamp_moves_max_coordinates_inward():
    """A cursor at max screen coords should be clamped inward."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    result = clamper.clamp((1920, 1080))
    assert result[0] < 1920
    assert result[1] < 1080


def test_clamp_leaves_center_coordinates_unchanged():
    """A cursor in the center of the screen should not be modified."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    result = clamper.clamp((960, 540))
    assert result == (960, 540)


def test_clamp_preserves_valid_coordinates():
    """Coordinates within the valid region pass through unchanged."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    # 10 % from each edge — well inside the 5 % dead zone
    result = clamper.clamp((250, 150))
    assert result == (250, 150)


# ── Dead zone configuration ─────────────────────────────────────────────────


def test_default_dead_zone_is_5_percent():
    """Default config uses 5 % edge margin on each side."""
    clamper = DeadZoneClamper(screen_width=1000, screen_height=1000)
    min_x, min_y, max_x, max_y = clamper.bounds
    # 5 % of 1000 = 50 px on each side
    assert min_x == 50
    assert min_y == 50
    assert max_x == 949
    assert max_y == 949


def test_custom_dead_zone_percentage():
    """A larger edge margin should produce wider dead zones."""
    config = DeadZoneConfig(edge_margin_pct=0.10)  # 10 %
    clamper = DeadZoneClamper(screen_width=1000, screen_height=1000, config=config)
    min_x, min_y, max_x, max_y = clamper.bounds
    assert min_x == 100
    assert min_y == 100
    assert max_x == 899
    assert max_y == 899


def test_invalid_dead_zone_percentage_raises():
    """edge_margin_pct outside (0, 0.5) should raise ValueError."""
    with pytest.raises(ValueError):
        DeadZoneConfig(edge_margin_pct=0.0)
    with pytest.raises(ValueError):
        DeadZoneConfig(edge_margin_pct=0.5)
    with pytest.raises(ValueError):
        DeadZoneConfig(edge_margin_pct=-0.1)
    with pytest.raises(ValueError):
        DeadZoneConfig(edge_margin_pct=0.9)


# ── Screen resize ───────────────────────────────────────────────────────────


def test_update_screen_recomputes_bounds():
    """Changing screen size should update the clamping bounds."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    old_bounds = clamper.bounds
    clamper.update_screen(3840, 2160)
    new_bounds = clamper.bounds
    assert new_bounds[0] > old_bounds[0]  # min_x increased
    assert new_bounds[1] > old_bounds[1]  # min_y increased
    assert new_bounds[2] > old_bounds[2]  # max_x increased
    assert new_bounds[3] > old_bounds[3]  # max_y increased


def test_update_screen_clamps_correctly_after_resize():
    """After resize, a cursor at the old edge should be clamped."""
    clamper = DeadZoneClamper(screen_width=1920, screen_height=1080)
    clamper.update_screen(3840, 2160)
    # Old edge (1920, 1080) is now well inside the new screen
    result = clamper.clamp((1920, 1080))
    assert result == (1920, 1080)  # not clamped — inside new bounds


# ── Minimum margin ──────────────────────────────────────────────────────────


def test_minimum_margin_applies_on_small_screens():
    """On very small screens, the minimum 40 px margin should apply."""
    clamper = DeadZoneClamper(screen_width=200, screen_height=200)
    min_x, min_y, max_x, max_y = clamper.bounds
    assert min_x >= 40
    assert min_y >= 40
    assert max_x <= 159  # 200 - 1 - 40
    assert max_y <= 159


# ── Integration with main.py flow ───────────────────────────────────────────


def test_dead_zone_clamp_before_fine_aim():
    """Verify dead zone is applied before fine aim in the main loop."""
    from freehands.main import run_system
    import inspect

    source = inspect.getsource(run_system)
    # dead_zone.clamp should appear before fine_aim.update
    clamp_pos = source.find("dead_zone.clamp")
    fine_aim_pos = source.find("fine_aim.update")
    assert clamp_pos >= 0, "dead_zone.clamp not found in run_system"
    assert fine_aim_pos >= 0, "fine_aim.update not found in run_system"
    assert clamp_pos < fine_aim_pos, "dead_zone.clamp should be called before fine_aim.update"


def test_dead_zone_imported_in_main():
    """DeadZoneClamper should be imported in main.py."""
    from freehands import main
    assert hasattr(main, "DeadZoneClamper")
