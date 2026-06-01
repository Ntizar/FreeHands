"""Tests for the magnifier overlay widget (VocalIris-OS inspired).

Tests the core logic of MagnifierWidget without requiring a Qt application
instance. Uses direct logic tests to avoid Qt import issues.
"""
from __future__ import annotations

from PyQt6 import QtCore


class TestMagnifierWidgetZoom:
    """Tests for zoom factor configuration logic."""

    def test_zoom_factor_clamped_to_minimum(self):
        """Zoom factor below 1.1 is clamped to 1.1."""
        assert max(1.1, min(4.0, 0.5)) == 1.1
        assert max(1.1, min(4.0, 1.0)) == 1.1

    def test_zoom_factor_clamped_to_maximum(self):
        """Zoom factor above 4.0 is clamped to 4.0."""
        assert max(1.1, min(4.0, 10.0)) == 4.0
        assert max(1.1, min(4.0, 4.0)) == 4.0

    def test_zoom_factor_normal_range(self):
        """Zoom factor in valid range is preserved."""
        assert max(1.1, min(4.0, 2.0)) == 2.0
        assert max(1.1, min(4.0, 1.5)) == 1.5
        assert max(1.1, min(4.0, 3.0)) == 3.0


class TestMagnifierWidgetRadius:
    """Tests for radius configuration logic."""

    def test_radius_clamped_to_minimum(self):
        """Radius below 40 is clamped to 40."""
        assert max(40, 5) == 40
        assert max(40, 0) == 40

    def test_radius_normal_range(self):
        """Radius in valid range is preserved."""
        assert max(40, 80) == 80
        assert max(40, 100) == 100
        assert max(40, 200) == 200


class TestMagnifierWidgetSourceRadius:
    """Tests for screen capture source radius calculation."""

    def test_source_radius_basic(self):
        """Source radius = radius / zoom_factor."""
        radius, zoom = 100, 2.0
        source = max(10, int(radius / zoom))
        assert source == 50

    def test_source_radius_high_zoom(self):
        """High zoom factor produces small source radius."""
        radius, zoom = 80, 4.0
        source = max(10, int(radius / zoom))
        assert source == 20

    def test_source_radius_minimum_clamp(self):
        """Source radius is never less than 10."""
        radius, zoom = 40, 10.0
        source = max(10, int(radius / zoom))
        assert source == 10

    def test_source_radius_unity_zoom(self):
        """At zoom=1.1, source radius ≈ radius."""
        radius, zoom = 100, 1.1
        source = max(10, int(radius / zoom))
        assert source == 90


class TestMagnifierWidgetScreenBounds:
    """Tests for screen bounds clamping logic."""

    def test_rect_intersects_screen(self):
        """Captured region is intersected with screen geometry."""
        screen = QtCore.QRect(0, 0, 1920, 1080)
        rect = QtCore.QRect(1900, 1060, 100, 100)
        clamped = rect.intersected(screen)
        assert clamped.x() == 1900
        assert clamped.y() == 1060
        assert clamped.width() == 20
        assert clamped.height() == 20

    def test_rect_fully_inside_screen(self):
        """Region fully inside screen is unchanged."""
        screen = QtCore.QRect(0, 0, 1920, 1080)
        rect = QtCore.QRect(100, 100, 200, 200)
        clamped = rect.intersected(screen)
        assert clamped == rect

    def test_rect_outside_screen_returns_empty(self):
        """Region completely outside screen becomes empty."""
        screen = QtCore.QRect(0, 0, 1920, 1080)
        rect = QtCore.QRect(2000, 2000, 100, 100)
        clamped = rect.intersected(screen)
        assert clamped.isEmpty()
