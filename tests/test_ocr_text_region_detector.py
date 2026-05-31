"""Tests for OCR text region detection module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from freehands.ocr.text_region_detector import (
    TextRegion,
    TextRegionDetector,
    MIN_DENSITY,
    MIN_TEXT_REGION_H,
    MIN_TEXT_REGION_W,
)


class TestTextRegion:
    """Tests for the TextRegion dataclass."""

    def test_centre(self):
        region = TextRegion(x=10, y=20, width=100, height=50, density=0.5,
                            contour_count=5, confidence=0.7)
        assert region.centre == (60, 45)

    def test_rect(self):
        region = TextRegion(x=10, y=20, width=100, height=50, density=0.5,
                            contour_count=5, confidence=0.7)
        assert region.rect == (10, 20, 100, 50)

    def test_contains_point_inside(self):
        region = TextRegion(x=0, y=0, width=100, height=100, density=0.5,
                            contour_count=1, confidence=0.5)
        assert region.contains_point(50, 50) is True

    def test_contains_point_edge(self):
        region = TextRegion(x=0, y=0, width=100, height=100, density=0.5,
                            contour_count=1, confidence=0.5)
        assert region.contains_point(0, 0) is True
        assert region.contains_point(100, 100) is False

    def test_contains_point_outside(self):
        region = TextRegion(x=0, y=0, width=100, height=100, density=0.5,
                            contour_count=1, confidence=0.5)
        assert region.contains_point(-1, 50) is False
        assert region.contains_point(50, -1) is False
        assert region.contains_point(101, 50) is False


class TestTextRegionDetector:
    """Tests for the TextRegionDetector class."""

    def test_initial_state(self):
        detector = TextRegionDetector()
        assert detector.region_count == 0
        assert detector.regions == []

    def test_clear(self):
        detector = TextRegionDetector()
        detector._regions = [
            TextRegion(x=0, y=0, width=100, height=50, density=0.5,
                       contour_count=1, confidence=0.5)
        ]
        detector.clear()
        assert detector.region_count == 0

    def test_detect_empty_screen(self):
        """Empty white screen should produce no text regions."""
        detector = TextRegionDetector()
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        regions = detector._find_text_regions(img)
        assert len(regions) == 0

    def test_detect_dark_screen(self):
        """Pure black screen should produce no text regions."""
        detector = TextRegionDetector()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        regions = detector._find_text_regions(img)
        assert len(regions) == 0

    def test_detect_text_like_regions(self):
        """Screen with text-like dark rectangles should detect regions."""
        detector = TextRegionDetector()
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        img[50:100, 50:200] = [20, 20, 20]

        regions = detector._find_text_regions(img)
        assert len(regions) >= 1
        region = regions[0]
        assert region.x >= 0
        assert region.y >= 0
        assert region.width > 0
        assert region.height > 0
        assert 0 < region.density <= 1.0
        assert 0 < region.confidence <= 1.0

    def test_small_regions_filtered(self):
        """Very small dark spots should be filtered out."""
        detector = TextRegionDetector()
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        img[100:105, 100:105] = [20, 20, 20]

        regions = detector._find_text_regions(img)
        for r in regions:
            assert r.width >= MIN_TEXT_REGION_W or r.height >= MIN_TEXT_REGION_H

    def test_density_filtering(self):
        """Sparse dark pixels should not qualify as text."""
        detector = TextRegionDetector()
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        img[50:200, 50:400] = 240
        img[100, 100] = 20
        img[100, 200] = 20
        img[150, 150] = 20

        regions = detector._find_text_regions(img)
        for r in regions:
            assert r.density >= MIN_DENSITY or r.width < MIN_TEXT_REGION_W

    def test_confidence_ranking(self):
        """Higher density regions should have higher confidence."""
        detector = TextRegionDetector()
        img = np.full((600, 800, 3), 240, dtype=np.uint8)
        img[50:150, 50:300] = [20, 20, 20]
        img[200:300, 50:300] = 240
        img[250, 100] = 20
        img[250, 150] = 20
        img[250, 200] = 20

        regions = detector._find_text_regions(img)
        if len(regions) >= 2:
            assert regions[0].confidence >= regions[1].confidence

    def test_merge_adjacent_regions(self):
        """Adjacent regions should be merged."""
        detector = TextRegionDetector()
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        img[50:100, 50:150] = [20, 20, 20]
        img[50:100, 155:255] = [20, 20, 20]

        regions = detector._find_text_regions(img)
        if len(regions) >= 1:
            merged = regions[0]
            assert merged.width >= 200

    def test_grayscale_image(self):
        """Detector should handle grayscale (2D) images."""
        detector = TextRegionDetector()
        img = np.full((480, 640), 240, dtype=np.uint8)
        img[50:100, 50:200] = 20

        regions = detector._find_text_regions(img)
        assert len(regions) >= 1

    def test_scan_cooldown(self):
        """Rapid calls should be throttled by cooldown."""
        detector = TextRegionDetector()
        # Mock capture to return a simple image
        img = np.full((480, 640, 3), 240, dtype=np.uint8)
        img[50:100, 50:200] = [20, 20, 20]

        with patch.object(detector, '_capture_screen', return_value=img):
            detector._last_capture = 0.0
            detector.detect()
            result = detector.detect()
            assert result is detector.regions


class TestCvRgbToBgr:
    """Tests for the cv2_rgb_to_bgr helper."""

    def test_convert_rgb_to_bgr(self):
        from freehands.ocr.text_region_detector import cv2_rgb_to_bgr
        img = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
        result = cv2_rgb_to_bgr(img)
        assert result[0, 0, 0] == 0
        assert result[0, 0, 2] == 255

    def test_non_rgb_image(self):
        from freehands.ocr.text_region_detector import cv2_rgb_to_bgr
        img = np.array([[[255]], [[0]]], dtype=np.uint8)
        result = cv2_rgb_to_bgr(img)
        np.testing.assert_array_equal(result, img)
