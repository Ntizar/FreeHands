"""OCR-free text region detection for gaze typing.

Detects text-like regions on screen using contrast analysis and geometry.
Works without Tesseract — identifies rectangular high-contrast regions that
match the visual signature of text blocks (dense dark pixels on light background
in a rectangular layout).

This module captures the screen, converts to grayscale, and finds candidate
text regions using adaptive thresholding and contour analysis. Each detected
region is returned with its bounding box and a confidence score.

Design rules
------------
* No external OCR engine required — uses OpenCV contour analysis only.
* Screen capture via ``mss`` for performance (faster than PIL).
* Regions are filtered by size, aspect ratio, and pixel density to avoid
  false positives (images, buttons, icons).
* Returns ``TextRegion`` objects with screen coordinates and density metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

# Minimum/maximum bounding box sizes (pixels) for a text region
MIN_TEXT_REGION_W: Final[int] = 30
MIN_TEXT_REGION_H: Final[int] = 12
MAX_TEXT_REGION_W: Final[int] = 2000
MAX_TEXT_REGION_H: Final[int] = 500

# Minimum pixel density (dark pixels / total pixels) to qualify as text
MIN_DENSITY: Final[float] = 0.15

# Maximum aspect ratio to avoid long thin lines (dividers, rules)
MAX_ASPECT_RATIO: Final[float] = 20.0

# Minimum number of contours inside a region to qualify as text block
MIN_CONTOURS: Final[int] = 3

# Screen capture interval (seconds) — throttle to avoid excessive CPU
CAPTURE_COOLDOWN: Final[float] = 1.0


@dataclass
class TextRegion:
    """A detected text region on screen."""
    x: int                         # top-left x
    y: int                         # top-left y
    width: int                     # bounding box width
    height: int                    # bounding box height
    density: float                 # dark pixel ratio (0..1)
    contour_count: int             # number of internal contours
    confidence: float              # 0..1 score combining density + geometry

    @property
    def centre(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def contains_point(self, px: int, py: int) -> bool:
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)


class TextRegionDetector:
    """Detects text regions on screen using OpenCV contour analysis.

    The detection pipeline:
    1. Capture screen via mss (fast, no GUI dependency)
    2. Convert to grayscale
    3. Apply adaptive threshold to isolate text-like dark regions
    4. Find contours and filter by size, density, and geometry
    5. Group nearby contours into text blocks
    6. Return sorted list of TextRegion objects

    Throttled by CAPTURE_COOLDOWN to avoid excessive CPU usage.
    """

    def __init__(self) -> None:
        self._last_capture: float = 0.0
        self._regions: list[TextRegion] = []
        self._last_result_time: float = 0.0

    @property
    def regions(self) -> list[TextRegion]:
        return self._regions

    @property
    def region_count(self) -> int:
        return len(self._regions)

    def detect(self) -> list[TextRegion]:
        """Capture screen and detect text regions.

        Returns the list of detected TextRegion objects. Results are cached
        for CAPTURE_COOLDOWN seconds to avoid redundant captures.

        Uses mss for fast screen capture. Falls back to pyautogui if mss
        is unavailable.
        """
        now = time.monotonic()
        if now - self._last_capture < CAPTURE_COOLDOWN:
            return self._regions

        self._last_capture = now

        # Capture screen
        img = self._capture_screen()
        if img is None:
            return []

        # Detect regions
        self._regions = self._find_text_regions(img)
        self._last_result_time = now
        return self._regions

    def _capture_screen(self) -> np.ndarray | None:
        """Capture the screen as a numpy array (BGR, OpenCV format)."""
        try:
            import mss
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # primary monitor
                screenshot = sct.grab(monitor)
                img = np.asarray(screenshot)
                # mss returns BGRA → convert to BGR (OpenCV standard)
                return img[:, :, :3]
        except ImportError:
            pass

        # Fallback: pyautogui → PIL → numpy
        try:
            import pyautogui
            screenshot = pyautogui.screenshot()
            img = np.asarray(screenshot)
            # PIL RGB → BGR
            return cv2_rgb_to_bgr(img)
        except Exception:
            return None

    def _find_text_regions(self, img: np.ndarray) -> list[TextRegion]:
        """Find text-like regions in a grayscale image."""
        import cv2

        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Adaptive threshold to isolate text (dark text on light bg)
        # Use adaptiveMean to handle varying lighting
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,  # text is dark → becomes white
            15,  # block size
            8,   # C constant
        )

        # Morphological operations to connect broken text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and group contours into text regions
        regions: list[TextRegion] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Size filters
            if (w < MIN_TEXT_REGION_W or h < MIN_TEXT_REGION_H or
                    w > MAX_TEXT_REGION_W or h > MAX_TEXT_REGION_H):
                continue

            # Aspect ratio filter
            if w > 0 and (w / h > MAX_ASPECT_RATIO):
                continue

            # Pixel density: how much of the bounding box is text pixels
            roi = eroded[y:y + h, x:x + w]
            total_pixels = w * h
            text_pixels = cv2.countNonZero(roi)
            density = text_pixels / total_pixels if total_pixels > 0 else 0.0

            if density < MIN_DENSITY:
                continue

            # Confidence: combine density and size factors
            size_factor = min(1.0, (w * h) / (8000))  # prefer medium-sized regions
            confidence = density * 0.6 + size_factor * 0.4

            regions.append(TextRegion(
                x=x, y=y, width=w, height=h,
                density=density,
                contour_count=1,
                confidence=confidence,
            ))

        # Group nearby regions into blocks (merge overlapping/adjacent boxes)
        regions = self._merge_regions(regions)

        # Sort by confidence descending (most text-like first)
        regions.sort(key=lambda r: r.confidence, reverse=True)

        return regions

    def _merge_regions(self, regions: list[TextRegion]) -> list[TextRegion]:
        """Merge overlapping or adjacent text regions into blocks."""
        if not regions:
            return regions

        # Sort by x position for merging
        regions.sort(key=lambda r: r.x)

        merged: list[TextRegion] = [regions[0]]
        for region in regions[1:]:
            prev = merged[-1]
            # Check if regions overlap or are adjacent (within 20px gap)
            gap_threshold = 20
            if (region.x < prev.x + prev.width + gap_threshold and
                    region.y < prev.y + prev.height + gap_threshold and
                    region.y + region.height > prev.y):
                # Merge: expand bounding box
                new_x = min(prev.x, region.x)
                new_y = min(prev.y, region.y)
                new_w = max(prev.x + prev.width, region.x + region.width) - new_x
                new_h = max(prev.y + prev.height, region.y + region.height) - new_y
                new_density = (prev.density + region.density) / 2
                new_conf = (prev.confidence + region.confidence) / 2
                merged[-1] = TextRegion(
                    x=new_x, y=new_y, width=new_w, height=new_h,
                    density=new_density,
                    contour_count=prev.contour_count + region.contour_count,
                    confidence=new_conf,
                )
            else:
                merged.append(region)

        return merged

    def clear(self) -> None:
        """Clear cached regions."""
        self._regions = []
        self._last_capture = 0.0


def cv2_rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array to BGR (OpenCV format)."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return img[:, :, ::-1]
    return img
