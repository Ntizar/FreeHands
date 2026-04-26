"""Face / tongue gesture stub (Phase 4).

Placeholder that always returns ``None`` so the rest of the pipeline can be
wired today. A real implementation will train a small CNN on the
mouth ROI cropped via FaceMesh landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FaceObservation:
    tongue_out: bool = False
    confidence: float = 0.0


class FaceTracker:
    def detect(self, frame_bgr) -> FaceObservation:
        return FaceObservation()

    def close(self) -> None:
        pass
