import cv2
import numpy as np

from freehands.gaze.tracker import _detect_dark_pupil


def test_detect_dark_pupil_finds_dark_blob_inside_eye_polygon() -> None:
    gray = np.full((80, 140), 220, dtype=np.uint8)
    eye = [
        np.array([35, 34]),
        np.array([55, 24]),
        np.array([88, 24]),
        np.array([108, 34]),
        np.array([88, 46]),
        np.array([55, 46]),
    ]
    cv2.circle(gray, (72, 35), 8, 20, -1)

    pupil = _detect_dark_pupil(gray, eye)

    assert pupil is not None
    np.testing.assert_allclose(pupil, np.array([72, 35]), atol=3)