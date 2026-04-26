"""Threaded webcam capture with a single-slot frame buffer (always latest)."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float


class Camera:
    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}")
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CameraThread", daemon=True)

    def start(self) -> "Camera":
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                continue
            img = cv2.flip(img, 1)  # mirror so user feels natural
            with self._lock:
                self._latest = Frame(image=img, timestamp=cv2.getTickCount() / cv2.getTickFrequency())

    def read(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()
