"""Threaded webcam capture with a single-slot frame buffer (always latest)."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from os import name as os_name

import cv2
import numpy as np

from ..config import CAMERA_INDEX


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float


class Camera:
    def __init__(self, index: int = CAMERA_INDEX) -> None:
        self._index = index
        self._cap = self._open_capture(index)
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CameraThread", daemon=True)

    @staticmethod
    def _open_capture(index: int) -> cv2.VideoCapture:
        for backend in _capture_backends():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            for _ in range(3):
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        raise RuntimeError(f"Could not open camera {index} with readable frames")

    @property
    def index(self) -> int:
        return self._index

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
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()

    def reopen(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
        with self._lock:
            self._latest = None
        self._cap = self._open_capture(self._index)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CameraThread", daemon=True)
        self._thread.start()

    def switch(self, index: int) -> None:
        if index == self._index:
            self.reopen()
            return
        new_cap = self._open_capture(index)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._cap.release()
        with self._lock:
            self._latest = None
        self._index = index
        self._cap = new_cap
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CameraThread", daemon=True)
        self._thread.start()


def list_available_cameras(max_index: int = 4) -> list[int]:
    found: list[int] = []
    for index in range(max_index):
        for backend in _probe_backends():
            cap = cv2.VideoCapture(index, backend)
            ok = False
            try:
                if cap.isOpened():
                    for _ in range(1):
                        ok, _ = cap.read()
                        if ok:
                            break
            finally:
                cap.release()
            if ok:
                found.append(index)
                break
    return found


def _capture_backends() -> list[int]:
    if os_name != "nt":
        return [cv2.CAP_ANY]
    return _unique_backends([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY])


def _probe_backends() -> list[int]:
    if os_name != "nt":
        return [cv2.CAP_ANY]
    return _unique_backends([cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY])


def _unique_backends(backends: list[int]) -> list[int]:
    unique: list[int] = []
    for backend in backends:
        if backend not in unique:
            unique.append(backend)
    return unique
