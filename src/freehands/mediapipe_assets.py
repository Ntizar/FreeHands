"""Download/cache MediaPipe Tasks models used by FreeHands."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from .config import DATA_DIR

MODEL_DIR = DATA_DIR / "models" / "mediapipe"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "face_landmarker": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    ),
    "gesture_recognizer": (
        "gesture_recognizer.task",
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    ),
}


def ensure_model(name: str) -> Path:
    if name not in MODELS:
        raise KeyError(f"Unknown MediaPipe model: {name}")
    filename, url = MODELS[name]
    path = MODEL_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(path.suffix + ".download")
    print(f"[FreeHands] Downloading MediaPipe model: {name}")
    urlretrieve(url, tmp)
    tmp.replace(path)
    return path
