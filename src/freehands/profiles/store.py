"""Profile schema + JSON persistence."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import (
    DEFAULT_DWELL_MS,
    DEFAULT_GESTURE_CONFIDENCE,
    DEFAULT_STABILITY_FRAMES,
    PROFILES_DIR,
)


class GestureThreshold(BaseModel):
    confidence_min: float = DEFAULT_GESTURE_CONFIDENCE
    stability_frames: int = DEFAULT_STABILITY_FRAMES


class GazeModel(BaseModel):
    type: str = "ridge_regression"
    weights_x: list[float] = Field(default_factory=list)
    weights_y: list[float] = Field(default_factory=list)
    bias_x: float = 0.0
    bias_y: float = 0.0
    personal_offset: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})


class Profile(BaseModel):
    user_id: str
    calibration_date: str = Field(default_factory=lambda: date.today().isoformat())
    gaze_calibrated_at: str | None = None
    gesture_calibrated_at: str | None = None
    gesture_calibration_results: dict[str, dict[str, float | int | bool]] = Field(default_factory=dict)
    gaze_model: GazeModel = Field(default_factory=GazeModel)
    dwell_time_ms: int = DEFAULT_DWELL_MS
    gesture_thresholds: dict[str, GestureThreshold] = Field(
        default_factory=lambda: {
            "thumb_up":    GestureThreshold(),
            "thumb_down":  GestureThreshold(),
            "pinch_open":  GestureThreshold(stability_frames=5, confidence_min=0.90),
            "pinch_close": GestureThreshold(stability_frames=5, confidence_min=0.90),
            "fist_pause":  GestureThreshold(stability_frames=15),
        }
    )
    gesture_bindings: dict[str, str] = Field(
        default_factory=lambda: {
            "thumb_up":    "click",
            "thumb_down":  "escape",
            "pinch_open":  "zoom_in",
            "pinch_close": "zoom_out",
            "tongue_out":  "right_click",
            "fist_pause":  "toggle_pause",
        }
    )
    voice_enabled: bool = True
    voice_language: str = "es"
    voice_asr_backend: str = "faster_whisper"
    voice_tts_backend: str = "none"
    voice_wake_words: list[str] = Field(default_factory=lambda: ["freehands", "free hands", "ntizar"])
    magnification_enabled: bool = True
    magnification_factor: float = 2.0


# ── I/O ────────────────────────────────────────────────────────────────────
def profile_path(user_id: str) -> Path:
    return PROFILES_DIR / f"{user_id}.json"


def save_profile(profile: Profile) -> Path:
    path = profile_path(profile.user_id)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_profile(user_id: str) -> Profile:
    path = profile_path(user_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Profile '{user_id}' not found. Run: freehands calibrate --user {user_id}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return Profile.model_validate(data)


def get_or_create_profile(user_id: str) -> Profile:
    try:
        return load_profile(user_id)
    except FileNotFoundError:
        p = Profile(user_id=user_id)
        save_profile(p)
        return p
