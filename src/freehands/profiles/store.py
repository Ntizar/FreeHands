"""Profile schema + JSON persistence."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..config import (
    CAMERA_INDEX,
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
    feature_version: int = 1
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
    camera_index: int = CAMERA_INDEX
    pointer_control_enabled: bool = True
    dwell_time_ms: int = DEFAULT_DWELL_MS
    gesture_thresholds: dict[str, GestureThreshold] = Field(
        default_factory=lambda: {
            "thumb_up":    GestureThreshold(),
            "thumb_down":  GestureThreshold(),
            "pointing_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "middle_up":   GestureThreshold(stability_frames=2, confidence_min=0.60),
            "two_fingers_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "left_pointing_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "right_pointing_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "left_middle_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "right_middle_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "left_two_fingers_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "right_two_fingers_up": GestureThreshold(stability_frames=2, confidence_min=0.60),
            "two_hands_together": GestureThreshold(stability_frames=6, confidence_min=0.80),
            "two_hands_apart": GestureThreshold(stability_frames=6, confidence_min=0.80),
            "pinch_open":  GestureThreshold(stability_frames=5, confidence_min=0.90),
            "pinch_close": GestureThreshold(stability_frames=5, confidence_min=0.90),
            "left_open_palm": GestureThreshold(stability_frames=10, confidence_min=0.80),
            "right_open_palm": GestureThreshold(stability_frames=60, confidence_min=0.80),
            "fist_pause":  GestureThreshold(stability_frames=15),
        }
    )
    gesture_bindings: dict[str, str] = Field(
        default_factory=lambda: {
            "thumb_up":    "click",
            "thumb_down":  "escape",
            "pointing_up": "click",
            "middle_up":   "right_click",
            "two_fingers_up": "double_click",
            "left_pointing_up": "click",
            "right_pointing_up": "click",
            "left_middle_up": "right_click",
            "right_middle_up": "right_click",
            "left_two_fingers_up": "double_click",
            "right_two_fingers_up": "double_click",
            "two_hands_together": "zoom_in",
            "two_hands_apart": "zoom_out",
            "pinch_open":  "zoom_in",
            "pinch_close": "zoom_out",
            "tongue_out":  "right_click",
            "left_open_palm": "undo",
            "right_open_palm": "toggle_pause",
            "fist_pause":  "",
        }
    )
    voice_enabled: bool = True
    voice_language: str = "auto"
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


def _threshold_matches(threshold: GestureThreshold, frames: int, confidence: float) -> bool:
    return threshold.stability_frames == frames and abs(threshold.confidence_min - confidence) < 0.001


def _migrate_threshold_if_default(
    profile: Profile,
    gesture: str,
    new_threshold: GestureThreshold,
    old_defaults: tuple[tuple[int, float], ...],
) -> None:
    current = profile.gesture_thresholds.get(gesture)
    if current is None or any(_threshold_matches(current, frames, confidence) for frames, confidence in old_defaults):
        profile.gesture_thresholds[gesture] = new_threshold


def load_profile(user_id: str) -> Profile:
    path = profile_path(user_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Profile '{user_id}' not found. Run: freehands calibrate --user {user_id}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    profile = Profile.model_validate(data)
    defaults = Profile(user_id=user_id)
    for gesture, threshold in defaults.gesture_thresholds.items():
        profile.gesture_thresholds.setdefault(gesture, threshold)
    for gesture in (
        "pointing_up", "middle_up", "two_fingers_up",
        "left_pointing_up", "right_pointing_up",
        "left_middle_up", "right_middle_up",
        "left_two_fingers_up", "right_two_fingers_up",
    ):
        _migrate_threshold_if_default(
            profile,
            gesture,
            GestureThreshold(stability_frames=2, confidence_min=0.60),
            ((8, 0.85), (5, 0.75), (3, 0.70), (2, 0.60)),
        )
    _migrate_threshold_if_default(
        profile,
        "left_open_palm",
        GestureThreshold(stability_frames=10, confidence_min=0.80),
        ((8, 0.85), (10, 0.80)),
    )
    _migrate_threshold_if_default(
        profile,
        "right_open_palm",
        GestureThreshold(stability_frames=60, confidence_min=0.80),
        ((8, 0.85), (12, 0.80), (60, 0.80)),
    )
    for gesture, action in defaults.gesture_bindings.items():
        profile.gesture_bindings.setdefault(gesture, action)
    for gesture, action in {
        "left_pointing_up": "click",
        "right_pointing_up": "click",
        "left_middle_up": "right_click",
        "right_middle_up": "right_click",
        "left_two_fingers_up": "double_click",
        "right_two_fingers_up": "double_click",
        "left_open_palm": "undo",
    }.items():
        profile.gesture_bindings.setdefault(gesture, action)
    profile.gesture_bindings.setdefault("right_open_palm", "toggle_pause")
    profile.gesture_bindings["fist_pause"] = ""
    return profile


def get_or_create_profile(user_id: str) -> Profile:
    try:
        return load_profile(user_id)
    except FileNotFoundError:
        p = Profile(user_id=user_id)
        save_profile(p)
        return p
