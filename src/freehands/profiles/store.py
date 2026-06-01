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


# ── External gesture profiles ──────────────────────────────────────────────
GESTURE_PROFILES_DIR = PROFILES_DIR / "gesture_profiles"
GESTURE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def gesture_profiles_dir() -> Path:
    """Return the directory where external gesture profile JSONs live."""
    return GESTURE_PROFILES_DIR


def list_gesture_profiles() -> list[str]:
    """Return sorted list of gesture profile names (without .json extension)."""
    if not GESTURE_PROFILES_DIR.exists():
        return []
    return sorted(
        p.stem for p in GESTURE_PROFILES_DIR.glob("*.json")
        if p.is_file()
    )


class GestureProfileOverride(BaseModel):
    """External gesture bindings/thresholds override."""
    gesture_bindings: dict[str, str] = Field(default_factory=dict)
    gesture_thresholds: dict[str, dict[str, float | int | bool]] = Field(default_factory=dict)


def load_gesture_profile(name: str) -> GestureProfileOverride:
    """Load an external gesture profile by name (without .json).

    The profile is a JSON file in the gesture_profiles directory containing
    optional ``gesture_bindings`` and ``gesture_thresholds`` overrides.
    Missing keys simply mean "use defaults from the main profile".
    """
    path = GESTURE_PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Gesture profile '{name}' not found. "
            f"Create it at {GESTURE_PROFILES_DIR}/{name}.json"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return GestureProfileOverride.model_validate(data)


def merge_gesture_profile(profile: Profile, gesture_name: str) -> None:
    """Apply an external gesture profile on top of a user profile (in-place).

    External bindings override defaults; external thresholds merge field-by-field.
    Repairs only touch gestures NOT explicitly set in the external profile.
    """
    ext = load_gesture_profile(gesture_name)
    explicit_gestures = set(ext.gesture_bindings.keys())

    # Merge bindings: external overrides defaults
    profile.gesture_bindings.update(ext.gesture_bindings)

    # Merge thresholds: external fields override per-gesture
    VALID_THRESHOLD_KEYS = {"confidence_min", "stability_frames"}
    for gesture, fields in ext.gesture_thresholds.items():
        filtered: dict[str, int | float] = {}
        for k, v in fields.items():
            if k in VALID_THRESHOLD_KEYS:
                filtered[k] = int(v) if k == "stability_frames" else float(v)  # type: ignore[assignment]
        existing = profile.gesture_thresholds.get(gesture)
        if existing is None:
            profile.gesture_thresholds[gesture] = GestureThreshold(**filtered)  # type: ignore[arg-type]
        else:
            for key, value in filtered.items():
                if key == "confidence_min":
                    existing.confidence_min = value  # type: ignore[attr-defined]
                elif key == "stability_frames":
                    existing.stability_frames = value  # type: ignore[attr-defined]

    # Re-run repairs after merge to maintain invariants
    _repair_instant_mouse_thresholds(profile)
    _repair_essential_bindings(
        profile.gesture_bindings,
        set(profile.gesture_thresholds),
        explicit_gestures=explicit_gestures,
    )
    _dedupe_bindings(profile.gesture_bindings)
    _repair_essential_bindings(
        profile.gesture_bindings,
        set(profile.gesture_thresholds),
        explicit_gestures=explicit_gestures,
    )


# ── Original constants ─────────────────────────────────────────────────────
GESTURE_BINDING_PRIORITY = [
    "pointing_up",
    "middle_up",
    "two_fingers_up",
    "right_open_palm",
    "left_open_palm",
    "two_hands_together",
    "two_hands_apart",
    "thumb_down",
    "left_pointing_up",
    "right_pointing_up",
    "left_middle_up",
    "right_middle_up",
    "left_two_fingers_up",
    "right_two_fingers_up",
    "thumb_up",
    "pinch_open",
    "pinch_close",
    "fist_pause",
    # Palm-scroll gestures (motion-based, auto-detected)
    "palm_scroll_up",
    "palm_scroll_down",
    "left_palm_scroll_up",
    "left_palm_scroll_down",
    "right_palm_scroll_up",
    "right_palm_scroll_down",
    # Air-scroll / swipe gestures (motion-based, auto-detected)
    "air_scroll_up",
    "air_scroll_down",
    "left_air_scroll_up",
    "left_air_scroll_down",
    "right_air_scroll_up",
    "right_air_scroll_down",
    # Facial expression gestures (state-based, auto-detected)
    "smile",
    "frown",
    "surprise",
    "raised_eyebrows",
    "furrowed_brows",
    "mouth_open",
    "tongue_out",
    # Volume control by hand vertical position (auto-detected)
    "volume_up",
    "volume_down",
]

INSTANT_MOUSE_GESTURES = (
    "pointing_up",
    "middle_up",
    "two_fingers_up",
    "left_pointing_up",
    "right_pointing_up",
    "left_middle_up",
    "right_middle_up",
    "left_two_fingers_up",
    "right_two_fingers_up",
    # Palm-scroll gestures are inherently instant (motion-based)
    "palm_scroll_up",
    "palm_scroll_down",
    "left_palm_scroll_up",
    "left_palm_scroll_down",
    "right_palm_scroll_up",
    "right_palm_scroll_down",
    # Air-scroll / swipe gestures are inherently instant (motion-based)
    "air_scroll_up",
    "air_scroll_down",
    "left_air_scroll_up",
    "left_air_scroll_down",
    "right_air_scroll_up",
    "right_air_scroll_down",
    # Volume control gestures are inherently instant (position-based)
    "volume_up",
    "volume_down",
)

CLICK_FAMILY_GESTURES = {
    "click": ("pointing_up", "left_pointing_up", "right_pointing_up"),
    "right_click": ("middle_up", "left_middle_up", "right_middle_up"),
    "double_click": ("two_fingers_up", "left_two_fingers_up", "right_two_fingers_up"),
}

ESSENTIAL_GESTURE_BINDINGS = {
    "pointing_up": "click",
    "middle_up": "right_click",
    "two_fingers_up": "double_click",
}

LEGACY_UNDETECTABLE_GESTURES = {"tongue_out"}


class GestureThreshold(BaseModel):
    confidence_min: float = DEFAULT_GESTURE_CONFIDENCE
    stability_frames: int = DEFAULT_STABILITY_FRAMES


class GazeModel(BaseModel):
    type: str = "ridge_regression"
    feature_version: int = 1
    # weights_x/y can be either flat (ridge: list[float]) or 2D (knn: list[list[float]])
    weights_x: list[float] | list[list[float]] = Field(default_factory=list)
    weights_y: list[float] = Field(default_factory=list)
    bias_x: list[float] | float = Field(default=0.0)
    bias_y: float | int = 0
    personal_offset: dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0})
    # KNN-specific: instance-based storage (Deer Mouse pattern)
    # training_features stores the X matrix (list of design vectors)
    training_features: list[list[float]] = Field(default_factory=list)
    # training_targets_x stores the y_x vector
    training_targets_x: list[float] = Field(default_factory=list)
    # training_targets_y stores the y_y vector
    training_targets_y: list[float] = Field(default_factory=list)
    # k stores the number of neighbours for KNN
    k: int = 0


class GPGazeModel(BaseModel):
    """Serializable Gaussian Process model for gaze calibration.

    Stores kernel hyperparameters and a subset of training samples so the
    GP can be reconstituted (retrained) from JSON.  The GP itself is not
    serialisable, so we persist the ingredients and rebuild on load.
    """
    kernel_type: str = "RBF"          # RBF, Matern, ConstantKernel
    lengthscale: float = 1.0
    noise_level: float = 0.1
    alpha: float = 1e-6               # regularisation
    feature_version: int = 1
    training_features: list[list[float]] = Field(default_factory=list)  # X
    training_targets_x: list[list[float]] = Field(default_factory=list)  # y_x
    training_targets_y: list[list[float]] = Field(default_factory=list)  # y_y
    n_samples: int = 0                # quick check that data is present


class Profile(BaseModel):
    user_id: str
    calibration_date: str = Field(default_factory=lambda: date.today().isoformat())
    gaze_calibrated_at: str | None = None
    gesture_calibrated_at: str | None = None
    gesture_calibration_results: dict[str, dict[str, float | int | bool]] = Field(default_factory=dict)
    gaze_model: GazeModel = Field(default_factory=GazeModel)
    gp_model: GPGazeModel = Field(default_factory=GPGazeModel)
    camera_index: int = CAMERA_INDEX
    swap_handedness: bool = False
    pointer_control_enabled: bool = True
    dwell_time_ms: int = DEFAULT_DWELL_MS
    gesture_thresholds: dict[str, GestureThreshold] = Field(
        default_factory=lambda: {
            "thumb_up":    GestureThreshold(),
            "thumb_down":  GestureThreshold(),
            "pointing_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "middle_up":   GestureThreshold(stability_frames=1, confidence_min=0.50),
            "two_fingers_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "left_pointing_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "right_pointing_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "left_middle_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "right_middle_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "left_two_fingers_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "right_two_fingers_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "two_hands_together": GestureThreshold(stability_frames=6, confidence_min=0.80),
            "two_hands_apart": GestureThreshold(stability_frames=6, confidence_min=0.80),
            "pinch_open":  GestureThreshold(stability_frames=5, confidence_min=0.90),
            "pinch_close": GestureThreshold(stability_frames=5, confidence_min=0.90),
            "left_open_palm": GestureThreshold(stability_frames=60, confidence_min=0.80),
            "right_open_palm": GestureThreshold(stability_frames=60, confidence_min=0.80),
            "fist_pause":  GestureThreshold(stability_frames=15),
            # Facial expression gestures (state-based, moderate stability)
            "smile": GestureThreshold(stability_frames=8, confidence_min=0.70),
            "frown": GestureThreshold(stability_frames=8, confidence_min=0.70),
            "surprise": GestureThreshold(stability_frames=3, confidence_min=0.80),
            "raised_eyebrows": GestureThreshold(stability_frames=6, confidence_min=0.70),
            "furrowed_brows": GestureThreshold(stability_frames=6, confidence_min=0.70),
            "mouth_open": GestureThreshold(stability_frames=5, confidence_min=0.65),
            "tongue_out": GestureThreshold(stability_frames=5, confidence_min=0.65),
            # Volume control gestures (auto-detected, instant)
            "volume_up": GestureThreshold(stability_frames=1, confidence_min=0.50),
            "volume_down": GestureThreshold(stability_frames=1, confidence_min=0.50),
        }
    )
    gesture_bindings: dict[str, str] = Field(
        default_factory=lambda: {
            "thumb_up":    "",
            "thumb_down":  "escape",
            "pointing_up": "click",
            "middle_up":   "right_click",
            "two_fingers_up": "double_click",
            "left_pointing_up": "",
            "right_pointing_up": "",
            "left_middle_up": "",
            "right_middle_up": "",
            "left_two_fingers_up": "",
            "right_two_fingers_up": "",
            "two_hands_together": "zoom_in",
            "two_hands_apart": "zoom_out",
            "pinch_open":  "",
            "pinch_close": "",
            "tongue_out":  "",
            "left_open_palm": "",
            "right_open_palm": "toggle_pause",
            "fist_pause":  "",
            # Palm-scroll gestures (motion-based, auto-detected)
            "palm_scroll_up": "scroll_up",
            "palm_scroll_down": "scroll_down",
            "left_palm_scroll_up": "scroll_up",
            "left_palm_scroll_down": "scroll_down",
            "right_palm_scroll_up": "scroll_up",
            "right_palm_scroll_down": "scroll_down",
            # Air-scroll / swipe gestures (motion-based, auto-detected)
            # Works with any hand pose — pointing, fist, open palm, etc.
            "air_scroll_up": "scroll_up",
            "air_scroll_down": "scroll_down",
            "left_air_scroll_up": "scroll_up",
            "left_air_scroll_down": "scroll_down",
            "right_air_scroll_up": "scroll_up",
            "right_air_scroll_down": "scroll_down",
            # Facial expression gestures (state-based, auto-detected via FaceMesh)
            # Map to actions — default bindings are empty; user can customise.
            "smile": "",
            "frown": "",
            "surprise": "",
            "raised_eyebrows": "",
            "furrowed_brows": "",
            "mouth_open": "",
            "tongue_out": "",
            # Volume control by hand vertical position (auto-detected)
            # These are handled directly by the VolumeControl module,
            # not through gesture bindings — always active when hand visible.
            "volume_up": "",
            "volume_down": "",
        }
    )
    voice_enabled: bool = True
    voice_language: str = "auto"
    voice_asr_backend: str = "faster_whisper"
    voice_tts_backend: str = "none"
    voice_wake_words: list[str] = Field(default_factory=lambda: ["freehands", "free hands", "ntizar"])
    voice_vosk_model_path: str = ""
    voice_model_size: str = "tiny"
    audio_feedback_enabled: bool = True
    magnification_enabled: bool = True
    magnification_factor: float = 2.0
    plugins_dir: str = ""


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


def _dedupe_bindings(bindings: dict[str, str]) -> None:
    seen_actions: set[str] = set()
    ordered_gestures = [*GESTURE_BINDING_PRIORITY, *[gesture for gesture in bindings if gesture not in GESTURE_BINDING_PRIORITY]]
    for gesture in ordered_gestures:
        action = bindings.get(gesture, "")
        if not action:
            continue
        if action in seen_actions:
            bindings[gesture] = ""
        else:
            seen_actions.add(action)


def _has_detectable_action(bindings: dict[str, str], detectable_gestures: set[str], action: str) -> bool:
    return any(bindings.get(gesture) == action for gesture in detectable_gestures)


def _custom_non_family_click_actions(bindings: dict[str, str]) -> set[str]:
    actions: set[str] = set()
    for action, family_gestures in CLICK_FAMILY_GESTURES.items():
        family = set(family_gestures)
        for gesture, bound_action in bindings.items():
            if gesture in LEGACY_UNDETECTABLE_GESTURES:
                continue
            if bound_action == action and gesture not in family:
                actions.add(action)
    return actions


def _repair_instant_mouse_thresholds(profile: Profile) -> None:
    for gesture in INSTANT_MOUSE_GESTURES:
        current = profile.gesture_thresholds.get(gesture)
        if current is None or current.stability_frames > 3 or current.confidence_min > 0.75:
            profile.gesture_thresholds[gesture] = GestureThreshold(stability_frames=1, confidence_min=0.50)


def _repair_essential_bindings(
    bindings: dict[str, str],
    detectable_gestures: set[str],
    explicit_gestures: set[str] | None = None,
) -> None:
    explicit_gestures = explicit_gestures or set()
    for gesture in LEGACY_UNDETECTABLE_GESTURES:
        bindings[gesture] = ""

    for gesture in INSTANT_MOUSE_GESTURES:
        if bindings.get(gesture) == "toggle_pause":
            bindings[gesture] = ""

    if not _has_detectable_action(bindings, detectable_gestures, "toggle_pause"):
        bindings["right_open_palm"] = "toggle_pause"

    for action, gestures in CLICK_FAMILY_GESTURES.items():
        generic_gesture = gestures[0]
        if any(bindings.get(gesture) == action for gesture in gestures):
            bindings[generic_gesture] = action
            for side_gesture in gestures[1:]:
                if bindings.get(side_gesture) == action:
                    bindings[side_gesture] = ""
        elif not _has_detectable_action(bindings, detectable_gestures, action):
            # Only assign default if not explicitly overridden in external profile
            if generic_gesture not in explicit_gestures:
                bindings[generic_gesture] = action


def load_profile(user_id: str) -> Profile:
    path = profile_path(user_id)
    if not path.exists():
        raise FileNotFoundError(
            f"Profile '{user_id}' not found. Run: freehands calibrate --user {user_id}"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    raw_bindings = data.get("gesture_bindings", {})
    explicit_bindings = set(raw_bindings) if isinstance(raw_bindings, dict) else set()
    has_custom_pause_binding = isinstance(raw_bindings, dict) and any(
        gesture != "right_open_palm"
        and gesture not in INSTANT_MOUSE_GESTURES
        and action == "toggle_pause"
        for gesture, action in raw_bindings.items()
    )
    profile = Profile.model_validate(data)
    defaults = Profile(user_id=user_id)
    for gesture, threshold in defaults.gesture_thresholds.items():
        profile.gesture_thresholds.setdefault(gesture, threshold)
    _repair_instant_mouse_thresholds(profile)
    _migrate_threshold_if_default(
        profile,
        "left_open_palm",
        GestureThreshold(stability_frames=60, confidence_min=0.80),
        ((8, 0.85), (10, 0.80), (60, 0.80)),
    )
    _migrate_threshold_if_default(
        profile,
        "right_open_palm",
        GestureThreshold(stability_frames=60, confidence_min=0.80),
        ((8, 0.85), (12, 0.80), (60, 0.80)),
    )
    for gesture, action in defaults.gesture_bindings.items():
        profile.gesture_bindings.setdefault(gesture, action)
    custom_non_family_actions = _custom_non_family_click_actions(raw_bindings) if isinstance(raw_bindings, dict) else set()
    for action in custom_non_family_actions:
        generic_gesture = CLICK_FAMILY_GESTURES[action][0]
        if generic_gesture not in explicit_bindings:
            profile.gesture_bindings[generic_gesture] = ""
    if has_custom_pause_binding and "right_open_palm" not in explicit_bindings:
        profile.gesture_bindings["right_open_palm"] = ""
    profile.gesture_bindings.setdefault("left_open_palm", "")
    profile.gesture_bindings["fist_pause"] = ""
    detectable_gestures = set(defaults.gesture_thresholds)
    _repair_essential_bindings(profile.gesture_bindings, detectable_gestures)
    _dedupe_bindings(profile.gesture_bindings)
    _repair_essential_bindings(profile.gesture_bindings, detectable_gestures)
    return profile


def get_or_create_profile(user_id: str) -> Profile:
    try:
        return load_profile(user_id)
    except FileNotFoundError:
        p = Profile(user_id=user_id)
        save_profile(p)
        return p
