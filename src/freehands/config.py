"""Global constants and runtime configuration."""
from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir

# ── Paths ──────────────────────────────────────────────────────────────────
APP_NAME = "FreeHands"
APP_AUTHOR = "Ntizar"

DATA_DIR = Path(user_data_dir(APP_NAME, APP_AUTHOR))
PROFILES_DIR = DATA_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# ── Capture ────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
POINTER_MOVE_INTERVAL_MS = 35
POINTER_MOVE_MIN_DELTA_PX = 3
POINTER_FINE_AIM_HOLD_MS = 1000
POINTER_FINE_AIM_RADIUS_PX = 70
POINTER_FINE_AIM_RELEASE_PX = 135
POINTER_FINE_AIM_ALPHA = 0.22

# ── Fusion / anti-false-positive defaults ──────────────────────────────────
DEFAULT_DWELL_MS = 600
DEFAULT_STABILITY_FRAMES = 8           # ≈ 270 ms @ 30 fps
DEFAULT_GESTURE_CONFIDENCE = 0.85
COOLDOWN_MS_AFTER_ACTION = 500
PAUSE_GESTURE_HOLD_MS = 1000

# ── Calibration minigame ───────────────────────────────────────────────────
# Gaze calibration uses a standard 9-point grid (3×3) — corners, edge midpoints,
# and centre. This is the ISO-standard layout used by WebGazer, EyeTrack, etc.
# Each point is confirmed once after a short stable gaze window.
CALIBRATION_POINTS = [
    (0.00, 0.00), (0.50, 0.00), (1.00, 0.00),   # top row: left, centre, right
    (0.00, 0.50), (0.50, 0.50), (1.00, 0.50),   # middle row: left, centre, right
    (0.00, 1.00), (0.50, 1.00), (1.00, 1.00),   # bottom row: left, centre, right
]
SAMPLES_PER_POINT = 1
