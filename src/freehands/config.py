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

# ── Fusion / anti-false-positive defaults ──────────────────────────────────
DEFAULT_DWELL_MS = 600
DEFAULT_STABILITY_FRAMES = 8           # ≈ 270 ms @ 30 fps
DEFAULT_GESTURE_CONFIDENCE = 0.85
COOLDOWN_MS_AFTER_ACTION = 500
PAUSE_GESTURE_HOLD_MS = 1000

# ── Calibration minigame ───────────────────────────────────────────────────
# Gaze calibration uses a denser point map, but each visible point is confirmed
# once after a short stable gaze window instead of repeating the same point.
CALIBRATION_POINTS = [
    (0.00, 0.00), (1.00, 0.00), (1.00, 1.00), (0.00, 1.00),
    (0.50, 0.50),
    (0.50, 0.08), (0.92, 0.50), (0.50, 0.92), (0.08, 0.50),
    (0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75),
    (0.50, 0.30), (0.70, 0.50), (0.50, 0.70), (0.30, 0.50),
]
SAMPLES_PER_POINT = 1
