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
# Fast test profile: 10 clicks total. Later, raise SAMPLES_PER_POINT or add
# more points for precision once the interaction loop feels good.
CALIBRATION_POINTS = [
    (0.08, 0.10), (0.50, 0.10), (0.92, 0.10),
    (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
    (0.08, 0.90), (0.50, 0.90), (0.92, 0.90),
    (0.50, 0.30),
]
SAMPLES_PER_POINT = 1
