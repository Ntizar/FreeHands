"""Diagnostics: verify camera, microphone, and key dependencies."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repair_dependencies() -> int:
    """Install/reinstall runtime dependencies into the active Python env."""
    root = Path(__file__).resolve().parents[2]
    requirements = root / "requirements.txt"
    print("FreeHands · repair")
    print("=" * 50)
    print(f"Python: {sys.executable}")
    if not requirements.exists():
        print(f"[ERR] requirements.txt not found at {requirements}")
        return 1

    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "mediapipe"],
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        [sys.executable, "-m", "pip", "install", "-e", str(root)],
    ]
    for cmd in commands:
        print("$ " + " ".join(cmd))
        completed = subprocess.run(cmd, cwd=root)
        if completed.returncode != 0:
            print(f"[ERR] command failed with {completed.returncode}")
            return completed.returncode
    return run_doctor()


def run_doctor() -> int:
    print("FreeHands · doctor")
    print("=" * 50)

    ok = True

    # Camera
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            print(f"  [OK] Camera 0 reachable (frame={ret})")
        else:
            print("  [ERR] Camera 0 not available"); ok = False
        cap.release()
    except Exception as e:
        print(f"  [ERR] OpenCV error: {e}"); ok = False

    # MediaPipe trackers (legacy solutions or Tasks, depending on installed version)
    try:
        from .gaze import GazeTracker
        from .gestures import HandTracker

        gaze = GazeTracker()
        gaze.close()
        print("  [OK] GazeTracker available")
        hands = HandTracker()
        hands.close()
        print("  [OK] HandTracker available")
    except Exception as e:
        print(f"  [ERR] MediaPipe tracker missing/broken: {e}"); ok = False
        print("       Run: FreeHands.bat repair")

    # PyQt6
    try:
        from PyQt6 import QtCore  # noqa: F401
        print("  [OK] PyQt6 importable")
    except Exception as e:
        print(f"  [ERR] PyQt6 missing: {e}"); ok = False

    # Microphone (optional)
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        ins = [d for d in devs if d.get("max_input_channels", 0) > 0]
        print(f"  [OK] {len(ins)} input audio device(s) detected")
    except Exception as e:
        print(f"  [!] Audio not available: {e}")

    # Whisper (optional, Phase 3 voice)
    try:
        import faster_whisper  # noqa: F401
        print("  [OK] faster-whisper importable")
    except Exception as e:
        print(f"  [!] faster-whisper not available: {e}")

    print("=" * 50)
    print("OK" if ok else "Issues detected")
    return 0 if ok else 1
