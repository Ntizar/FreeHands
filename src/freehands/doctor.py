"""Diagnostics: verify camera, microphone, and key dependencies."""
from __future__ import annotations


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

    # MediaPipe
    try:
        import mediapipe  # noqa: F401
        print("  [OK] MediaPipe importable")
    except Exception as e:
        print(f"  [ERR] MediaPipe missing: {e}"); ok = False

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
