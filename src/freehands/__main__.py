"""CLI entry point: ``python -m freehands ...``"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freehands", description="Hands-free screen control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Run the aim-trainer calibration minigame")
    p_cal.add_argument("--user", required=True, help="Profile id (e.g. 'Ntizar')")

    p_cal_gaze = sub.add_parser("calibrate-gaze", help="Recalibrate only gaze for a profile")
    p_cal_gaze.add_argument("--user", required=True, help="Profile id (e.g. 'Ntizar')")

    p_cal_gestures = sub.add_parser("calibrate-gestures", help="Recalibrate only hand gestures")
    p_cal_gestures.add_argument("--user", required=True, help="Profile id (e.g. 'Ntizar')")

    p_camera = sub.add_parser("camera", help="Select and save the preferred camera")
    p_camera.add_argument("--user", required=True, help="Profile id (e.g. 'Ntizar')")

    p_run = sub.add_parser("run", help="Start the multimodal control system")
    p_run.add_argument("--user", required=True, help="Profile id to load")
    p_run.add_argument("--no-voice", action="store_true", help="Disable voice listener")

    p_check = sub.add_parser("doctor", help="Check camera / mic / dependencies")
    p_repair = sub.add_parser("repair", help="Install/reinstall runtime dependencies")

    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        from .ui.calibration_game import run_calibration
        return run_calibration(user_id=args.user)
    if args.cmd == "calibrate-gaze":
        from .ui.calibration_game import run_gaze_calibration
        return run_gaze_calibration(user_id=args.user)
    if args.cmd == "calibrate-gestures":
        from .ui.calibration_game import run_gesture_calibration
        return run_gesture_calibration(user_id=args.user)
    if args.cmd == "camera":
        from .ui.camera_selector import run_camera_selector
        return run_camera_selector(user_id=args.user)
    if args.cmd == "run":
        from .main import run_system
        return run_system(user_id=args.user, voice_enabled=not args.no_voice)
    if args.cmd == "doctor":
        from .doctor import run_doctor
        return run_doctor()
    if args.cmd == "repair":
        from .doctor import repair_dependencies
        return repair_dependencies()
    return 1


if __name__ == "__main__":
    sys.exit(main())
