"""CLI entry point: ``python -m freehands ...``"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freehands", description="Hands-free screen control")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Run the aim-trainer calibration minigame")
    p_cal.add_argument("--user", required=True, help="Profile id (e.g. 'luis')")

    p_run = sub.add_parser("run", help="Start the multimodal control system")
    p_run.add_argument("--user", required=True, help="Profile id to load")
    p_run.add_argument("--no-voice", action="store_true", help="Disable voice listener")

    p_check = sub.add_parser("doctor", help="Check camera / mic / dependencies")

    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        from .ui.calibration_game import run_calibration
        return run_calibration(user_id=args.user)
    if args.cmd == "run":
        from .main import run_system
        return run_system(user_id=args.user, voice_enabled=not args.no_voice)
    if args.cmd == "doctor":
        from .doctor import run_doctor
        return run_doctor()
    return 1


if __name__ == "__main__":
    sys.exit(main())
