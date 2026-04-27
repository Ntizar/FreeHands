from freehands.ui.calibration_game import GESTURE_ALIASES


def test_right_open_palm_calibration_requires_right_side() -> None:
    assert GESTURE_ALIASES["right_open_palm"] == {"right_open_palm"}
