from freehands.main import PauseHoldGate, suppress_open_palm_toggle_without_hold
from freehands.profiles import GestureThreshold, Profile


def test_pause_hold_ignores_unbound_left_palm_by_default() -> None:
    profile = Profile(user_id="test")
    gate = PauseHoldGate()

    fired = [
        gate.update("left_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0]
        for _ in range(profile.gesture_thresholds["right_open_palm"].stability_frames + 2)
    ]

    assert fired == [None] * len(fired)


def test_pause_hold_uses_custom_left_palm_toggle() -> None:
    profile = Profile(user_id="test")
    profile.gesture_bindings["right_open_palm"] = ""
    profile.gesture_bindings["left_open_palm"] = "toggle_pause"
    profile.gesture_thresholds["left_open_palm"] = GestureThreshold(
        stability_frames=3,
        confidence_min=0.80,
    )
    gate = PauseHoldGate()

    fired = [
        gate.update("left_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0]
        for _ in range(3)
    ]

    assert fired == [None, None, "left_open_palm"]


def test_pause_hold_fires_once_until_released() -> None:
    profile = Profile(user_id="test")
    profile.gesture_thresholds["right_open_palm"] = GestureThreshold(
        stability_frames=3,
        confidence_min=0.80,
    )
    gate = PauseHoldGate()

    fired = [
        gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0]
        for _ in range(8)
    ]

    assert fired == [None, None, "right_open_palm", None, None, None, None, None]

    for _ in range(3):
        assert gate.update("none", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] == "right_open_palm"


def test_pause_hold_repairs_progress_after_short_detection_gap() -> None:
    profile = Profile(user_id="test")
    profile.gesture_thresholds["right_open_palm"] = GestureThreshold(
        stability_frames=4,
        confidence_min=0.80,
    )
    gate = PauseHoldGate()

    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("none", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None
    assert gate.update("right_open_palm", profile.gesture_bindings, profile.gesture_thresholds)[0] is None

    confirmed, _progress = gate.update(
        "right_open_palm",
        profile.gesture_bindings,
        profile.gesture_thresholds,
    )

    assert confirmed == "right_open_palm"


def test_open_palm_toggle_from_stabilizer_waits_for_hold_gate() -> None:
    profile = Profile(user_id="test")

    confirmed = suppress_open_palm_toggle_without_hold(
        "right_open_palm",
        profile.gesture_bindings,
    )

    assert confirmed is None


def test_non_pause_open_palm_action_is_not_suppressed() -> None:
    profile = Profile(user_id="test")
    profile.gesture_bindings["right_open_palm"] = "undo"

    confirmed = suppress_open_palm_toggle_without_hold(
        "right_open_palm",
        profile.gesture_bindings,
    )

    assert confirmed == "right_open_palm"
