from freehands.fusion import MultimodalFusion, State
from freehands.profiles import Profile


def test_default_profile_uses_right_open_palm_for_pause() -> None:
    profile = Profile(user_id="test")

    assert profile.gesture_bindings["right_open_palm"] == "toggle_pause"
    assert profile.gesture_bindings["fist_pause"] == ""
    assert profile.gesture_thresholds["two_fingers_up"].stability_frames == 3


def test_pointer_click_fires_without_dwell_when_pointer_control_is_enabled() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "pointing_up")

    assert result.fired_action == "click"
    assert result.state is State.COOLDOWN


def test_pointer_button_gestures_map_to_mouse_buttons() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True

    expected_actions = {
        "pointing_up": "click",
        "middle_up": "right_click",
        "two_fingers_up": "double_click",
    }
    for gesture, action in expected_actions.items():
        fusion = MultimodalFusion(profile)
        fusion.sm.activate()

        result = fusion.step((320, 240), gesture)

        assert result.fired_action == action
        assert result.state is State.COOLDOWN


def test_pointer_click_waits_for_confirming_when_pointer_control_is_disabled() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = False
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "pointing_up")

    assert result.fired_action is None
    assert result.state is State.ACTIVE


def test_right_open_palm_is_pause_gesture() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "right_open_palm")

    assert result.fired_action == "toggle_pause"
    assert result.state is State.IDLE


def test_right_open_palm_resumes_from_idle() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)

    result = fusion.step((320, 240), "right_open_palm")

    assert result.fired_action == "resume"
    assert result.state is State.ACTIVE


def test_closed_fist_no_longer_pauses_or_blocks_clicks() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "fist_pause")

    assert result.fired_action is None
    assert result.state is State.ACTIVE