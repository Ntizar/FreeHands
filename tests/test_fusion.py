from freehands.fusion import MultimodalFusion, State, action_for_gesture
from freehands.profiles import Profile


def test_default_profile_uses_right_open_palm_for_pause() -> None:
    profile = Profile(user_id="test")

    assert profile.gesture_bindings["right_open_palm"] == "toggle_pause"
    assert profile.gesture_bindings["left_open_palm"] == "undo"
    assert profile.gesture_bindings["fist_pause"] == ""
    assert profile.gesture_bindings["left_pointing_up"] == ""
    assert profile.gesture_bindings["right_pointing_up"] == ""
    assert profile.gesture_thresholds["two_fingers_up"].stability_frames == 1
    assert profile.gesture_thresholds["left_open_palm"].stability_frames == 10
    assert profile.gesture_thresholds["right_open_palm"].stability_frames == 60

    actions = [action for action in profile.gesture_bindings.values() if action]
    assert len(actions) == len(set(actions))


def test_side_gestures_fallback_to_generic_bindings() -> None:
    profile = Profile(user_id="test")

    assert action_for_gesture(profile.gesture_bindings, "left_pointing_up") == "click"
    assert action_for_gesture(profile.gesture_bindings, "right_pointing_up") == "click"
    assert action_for_gesture(profile.gesture_bindings, "left_two_fingers_up") == "double_click"


def test_pointer_click_fires_without_dwell_when_pointer_control_is_enabled() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "pointing_up")

    assert result.fired_action == "click"
    assert result.state is State.ACTIVE


def test_pointer_button_gestures_map_to_mouse_buttons() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True

    expected_actions = {
        "pointing_up": "click",
        "left_pointing_up": "click",
        "right_pointing_up": "click",
        "middle_up": "right_click",
        "left_middle_up": "right_click",
        "right_middle_up": "right_click",
        "two_fingers_up": "double_click",
        "left_two_fingers_up": "double_click",
        "right_two_fingers_up": "double_click",
    }
    for gesture, action in expected_actions.items():
        fusion = MultimodalFusion(profile)
        fusion.sm.activate()

        result = fusion.step((320, 240), gesture)

        assert result.fired_action == action
        assert result.state is State.ACTIVE


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


def test_toggle_pause_can_be_mapped_to_another_gesture() -> None:
    profile = Profile(user_id="test")
    profile.gesture_bindings["left_open_palm"] = "toggle_pause"
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "left_open_palm")

    assert result.fired_action == "toggle_pause"
    assert result.state is State.IDLE


def test_left_open_palm_fires_undo_when_active() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "left_open_palm")

    assert result.fired_action == "undo"
    assert result.state is State.ACTIVE


def test_closed_fist_no_longer_pauses_or_blocks_clicks() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "fist_pause")

    assert result.fired_action is None
    assert result.state is State.ACTIVE