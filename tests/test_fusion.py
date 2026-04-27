from freehands.fusion import MultimodalFusion, State
from freehands.profiles import Profile


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