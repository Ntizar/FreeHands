from freehands.fusion import MultimodalFusion, State, action_for_gesture
from freehands.gaze.blink_detector import BlinkEventType
from freehands.profiles import Profile


def test_default_profile_uses_right_open_palm_for_pause() -> None:
    profile = Profile(user_id="test")

    assert profile.gesture_bindings["right_open_palm"] == "toggle_pause"
    assert profile.gesture_bindings["left_open_palm"] == ""
    assert profile.gesture_bindings["fist_pause"] == ""
    assert profile.gesture_bindings["left_pointing_up"] == ""
    assert profile.gesture_bindings["right_pointing_up"] == ""
    assert profile.gesture_thresholds["two_fingers_up"].stability_frames == 1
    assert profile.gesture_thresholds["left_open_palm"].stability_frames == 60
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


def test_left_open_palm_is_unbound_by_default() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "left_open_palm")

    assert result.fired_action is None
    assert result.state is State.ACTIVE


def test_closed_fist_no_longer_pauses_or_blocks_clicks() -> None:
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "fist_pause")

    assert result.fired_action is None
    assert result.state is State.ACTIVE


def test_repeated_clicks_never_get_throttled_by_side_jitter() -> None:
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    sequence = [
        "right_pointing_up",
        "pointing_up",
        "left_pointing_up",
        "right_pointing_up",
        "pointing_up",
    ]
    fired = [fusion.step((320, 240), gesture).fired_action for gesture in sequence]

    assert fired == ["click"] * len(sequence)


# ── Palm-scroll tests ──────────────────────────────────────────────────────

def test_palm_scroll_up_fires_scroll_up() -> None:
    """Palm-scroll-up gesture should immediately fire scroll_up action."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "palm_scroll_up")

    assert result.fired_action == "scroll_up"
    assert result.state is State.ACTIVE


def test_palm_scroll_down_fires_scroll_down() -> None:
    """Palm-scroll-down gesture should immediately fire scroll_down action."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step((320, 240), "palm_scroll_down")

    assert result.fired_action == "scroll_down"
    assert result.state is State.ACTIVE


def test_palm_scroll_works_in_idle_state() -> None:
    """Palm-scroll should fire even when state is IDLE (no activation needed)."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    # Not calling activate() — stays IDLE

    result = fusion.step((320, 240), "palm_scroll_up")

    assert result.fired_action == "scroll_up"
    assert result.state is State.IDLE


def test_palm_scroll_bypasses_dwell() -> None:
    """Palm-scroll should not require dwell confirmation."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    # With a normal gesture, idle state returns None
    result = fusion.step((320, 240), "pointing_up")
    assert result.fired_action is None

    # But palm-scroll fires immediately
    result = fusion.step((320, 240), "palm_scroll_down")
    assert result.fired_action == "scroll_down"


def test_all_palm_scroll_gestures_map_correctly() -> None:
    """All side-specific palm-scroll gestures should map to correct scroll actions."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    expected = {
        "palm_scroll_up": "scroll_up",
        "palm_scroll_down": "scroll_down",
        "left_palm_scroll_up": "scroll_up",
        "left_palm_scroll_down": "scroll_down",
        "right_palm_scroll_up": "scroll_up",
        "right_palm_scroll_down": "scroll_down",
    }
    for gesture, expected_action in expected.items():
        result = fusion.step((320, 240), gesture)
        assert result.fired_action == expected_action, f"Gesture {gesture} should fire {expected_action}"


def test_palm_scroll_does_not_change_state() -> None:
    """Palm-scroll should not transition the state machine."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    for _ in range(5):
        result = fusion.step((320, 240), "palm_scroll_up")
        assert result.state is State.ACTIVE


def test_palm_scroll_gestures_in_profile_bindings() -> None:
    """Palm-scroll gestures should be present in default profile bindings."""
    profile = Profile(user_id="test")
    assert profile.gesture_bindings["palm_scroll_up"] == "scroll_up"
    assert profile.gesture_bindings["palm_scroll_down"] == "scroll_down"
    assert profile.gesture_bindings["left_palm_scroll_up"] == "scroll_up"
    assert profile.gesture_bindings["right_palm_scroll_down"] == "scroll_down"


def test_single_blink_triggers_click() -> None:
    """A single blink should fire click action."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step(
        (320, 240),
        None,
        blink=True,
        blink_event=BlinkEventType.SINGLE,
    )

    assert result.fired_action == "click"
    assert result.blink is True
    assert result.blink_event is BlinkEventType.SINGLE


def test_double_blink_triggers_click() -> None:
    """A double blink should fire click action with DOUBLE type."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step(
        (320, 240),
        None,
        blink=True,
        blink_event=BlinkEventType.DOUBLE,
    )

    assert result.fired_action == "click"
    assert result.blink is True
    assert result.blink_event is BlinkEventType.DOUBLE


def test_prolonged_blink_triggers_drag_start() -> None:
    """A prolonged close should fire drag_start action."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step(
        (320, 240),
        None,
        blink=True,
        blink_event=BlinkEventType.PROLONGED,
    )

    assert result.fired_action == "drag_start"
    assert result.blink is True
    assert result.blink_event is BlinkEventType.PROLONGED


def test_blink_bypasses_state_machine() -> None:
    """Blink events should work even in IDLE state."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    # Not activating — stays IDLE

    result = fusion.step(
        (320, 240),
        None,
        blink=True,
        blink_event=BlinkEventType.SINGLE,
    )

    assert result.fired_action == "click"


def test_blink_without_event_type_uses_legacy_behavior() -> None:
    """When blink=True but blink_event=None, should fall back to legacy click."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step(
        (320, 240),
        None,
        blink=True,
        blink_event=None,
    )

    assert result.fired_action == "click"
    assert result.blink is True
    assert result.blink_event is None


def test_blink_event_type_is_preserved_in_result() -> None:
    """The blink_event type should be preserved in the FusionResult."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    for event_type in (BlinkEventType.SINGLE, BlinkEventType.DOUBLE, BlinkEventType.PROLONGED):
        result = fusion.step(
            (320, 240),
            None,
            blink=True,
            blink_event=event_type,
        )
        assert result.blink_event is event_type
