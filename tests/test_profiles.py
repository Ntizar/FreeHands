from freehands.profiles import Profile
from freehands.profiles.store import load_profile


INSTANT_MOUSE_GESTURES = (
    "pointing_up",
    "middle_up",
    "two_fingers_up",
    "left_pointing_up",
    "right_pointing_up",
    "left_middle_up",
    "right_middle_up",
    "left_two_fingers_up",
    "right_two_fingers_up",
)


def _write_profile(tmp_path, profile: Profile) -> None:
    (tmp_path / f"{profile.user_id}.json").write_text(
        profile.model_dump_json(),
        encoding="utf-8",
    )


def test_load_profile_repairs_click_latency_and_safety_bindings(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="broken")
    profile.gesture_bindings.update(
        {
            "middle_up": "toggle_pause",
            "right_open_palm": "",
            "tongue_out": "right_click",
            "pointing_up": "click",
            "two_fingers_up": "",
        }
    )
    for gesture in INSTANT_MOUSE_GESTURES:
        profile.gesture_thresholds[gesture].stability_frames = 20
        profile.gesture_thresholds[gesture].confidence_min = 0.85
    _write_profile(tmp_path, profile)

    repaired = load_profile("broken")

    for gesture in INSTANT_MOUSE_GESTURES:
        threshold = repaired.gesture_thresholds[gesture]
        assert threshold.stability_frames == 1
        assert threshold.confidence_min == 0.50
    assert repaired.gesture_bindings["right_open_palm"] == "toggle_pause"
    assert repaired.gesture_bindings["middle_up"] == "right_click"
    assert repaired.gesture_bindings["tongue_out"] == ""
    assert repaired.gesture_bindings["pointing_up"] == "click"
    assert repaired.gesture_bindings["two_fingers_up"] == "double_click"


def test_load_profile_preserves_non_mouse_pause_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="custom")
    profile.gesture_bindings["right_open_palm"] = ""
    profile.gesture_bindings["left_open_palm"] = "toggle_pause"
    _write_profile(tmp_path, profile)

    repaired = load_profile("custom")

    assert repaired.gesture_bindings["left_open_palm"] == "toggle_pause"
    assert repaired.gesture_bindings["right_open_palm"] == ""


def test_load_profile_preserves_partial_custom_left_palm_pause(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="partial-custom")
    data = profile.model_dump()
    data["gesture_bindings"]["left_open_palm"] = "toggle_pause"
    del data["gesture_bindings"]["right_open_palm"]
    (tmp_path / "partial-custom.json").write_text(
        profile.__class__.model_validate(data).model_dump_json(),
        encoding="utf-8",
    )

    repaired = load_profile("partial-custom")

    assert repaired.gesture_bindings["left_open_palm"] == "toggle_pause"
    assert repaired.gesture_bindings["right_open_palm"] == ""


def test_load_profile_keeps_low_latency_custom_threshold(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="custom-threshold")
    profile.gesture_thresholds["pointing_up"].stability_frames = 2
    profile.gesture_thresholds["pointing_up"].confidence_min = 0.65
    _write_profile(tmp_path, profile)

    repaired = load_profile("custom-threshold")

    threshold = repaired.gesture_thresholds["pointing_up"]
    assert threshold.stability_frames == 2
    assert threshold.confidence_min == 0.65


def test_load_profile_moves_side_click_binding_to_generic_for_jitter(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="side-click")
    profile.gesture_bindings["pointing_up"] = ""
    profile.gesture_bindings["right_pointing_up"] = "click"
    profile.gesture_bindings["middle_up"] = ""
    profile.gesture_bindings["left_middle_up"] = "right_click"
    profile.gesture_bindings["two_fingers_up"] = ""
    profile.gesture_bindings["right_two_fingers_up"] = "double_click"
    _write_profile(tmp_path, profile)

    repaired = load_profile("side-click")

    assert repaired.gesture_bindings["pointing_up"] == "click"
    assert repaired.gesture_bindings["right_pointing_up"] == ""
    assert repaired.gesture_bindings["middle_up"] == "right_click"
    assert repaired.gesture_bindings["left_middle_up"] == ""
    assert repaired.gesture_bindings["two_fingers_up"] == "double_click"
    assert repaired.gesture_bindings["right_two_fingers_up"] == ""


def test_load_profile_preserves_custom_non_family_click_binding(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("freehands.profiles.store.PROFILES_DIR", tmp_path)
    profile = Profile(user_id="custom-click")
    profile.gesture_bindings["pointing_up"] = ""
    profile.gesture_bindings["thumb_up"] = "click"
    profile.gesture_bindings["middle_up"] = ""
    profile.gesture_bindings["thumb_down"] = "right_click"
    _write_profile(tmp_path, profile)

    repaired = load_profile("custom-click")

    assert repaired.gesture_bindings["thumb_up"] == "click"
    assert repaired.gesture_bindings["pointing_up"] == ""
    assert repaired.gesture_bindings["thumb_down"] == "right_click"
    assert repaired.gesture_bindings["middle_up"] == ""
