"""Tests for external gesture profile loading and merging."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from freehands.profiles import (
    GestureProfileOverride,
    GestureThreshold,
    Profile,
    gesture_profiles_dir,
    list_gesture_profiles,
    load_gesture_profile,
    merge_gesture_profile,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_gesture_profile_dir(tmp_path: Path) -> Path:
    """Create a gesture_profiles directory under tmp_path and return it."""
    gp_dir = tmp_path / "gesture_profiles"
    gp_dir.mkdir(parents=True, exist_ok=True)
    return gp_dir


def _write_gesture_profile(gp_dir: Path, name: str, data: dict) -> Path:
    path = gp_dir / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ── load_gesture_profile ────────────────────────────────────────────────────


def test_load_gesture_profile_bindings_only(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-bindings", {
        "gesture_bindings": {
            "pointing_up": "custom_click",
            "thumb_up": "reload",
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = load_gesture_profile("test-bindings")

    assert profile.gesture_bindings == {
        "pointing_up": "custom_click",
        "thumb_up": "reload",
    }
    assert profile.gesture_thresholds == {}


def test_load_gesture_profile_thresholds_only(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-thresholds", {
        "gesture_thresholds": {
            "pointing_up": {"stability_frames": 5, "confidence_min": 0.65},
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = load_gesture_profile("test-thresholds")

    assert profile.gesture_bindings == {}
    assert profile.gesture_thresholds == {
        "pointing_up": {"stability_frames": 5, "confidence_min": 0.65},
    }


def test_load_gesture_profile_both(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-both", {
        "gesture_bindings": {"pointing_up": "click"},
        "gesture_thresholds": {
            "pointing_up": {"stability_frames": 2, "confidence_min": 0.50},
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = load_gesture_profile("test-both")

    assert profile.gesture_bindings == {"pointing_up": "click"}
    assert profile.gesture_thresholds == {
        "pointing_up": {"stability_frames": 2, "confidence_min": 0.50},
    }


def test_load_gesture_profile_minimal(tmp_path: Path) -> None:
    """Empty profile is valid — just means use all defaults."""
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-empty", {})

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = load_gesture_profile("test-empty")

    assert profile.gesture_bindings == {}
    assert profile.gesture_thresholds == {}


def test_load_gesture_profile_not_found(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        with pytest.raises(FileNotFoundError, match="Gesture profile 'nonexistent' not found"):
            load_gesture_profile("nonexistent")


# ── list_gesture_profiles ──────────────────────────────────────────────────


def test_list_gesture_profiles_empty(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        names = list_gesture_profiles()
    assert names == []


def test_list_gesture_profiles_sorted(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "zebra", {})
    _write_gesture_profile(gp_dir, "alpha", {})
    _write_gesture_profile(gp_dir, "middle", {})

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        names = list_gesture_profiles()

    assert names == ["alpha", "middle", "zebra"]


# ── merge_gesture_profile ──────────────────────────────────────────────────


def test_merge_gesture_profile_overrides_bindings(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-merge", {
        "gesture_bindings": {
            "pointing_up": "custom_click",
            "thumb_up": "reload",
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = Profile(user_id="test-merge-user")
        merge_gesture_profile(profile, "test-merge")

    assert profile.gesture_bindings["pointing_up"] == "custom_click"
    assert profile.gesture_bindings["thumb_up"] == "reload"
    # Default bindings preserved for non-overridden gestures
    assert profile.gesture_bindings["middle_up"] == "right_click"
    assert profile.gesture_bindings["two_fingers_up"] == "double_click"


def test_merge_gesture_profile_overrides_thresholds(tmp_path: Path) -> None:
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-thresh-merge", {
        "gesture_thresholds": {
            "pointing_up": {"stability_frames": 3, "confidence_min": 0.60},
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = Profile(user_id="test-thresh-user")
        merge_gesture_profile(profile, "test-thresh-merge")

    threshold = profile.gesture_thresholds["pointing_up"]
    assert threshold.stability_frames == 3
    assert threshold.confidence_min == 0.60


def test_merge_gesture_profile_creates_new_gesture_threshold(tmp_path: Path) -> None:
    """External profile can add thresholds for gestures not in defaults."""
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-new-gesture", {
        "gesture_thresholds": {
            "my_custom_gesture": {"stability_frames": 10, "confidence_min": 0.90},
        },
        "gesture_bindings": {
            "my_custom_gesture": "custom_action",
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = Profile(user_id="test-new-user")
        merge_gesture_profile(profile, "test-new-gesture")

    assert "my_custom_gesture" in profile.gesture_thresholds
    assert profile.gesture_thresholds["my_custom_gesture"].stability_frames == 10
    assert profile.gesture_thresholds["my_custom_gesture"].confidence_min == 0.90
    assert profile.gesture_bindings["my_custom_gesture"] == "custom_action"


def test_merge_gesture_profile_preserves_invariants(tmp_path: Path) -> None:
    """Merge must not break essential bindings or instant mouse thresholds."""
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-inv", {
        "gesture_bindings": {
            "pointing_up": "click",
            "middle_up": "right_click",
            "two_fingers_up": "double_click",
        },
        "gesture_thresholds": {
            "pointing_up": {"stability_frames": 1, "confidence_min": 0.40},
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = Profile(user_id="test-inv-user")
        merge_gesture_profile(profile, "test-inv")

    # Essential bindings preserved
    assert profile.gesture_bindings["pointing_up"] == "click"
    assert profile.gesture_bindings["middle_up"] == "right_click"
    assert profile.gesture_bindings["two_fingers_up"] == "double_click"
    # toggle_pause still assigned
    assert profile.gesture_bindings["right_open_palm"] == "toggle_pause"


def test_merge_gesture_profile_partial_override(tmp_path: Path) -> None:
    """Only specifying bindings, not thresholds — thresholds stay at defaults."""
    gp_dir = _create_gesture_profile_dir(tmp_path)
    _write_gesture_profile(gp_dir, "test-partial", {
        "gesture_bindings": {
            "thumb_up": "reload",
        }
    })

    with patch("freehands.profiles.store.GESTURE_PROFILES_DIR", gp_dir):
        profile = Profile(user_id="test-partial-user")
        merge_gesture_profile(profile, "test-partial")

    assert profile.gesture_bindings["thumb_up"] == "reload"
    # Default thresholds preserved
    assert profile.gesture_thresholds["pointing_up"].stability_frames == 1
    assert profile.gesture_thresholds["pointing_up"].confidence_min == 0.50
