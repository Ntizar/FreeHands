"""Tests for multimodal AND fusion (improvement #14).

AND fusion: voice actions for pointer/gesture operations (click, zoom,
scroll, …) only fire when gaze is also present and stable — requiring
the user to look at the target.
"""
from __future__ import annotations

import sys
import importlib.util

# ── Bypass gaze __init__.py (sklearn dependency) ──────────────────────
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load blink_detector first (no external deps)
blink_mod = _load_module(
    "freehands.gaze.blink_detector",
    "/root/workspace/FreeHands/src/freehands/gaze/blink_detector.py",
)

# Load config (no heavy deps)
config_mod = _load_module(
    "freehands.config",
    "/root/workspace/FreeHands/src/freehands/config.py",
)
sys.modules["freehands.config"] = config_mod

# Load profiles store (needs config, already loaded)
profiles_store_mod = _load_module(
    "freehands.profiles.store",
    "/root/workspace/FreeHands/src/freehands/profiles/store.py",
)
sys.modules["freehands.profiles.store"] = profiles_store_mod

# Load profiles __init__
profiles_mod = _load_module(
    "freehands.profiles",
    "/root/workspace/FreeHands/src/freehands/profiles/__init__.py",
)
sys.modules["freehands.profiles"] = profiles_mod

# Load state_machine (no deps)
sm_mod = _load_module(
    "freehands.fusion.state_machine",
    "/root/workspace/FreeHands/src/freehands/fusion/state_machine.py",
)
sys.modules["freehands.fusion.state_machine"] = sm_mod

# Load fusion (needs blink_detector, profiles, state_machine — all loaded)
fusion_mod = _load_module(
    "freehands.fusion.fusion",
    "/root/workspace/FreeHands/src/freehands/fusion/fusion.py",
)
sys.modules["freehands.fusion.fusion"] = fusion_mod

# Import what we need
from freehands.fusion.fusion import (
    AND_FUSION_ACTIONS,
    FusionResult,
    MultimodalFusion,
)
from freehands.fusion.state_machine import State
from freehands.gaze.blink_detector import BlinkEventType
from freehands.profiles import Profile


# ── AND_FUSION_ACTIONS constants ──────────────────────────────────────

def test_and_fusion_actions_contains_pointer_actions() -> None:
    """AND_FUSION_ACTIONS should include click, zoom, scroll."""
    assert "click" in AND_FUSION_ACTIONS
    assert "right_click" in AND_FUSION_ACTIONS
    assert "double_click" in AND_FUSION_ACTIONS
    assert "zoom_in" in AND_FUSION_ACTIONS
    assert "zoom_out" in AND_FUSION_ACTIONS
    assert "scroll_up" in AND_FUSION_ACTIONS
    assert "scroll_down" in AND_FUSION_ACTIONS
    assert "undo" in AND_FUSION_ACTIONS


def test_and_fusion_actions_excludes_system_commands() -> None:
    """AND_FUSION_ACTIONS should NOT include system commands."""
    assert "volume_up" not in AND_FUSION_ACTIONS
    assert "volume_down" not in AND_FUSION_ACTIONS
    assert "volume_mute" not in AND_FUSION_ACTIONS
    assert "screenshot" not in AND_FUSION_ACTIONS
    assert "show_desktop" not in AND_FUSION_ACTIONS


def test_and_fusion_actions_excludes_state_actions() -> None:
    """AND_FUSION_ACTIONS should NOT include state transitions."""
    assert "toggle_pause" not in AND_FUSION_ACTIONS
    assert "resume" not in AND_FUSION_ACTIONS


# ── FusionResult fields ──────────────────────────────────────────────

def test_fusion_result_has_voice_action_field() -> None:
    """FusionResult should have voice_action field."""
    result = FusionResult(
        cursor_xy=(320, 240),
        state=State.ACTIVE,
        dwell_progress=0.5,
        fired_action="click",
        blink=False,
        blink_event=None,
        voice_action="click",
        gaze_confirmed=True,
    )
    assert result.voice_action == "click"
    assert result.gaze_confirmed is True


def test_fusion_result_defaults_voice_fields() -> None:
    """voice_action should default to None, gaze_confirmed to False."""
    result = FusionResult(
        cursor_xy=(320, 240),
        state=State.ACTIVE,
        dwell_progress=0.0,
        fired_action=None,
    )
    assert result.voice_action is None
    assert result.gaze_confirmed is False


# ── step_and_voice: voice action with gaze present and stable ─────────

def test_voice_click_with_stable_gaze_fires() -> None:
    """Voice click should fire when gaze is present and stable."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    # First, build up the stability buffer by calling step without voice
    for _ in range(8):
        fusion.step((320, 240), "pointing_up")

    # Now with voice action + stable gaze
    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action="click",
    )

    assert result.fired_action == "click"
    assert result.voice_action == "click"
    assert result.gaze_confirmed is True
    assert result.state is State.COOLDOWN


def test_voice_zoom_in_with_stable_gaze_fires() -> None:
    """Voice zoom_in should fire when gaze is present and stable."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    for _ in range(8):
        fusion.step((320, 240), "pointing_up")

    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action="zoom_in",
    )

    assert result.fired_action == "zoom_in"
    assert result.voice_action == "zoom_in"
    assert result.gaze_confirmed is True


def test_voice_scroll_up_with_stable_gaze_fires() -> None:
    """Voice scroll_up should fire when gaze is present and stable."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    for _ in range(8):
        fusion.step((320, 240), "pointing_up")

    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action="scroll_up",
    )

    assert result.fired_action == "scroll_up"
    assert result.voice_action == "scroll_up"
    assert result.gaze_confirmed is True


# ── step_and_voice: voice action WITHOUT gaze ─────────────────────────

def test_voice_click_without_gaze_does_not_fire() -> None:
    """Voice click should NOT fire when gaze is absent (cursor=None)."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        None,  # no gaze
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.voice_action == "click"
    assert result.gaze_confirmed is False


def test_voice_zoom_without_gaze_does_not_fire() -> None:
    """Voice zoom should NOT fire when gaze is absent."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        None,
        None,
        voice_action="zoom_in",
    )

    assert result.fired_action is None
    assert result.gaze_confirmed is False


# ── step_and_voice: voice action in IDLE state ────────────────────────

def test_voice_click_in_idle_state_does_not_fire() -> None:
    """Voice click should NOT fire when state is IDLE."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    # Not activating — stays IDLE

    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.gaze_confirmed is False
    assert result.state is State.IDLE


# ── step_and_voice: gesture action takes precedence ───────────────────

def test_gesture_click_prevents_voice_click() -> None:
    """If gesture already fires click, voice click should not override."""
    profile = Profile(user_id="test")
    profile.pointer_control_enabled = True
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        (320, 240),
        "pointing_up",  # gesture already fires click
        voice_action="click",
    )

    assert result.fired_action == "click"
    assert result.voice_action == "click"
    assert result.gaze_confirmed is True


# ── step_and_voice: blink events bypass AND fusion ────────────────────

def test_blink_click_bypasses_and_fusion() -> None:
    """Blink click should fire even without gaze stability (AND fusion bypass)."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action=None,
        blink=True,
        blink_event=BlinkEventType.SINGLE,
    )

    assert result.fired_action == "click"
    assert result.blink is True
    assert result.blink_event is BlinkEventType.SINGLE


# ── step_and_voice: voice action not in AND_FUSION_ACTIONS fires directly ─

def test_non_pointer_voice_action_fires_without_gaze() -> None:
    """Voice actions NOT in AND_FUSION_ACTIONS (e.g. drag_start) should fire directly."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        None,  # no gaze
        None,
        voice_action="drag_start",
    )

    assert result.fired_action == "drag_start"
    assert result.gaze_confirmed is True


# ── step_and_voice: palm scroll bypasses AND fusion ───────────────────

def test_palm_scroll_bypasses_and_fusion() -> None:
    """Palm scroll gestures should fire immediately, ignoring voice AND fusion."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        (320, 240),
        "palm_scroll_up",
        voice_action="click",
    )

    # Palm scroll fires immediately, voice click is ignored
    assert result.fired_action == "scroll_up"
    assert result.state is State.ACTIVE


# ── step_and_voice: toggle_pause bypasses AND fusion ──────────────────

def test_toggle_pause_bypasses_and_fusion() -> None:
    """toggle_pause gesture should work regardless of voice commands."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    result = fusion.step_and_voice(
        (320, 240),
        "right_open_palm",
        voice_action="click",
    )

    assert result.fired_action == "toggle_pause"
    assert result.state is State.IDLE


# ── Integration: gaze unstable blocks voice AND fusion ────────────────

def test_voice_action_blocked_by_unstable_gaze() -> None:
    """When gaze is present but unstable (high variance), AND fusion should block."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    fusion.sm.activate()

    # Feed wildly varying positions to prevent stability
    for i in range(10):
        fusion.step((i * 100, i * 50), "pointing_up")

    # Gaze is present but unstable — AND fusion should block
    result = fusion.step_and_voice(
        (500, 400),
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.voice_action == "click"
    assert result.gaze_confirmed is False
