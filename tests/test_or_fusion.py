"""Tests for OR-mode fusion and confidence-based intention prioritisation
(improvement #30).

OR mode: any channel can activate an action. If voice proposes a pointer
action and gaze is present (even if unstable), the action fires.

Intention prioritisation: when both channels propose different actions,
the one with higher confidence wins.
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

# Load channel_priority (no deps)
cp_mod = _load_module(
    "freehands.fusion.channel_priority",
    "/root/workspace/FreeHands/src/freehands/fusion/channel_priority.py",
)
sys.modules["freehands.fusion.channel_priority"] = cp_mod

# Load fusion (needs blink_detector, profiles, state_machine — all loaded)
fusion_mod = _load_module(
    "freehands.fusion.fusion",
    "/root/workspace/FreeHands/src/freehands/fusion/fusion.py",
)
sys.modules["freehands.fusion.fusion"] = fusion_mod

# Import what we need
from freehands.fusion.channel_priority import (  # noqa: E402
    FusionMode,
    decide_channel_priority,
    decide_or_fusion,
)
from freehands.fusion.fusion import (  # noqa: E402
    MultimodalFusion,
)
from freehands.fusion.state_machine import State  # noqa: E402
from freehands.profiles import Profile  # noqa: E402


# ══════════════════════════════════════════════════════════════════════
# FusionMode enum
# ══════════════════════════════════════════════════════════════════════

def test_fusion_mode_has_and() -> None:
    """FusionMode should have AND member."""
    assert hasattr(FusionMode, "AND")
    assert FusionMode.AND is not None


def test_fusion_mode_has_or() -> None:
    """FusionMode should have OR member."""
    assert hasattr(FusionMode, "OR")
    assert FusionMode.OR is not None


def test_fusion_mode_and_not_or() -> None:
    """AND and OR should be distinct."""
    assert FusionMode.AND != FusionMode.OR


# ══════════════════════════════════════════════════════════════════════
# decide_channel_priority with mode parameter
# ══════════════════════════════════════════════════════════════════════

def test_and_mode_gesture_wins_different_actions() -> None:
    """In AND mode, gesture wins when actions differ (existing behaviour)."""
    result = decide_channel_priority(
        "click", "zoom_in", mode=FusionMode.AND
    )
    assert result.action == "click"
    assert result.source == "gesture"


def test_or_mode_voice_higher_confidence_wins() -> None:
    """In OR mode, voice with higher confidence wins over gesture."""
    result = decide_channel_priority(
        "click", "zoom_in",
        gesture_confidence=0.5,
        voice_confidence=0.9,
        mode=FusionMode.OR,
    )
    assert result.action == "zoom_in"
    assert result.source == "voice"


def test_or_mode_gesture_higher_confidence_wins() -> None:
    """In OR mode, gesture with higher confidence wins over voice."""
    result = decide_channel_priority(
        "click", "zoom_in",
        gesture_confidence=0.9,
        voice_confidence=0.5,
        mode=FusionMode.OR,
    )
    assert result.action == "click"
    assert result.source == "gesture"


def test_or_mode_equal_confidence_gesture_wins() -> None:
    """In OR mode with equal confidence, gesture wins (default)."""
    result = decide_channel_priority(
        "click", "zoom_in",
        gesture_confidence=0.7,
        voice_confidence=0.7,
        mode=FusionMode.OR,
    )
    assert result.action == "click"
    assert result.source == "gesture"


def test_or_mode_system_command_voice_wins() -> None:
    """System commands still override gesture in OR mode."""
    result = decide_channel_priority(
        "click", "volume_up",
        mode=FusionMode.OR,
    )
    assert result.action == "volume_up"
    assert result.source == "voice"


def test_or_mode_same_action_gesture_wins() -> None:
    """Same action from both: gesture wins in OR mode too."""
    result = decide_channel_priority(
        "click", "click",
        mode=FusionMode.OR,
    )
    assert result.action == "click"
    assert result.source == "gesture"


# ══════════════════════════════════════════════════════════════════════
# decide_or_fusion convenience function
# ══════════════════════════════════════════════════════════════════════

def test_decide_or_fusion_voice_wins_high_confidence() -> None:
    """decide_or_fusion should use OR mode with confidence tie-breaking."""
    result = decide_or_fusion(
        "click", "zoom_in",
        gesture_confidence=0.3,
        voice_confidence=0.95,
    )
    assert result.action == "zoom_in"
    assert result.source == "voice"


def test_decide_or_fusion_gesture_wins_high_confidence() -> None:
    """decide_or_fusion: gesture wins when its confidence is higher."""
    result = decide_or_fusion(
        "click", "zoom_in",
        gesture_confidence=0.95,
        voice_confidence=0.3,
    )
    assert result.action == "click"
    assert result.source == "gesture"


def test_decide_or_fusion_single_channel() -> None:
    """decide_or_fusion with single channel should return that channel."""
    result = decide_or_fusion("scroll_up", None)
    assert result.action == "scroll_up"
    assert result.source == "gesture"

    result = decide_or_fusion(None, "volume_up")
    assert result.action == "volume_up"
    assert result.source == "voice"


# ══════════════════════════════════════════════════════════════════════
# MultimodalFusion with OR mode
# ══════════════════════════════════════════════════════════════════════

def test_multimodal_fusion_default_is_and_mode() -> None:
    """MultimodalFusion should default to AND mode."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    assert fusion.fusion_mode == "AND"


def test_multimodal_fusion_or_mode_configurable() -> None:
    """MultimodalFusion should accept OR mode."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    assert fusion.fusion_mode == "OR"


def test_multimodal_fusion_confidence_defaults() -> None:
    """Confidence values should default to 1.0."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile)
    assert fusion.gesture_confidence == 1.0
    assert fusion.voice_confidence == 1.0


def test_multimodal_fusion_custom_confidence() -> None:
    """Custom confidence values should be stored."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(
        profile,
        gesture_confidence=0.8,
        voice_confidence=0.9,
    )
    assert fusion.gesture_confidence == 0.8
    assert fusion.voice_confidence == 0.9


def test_or_mode_voice_click_with_unstable_gaze_fires() -> None:
    """OR mode: voice click should fire even with unstable gaze."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    fusion.sm.activate()

    # Feed wildly varying positions to prevent stability
    for i in range(10):
        fusion.step((i * 100, i * 50), "pointing_up")

    # In OR mode, unstable gaze is OK — action should fire
    result = fusion.step_and_voice(
        (500, 400),
        None,
        voice_action="click",
    )

    assert result.fired_action == "click"
    assert result.voice_action == "click"
    assert result.gaze_confirmed is True
    assert result.state is State.COOLDOWN


def test_or_mode_voice_zoom_with_unstable_gaze_fires() -> None:
    """OR mode: voice zoom_in should fire even with unstable gaze."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    fusion.sm.activate()

    for i in range(10):
        fusion.step((i * 100, i * 50), "pointing_up")

    result = fusion.step_and_voice(
        (500, 400),
        None,
        voice_action="zoom_in",
    )

    assert result.fired_action == "zoom_in"
    assert result.gaze_confirmed is True


def test_or_mode_voice_scroll_with_unstable_gaze_fires() -> None:
    """OR mode: voice scroll_up should fire even with unstable gaze."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    fusion.sm.activate()

    for i in range(10):
        fusion.step((i * 100, i * 50), "pointing_up")

    result = fusion.step_and_voice(
        (500, 400),
        None,
        voice_action="scroll_up",
    )

    assert result.fired_action == "scroll_up"
    assert result.gaze_confirmed is True


def test_and_mode_still_requires_stable_gaze() -> None:
    """AND mode should still require stable gaze (regression test)."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="AND")
    fusion.sm.activate()

    for i in range(10):
        fusion.step((i * 100, i * 50), "pointing_up")

    # In AND mode, unstable gaze blocks the action
    result = fusion.step_and_voice(
        (500, 400),
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.gaze_confirmed is False


def test_or_mode_no_gaze_still_blocks() -> None:
    """OR mode: without gaze at all, voice should NOT fire pointer actions."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    fusion.sm.activate()

    result = fusion.step_and_voice(
        None,  # no gaze
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.gaze_confirmed is False


def test_or_mode_idle_state_blocks() -> None:
    """OR mode: in IDLE state, voice should NOT fire pointer actions."""
    profile = Profile(user_id="test")
    fusion = MultimodalFusion(profile, fusion_mode="OR")
    # Not activating — stays IDLE

    result = fusion.step_and_voice(
        (320, 240),
        None,
        voice_action="click",
    )

    assert result.fired_action is None
    assert result.gaze_confirmed is False
    assert result.state is State.IDLE


# ══════════════════════════════════════════════════════════════════════
# Intention prioritisation: confidence-based tie-breaking
# ══════════════════════════════════════════════════════════════════════

def test_intention_prioritisation_voice_wins() -> None:
    """Voice with higher confidence should win in OR mode."""
    result = decide_channel_priority(
        "scroll_down", "zoom_in",
        gesture_confidence=0.4,
        voice_confidence=0.85,
        mode=FusionMode.OR,
    )
    assert result.action == "zoom_in"
    assert result.source == "voice"


def test_intention_prioritisation_gesture_wins() -> None:
    """Gesture with higher confidence should win in OR mode."""
    result = decide_channel_priority(
        "scroll_down", "zoom_in",
        gesture_confidence=0.9,
        voice_confidence=0.3,
        mode=FusionMode.OR,
    )
    assert result.action == "scroll_down"
    assert result.source == "gesture"


def test_intention_prioritisation_equal_confidence_gesture_wins() -> None:
    """Equal confidence: gesture wins (spatial context advantage)."""
    result = decide_channel_priority(
        "scroll_down", "zoom_in",
        gesture_confidence=0.5,
        voice_confidence=0.5,
        mode=FusionMode.OR,
    )
    assert result.action == "scroll_down"
    assert result.source == "gesture"


# ══════════════════════════════════════════════════════════════════════
# Backward compatibility: existing AND mode tests still pass
# ══════════════════════════════════════════════════════════════════════

def test_default_mode_is_and() -> None:
    """Calling decide_channel_priority without mode should default to AND."""
    result = decide_channel_priority("click", "zoom_in")
    assert result.action == "click"
    assert result.source == "gesture"
