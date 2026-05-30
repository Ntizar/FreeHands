"""Tests for dynamic channel prioritisation (improvement #8)."""
from __future__ import annotations

from freehands.fusion import (
    ChannelDecision,
    decide_channel_priority,
    voice_should_bypass_fusion,
)


# ── No conflict tests ──────────────────────────────────────────────────────

def test_both_none_returns_none() -> None:
    """When neither channel proposes an action, result is None."""
    result = decide_channel_priority(None, None)
    assert result.action is None
    assert result.source is None


def test_gesture_only_returns_gesture() -> None:
    """When only gesture proposes an action, gesture wins."""
    result = decide_channel_priority("click", None)
    assert result.action == "click"
    assert result.source == "gesture"


def test_voice_only_returns_voice() -> None:
    """When only voice proposes an action, voice wins."""
    result = decide_channel_priority(None, "zoom_in")
    assert result.action == "zoom_in"
    assert result.source == "voice"


# ── Same action from both channels ─────────────────────────────────────────

def test_same_action_gesture_wins() -> None:
    """When both channels propose the same action, gesture wins (primary modality)."""
    result = decide_channel_priority("click", "click")
    assert result.action == "click"
    assert result.source == "gesture"


def test_same_voice_action_gesture_wins() -> None:
    """When both channels propose the same voice action, gesture wins."""
    result = decide_channel_priority("scroll_up", "scroll_up")
    assert result.action == "scroll_up"
    assert result.source == "gesture"


# ── System commands: voice always wins ─────────────────────────────────────

def test_voice_volume_overrides_gesture_click() -> None:
    """Voice system command (volume) should override gesture action (click)."""
    result = decide_channel_priority("click", "volume_up")
    assert result.action == "volume_up"
    assert result.source == "voice"


def test_voice_screenshot_overrides_gesture_zoom() -> None:
    """Voice system command (screenshot) should override gesture action (zoom)."""
    result = decide_channel_priority("zoom_in", "screenshot")
    assert result.action == "screenshot"
    assert result.source == "voice"


def test_voice_show_desktop_overrides_gesture_scroll() -> None:
    """Voice system command (show_desktop) should override gesture action (scroll)."""
    result = decide_channel_priority("scroll_up", "show_desktop")
    assert result.action == "show_desktop"
    assert result.source == "voice"


def test_voice_volume_down_overrides_gesture() -> None:
    """Voice volume_down should override any gesture."""
    result = decide_channel_priority("double_click", "volume_down")
    assert result.action == "volume_down"
    assert result.source == "voice"


def test_voice_volume_mute_overrides_gesture() -> None:
    """Voice volume_mute should override any gesture."""
    result = decide_channel_priority("pointing_up", "volume_mute")
    assert result.action == "volume_mute"
    assert result.source == "voice"


# ── Regular actions: gesture wins ──────────────────────────────────────────

def test_gesture_click_overrides_voice_zoom() -> None:
    """When both propose regular (non-system) actions, gesture wins."""
    result = decide_channel_priority("click", "zoom_in")
    assert result.action == "click"
    assert result.source == "gesture"


def test_gesture_scroll_overrides_voice_scroll() -> None:
    """Gesture scroll should win over voice scroll (same action already handled above)."""
    result = decide_channel_priority("scroll_up", "scroll_down")
    assert result.action == "scroll_up"
    assert result.source == "gesture"


def test_gesture_zoom_in_overrides_voice_zoom_out() -> None:
    """Gesture zoom_in should win over voice zoom_out."""
    result = decide_channel_priority("zoom_in", "zoom_out")
    assert result.action == "zoom_in"
    assert result.source == "gesture"


def test_gesture_right_click_overrides_voice_click() -> None:
    """Gesture right_click should win over voice click."""
    result = decide_channel_priority("right_click", "click")
    assert result.action == "right_click"
    assert result.source == "gesture"


# ── Confidence passthrough ─────────────────────────────────────────────────

def test_confidence_values_passed_through() -> None:
    """Confidence values should be passed through in the result."""
    result = decide_channel_priority(
        "click", None,
        gesture_confidence=0.85,
        voice_confidence=0.95,
    )
    assert result.gesture_confidence == 0.85
    assert result.voice_confidence == 0.95


def test_default_confidence_is_1_0() -> None:
    """Default confidence should be 1.0 when not specified."""
    result = decide_channel_priority("click", None)
    assert result.gesture_confidence == 1.0
    assert result.voice_confidence == 1.0


# ── voice_should_bypass_fusion tests ───────────────────────────────────────

def test_system_commands_bypass_fusion() -> None:
    """System commands should bypass the fusion state machine."""
    assert voice_should_bypass_fusion("volume_up") is True
    assert voice_should_bypass_fusion("volume_down") is True
    assert voice_should_bypass_fusion("volume_mute") is True
    assert voice_should_bypass_fusion("screenshot") is True
    assert voice_should_bypass_fusion("show_desktop") is True


def test_state_transitions_bypass_fusion() -> None:
    """State transitions should bypass the fusion state machine."""
    assert voice_should_bypass_fusion("toggle_pause") is True
    assert voice_should_bypass_fusion("resume") is True


def test_pointer_actions_dont_bypass_fusion() -> None:
    """Pointer actions should NOT bypass the fusion state machine."""
    assert voice_should_bypass_fusion("click") is False
    assert voice_should_bypass_fusion("scroll_up") is False
    assert voice_should_bypass_fusion("zoom_in") is False
    assert voice_should_bypass_fusion("double_click") is False


def test_none_returns_false() -> None:
    """None action should return False."""
    assert voice_should_bypass_fusion(None) is False
