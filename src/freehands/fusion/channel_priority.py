"""Dynamic channel prioritisation: decides which modality wins when gaze,
gesture and voice compete for the same action in the same frame.

Priority rules
--------------
1. **System commands** (volume, screenshot, show_desktop) always come from
   voice — they have no gesture counterpart and are safety controls.
2. **Pointer actions** (click, scroll, zoom) come from gesture because the
   gesture carries spatial context (where on screen the user wants to act).
3. **State transitions** (pause, resume) follow the *first* channel to fire
   — whichever arrives first wins (no conflict resolution needed).
4. When both channels propose the *same* action, gesture wins (it is the
   primary input modality for FreeHands).

Each function returns ``(winner_action, winner_source)`` where ``source`` is
``"gesture"``, ``"voice"``, or ``None`` (no action).
"""
from __future__ import annotations

from dataclasses import dataclass

# Actions that are system-level (no spatial context needed).
SYSTEM_ACTIONS: frozenset[str] = frozenset({
    "show_desktop", "screenshot",
    "volume_up", "volume_down", "volume_mute",
})

# Actions that are purely pointer-oriented.
POINTER_ACTIONS: frozenset[str] = frozenset({
    "click", "right_click", "double_click", "undo",
})

# Actions that move content but are gesture-driven.
GESTURE_ACTIONS: frozenset[str] = frozenset({
    "scroll_up", "scroll_down", "zoom_in", "zoom_out",
})

# Actions that control the fusion state machine.
STATE_ACTIONS: frozenset[str] = frozenset({
    "toggle_pause", "resume",
})


@dataclass(frozen=True)
class ChannelDecision:
    """Result of the channel prioritisation logic."""
    action: str | None          # which action to execute (or None)
    source: str | None          # "gesture", "voice", or None
    gesture_confidence: float = 1.0
    voice_confidence: float = 1.0


def decide_channel_priority(
    gesture_action: str | None,
    voice_action: str | None,
    *,
    gesture_confidence: float = 1.0,
    voice_confidence: float = 1.0,
) -> ChannelDecision:
    """Resolve a potential conflict between gesture and voice.

    Rules (applied in order):
    1. If only one channel proposes an action, that one wins.
    2. If both propose actions:
       a. Voice wins for system commands (volume, screenshot, show_desktop).
       b. Gesture wins for pointer/gesture actions (click, scroll, zoom).
       c. If both propose the *same* action, gesture wins (primary modality).
       d. If both propose different non-system actions, gesture wins.
    """
    # ── No conflict: single source ───────────────────────────────────────
    if gesture_action is None and voice_action is None:
        return ChannelDecision(None, None, gesture_confidence, voice_confidence)

    if gesture_action is not None and voice_action is None:
        return ChannelDecision(gesture_action, "gesture", gesture_confidence, voice_confidence)

    if gesture_action is None and voice_action is not None:
        return ChannelDecision(voice_action, "voice", gesture_confidence, voice_confidence)

    # ── Conflict: both channels propose actions ──────────────────────────
    if gesture_action == voice_action:
        # Same action from both channels — gesture wins (primary modality).
        return ChannelDecision(gesture_action, "gesture", gesture_confidence, voice_confidence)

    # Different actions — apply priority rules.
    if voice_action in SYSTEM_ACTIONS:
        # System commands always come from voice.
        return ChannelDecision(voice_action, "voice", gesture_confidence, voice_confidence)

    if gesture_action in SYSTEM_ACTIONS:
        # Voice wins over gesture system commands (shouldn't happen, but safety).
        return ChannelDecision(voice_action, "voice", gesture_confidence, voice_confidence)

    # Both are regular actions — gesture wins (has spatial context).
    return ChannelDecision(gesture_action, "gesture", gesture_confidence, voice_confidence)


def voice_should_bypass_fusion(action: str | None) -> bool:
    """Return True if a voice action should bypass the fusion state machine.

    System commands and state transitions (pause/resume) are handled directly
    by the voice handler in main.py and should not go through the gesture
    fusion pipeline.
    """
    if action is None:
        return False
    return action in SYSTEM_ACTIONS or action in STATE_ACTIONS
