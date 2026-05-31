"""Dictation intent detection — multimodal activation.

The continuous dictation engine (ContinuousDictationEngine) can be activated
by a voice command alone ("dictar", "FreeHands dictar"). This module adds a
**multimodal** activation path: the user must both **gaze at a text field**
(dwell on a text region detected by the OCR text-region detector) AND say
a trigger phrase ("escribe", "dictar").

This prevents accidental dictation activations and gives the user precise
control over where dictated text will be inserted.

Activation modes
----------------
1. **voice_only** — current behaviour: say "dictar" to start.
2. **gaze_plus_voice** — new: gaze at a text region for 500ms, then say
   "escribe". The dictation engine starts and the overlay shows a
   "dictando: [region]" indicator.

Design decisions
----------------
- Gaze dwell threshold is 500ms (shorter than the 1200ms for text selection
  to make activation snappy).
- The trigger phrase is shared with the voice listener ("escribe", "dictar").
- The overlay shows a pulsing blue ring around the cursor when in
  "gaze-to-dictate" mode.
- Falls back to voice_only if no text regions are detected.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ocr.text_region_detector import TextRegion


# ── Constants ────────────────────────────────────────────────────────────────

GAZE_TO_DICTATE_DWELL_MS: int = 500  # ms of gaze on text region before ready
GAZE_TO_DICTATE_ANIM_STEP: int = 16  # ms per animation frame (~60fps)


@dataclass
class DictationIntentState:
    """Tracks the state of gaze-plus-voice dictation intent."""
    # Whether the user is gazing at a text region
    gazing_at_text: bool = False
    # Which region is currently gazed at
    hovered_region: TextRegion | None = None
    # Dwell progress on the current region (0.0 to 1.0)
    dwell_progress: float = 0.0
    # Whether the user has gazed long enough to be "ready"
    ready_to_activate: bool = False
    # Timestamp when ready was achieved
    ready_at: float = 0.0
    # Last time the gaze left a text region
    last_gaze_lost_at: float = 0.0
    # Whether gaze was lost recently (debounce window)
    gaze_lost_debounce_ms: float = 300.0


class DictationIntentDetector:
    """Detects multimodal dictation intent (gaze + voice).

    Usage:
        detector = DictationIntentDetector()
        detector.update_dwell(cursor_xy, regions)
        if detector.ready_to_activate:
            # User is gazing at a text region — wait for voice trigger
            ...
    """

    def __init__(self, dwell_ms: int = GAZE_TO_DICTATE_DWELL_MS) -> None:
        self._dwell_ms = dwell_ms
        self._state = DictationIntentState()
        self._anim_step = GAZE_TO_DICTATE_ANIM_STEP
        self._dwell_steps = dwell_ms / self._anim_step

    @property
    def state(self) -> DictationIntentState:
        return self._state

    @property
    def gazing_at_text(self) -> bool:
        return self._state.gazing_at_text

    @property
    def ready_to_activate(self) -> bool:
        return self._state.ready_to_activate

    @property
    def hovered_region(self) -> TextRegion | None:
        return self._state.hovered_region

    def update_dwell(
        self,
        cursor_xy: tuple[int, int] | None,
        text_regions: list[TextRegion],
        centre_x: int,
        centre_y: int,
    ) -> None:
        """Update gaze state with current cursor position and text regions.

        Call this each frame with the current cursor position. The detector
        will update dwell progress on any text region the cursor is over.
        """
        now = time.monotonic()

        if cursor_xy is None:
            # No cursor — cancel everything
            self._state.gazing_at_text = False
            self._state.hovered_region = None
            self._state.dwell_progress = 0.0
            self._state.ready_to_activate = False
            return

        # Hit-test: find which text region the cursor is in
        hovered = self._hit_test(cursor_xy, centre_x, centre_y, text_regions)

        if hovered is not None:
            self._state.gazing_at_text = True
            self._state.hovered_region = hovered
            # Reset debounce when re-entering a region
            self._state.last_gaze_lost_at = 0.0

            # Update dwell progress
            self._state.dwell_progress = min(
                1.0,
                self._state.dwell_progress + 1.0 / self._dwell_steps,
            )

            if self._state.dwell_progress >= 1.0 and not self._state.ready_to_activate:
                self._state.ready_to_activate = True
                self._state.ready_at = now
        else:
            # Cursor left the text region area
            if self._state.gazing_at_text:
                # Just lost gaze — start debounce
                if self._state.ready_to_activate:
                    # Was ready, now lost gaze — debounce before resetting
                    elapsed = (now - self._state.last_gaze_lost_at) * 1000
                    if elapsed > self._state.gaze_lost_debounce_ms:
                        self._state.ready_to_activate = False
                        self._state.dwell_progress = 0.0
                self._state.last_gaze_lost_at = now

            self._state.gazing_at_text = False
            self._state.hovered_region = None
            self._state.dwell_progress = 0.0

    def reset(self) -> None:
        """Reset all state. Call when dictation starts or is cancelled."""
        self._state = DictationIntentState()

    def consume_ready(self) -> bool:
        """Consume the ready state — returns True if was ready, then resets.

        Call this when the voice trigger is detected while gazing at text.
        Returns True only if both gaze AND voice conditions were met.
        """
        if not self._state.ready_to_activate:
            return False
        self._state.ready_to_activate = False
        self._state.dwell_progress = 0.0
        return True

    # ── internals ────────────────────────────────────────────────────────

    def _hit_test(
        self,
        cursor_xy: tuple[int, int],
        cx: int,
        cy: int,
        regions: list[TextRegion],
    ) -> TextRegion | None:
        """Determine which text region the cursor is in, or None."""
        for region in regions:
            screen_x = cx - 400 + region.x  # TEXT_SELECTOR_WIDTH // 2 = 400
            screen_y = cy - 300 + region.y  # TEXT_SELECTOR_HEIGHT // 2 = 300
            if (screen_x <= cursor_xy[0] < screen_x + region.width and
                    screen_y <= cursor_xy[1] < screen_y + region.height):
                return region
        return None
