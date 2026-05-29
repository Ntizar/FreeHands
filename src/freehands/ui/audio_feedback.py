"""Minimal audio feedback for gesture and voice confirmation.

Uses the OS-native ``winsound.Beep`` on Windows (freehands runs on Windows).
Falls back to a harmless no-op on other platforms.

Design decisions
----------------
- Two tones: short-high for gestures, short-low for voice.
- Non-blocking: plays on the main thread but ``winsound`` is synchronous
  and very fast (~50 ms), so the UI tick loop is barely affected.
- Configurable via profile (enabled/disabled, volume).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AudioFeedback:
    """Plays short confirmation beeps on gesture/voice events."""

    enabled: bool = True
    _last_beep_at: float = field(default=0.0, repr=False)
    _min_interval_ms: int = 80  # debounce: max one beep every 80 ms

    # ── public API ────────────────────────────────────────────────────────

    def play_gesture_confirmation(self) -> None:
        """Short high beep — gesture confirmed (click, scroll, zoom …)."""
        if not self.enabled:
            return
        self._beep(1200, 40)

    def play_voice_confirmation(self) -> None:
        """Short low beep — voice command received and executed."""
        if not self.enabled:
            return
        self._beep(800, 50)

    def play_error(self) -> None:
        """Double low beep — something went wrong."""
        if not self.enabled:
            return
        self._beep(400, 60)
        time.sleep(0.08)
        self._beep(400, 60)

    # ── internals ─────────────────────────────────────────────────────────

    def _beep(self, frequency: int, duration_ms: int) -> None:
        """Platform-safe beep with debounce."""
        now = time.monotonic()
        if now - self._last_beep_at < self._min_interval_ms / 1000:
            return
        self._last_beep_at = now
        try:
            import winsound
            winsound.Beep(frequency, duration_ms)
        except ImportError:
            # Non-Windows: silently skip (no winsound available)
            pass
