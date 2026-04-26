"""Voice listener stub (Phase 3 — faster-whisper).

Importing :mod:`faster_whisper` is deferred so the MVP runs without it.
"""
from __future__ import annotations

import queue
import threading


class VoiceListener:
    def __init__(self, language: str = "es") -> None:
        self.language = language
        self.transcripts: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False

    def start(self) -> "VoiceListener":
        # Real implementation will spin up faster-whisper here.
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
