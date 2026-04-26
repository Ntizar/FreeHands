"""Voice command listener (Phase 3: faster-whisper).

The heavy imports (``faster_whisper`` and ``sounddevice``) are deferred so the
rest of FreeHands keeps working on machines without microphone support.
"""
from __future__ import annotations

import re
import queue
import threading
import unicodedata
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceCommand:
    text: str
    action: str
    confidence: float = 1.0


WAKE_WORDS = ("freehands", "free hands", "ntizar")

COMMAND_PHRASES: dict[str, tuple[str, ...]] = {
    "double_click": ("doble click", "doble clic"),
    "right_click": ("click derecho", "clic derecho", "boton derecho", "menu contextual"),
    "zoom_in": ("zoom mas", "acercar", "ampliar", "aumentar"),
    "zoom_out": ("zoom menos", "alejar", "reducir", "disminuir"),
    "scroll_up": ("scroll arriba", "desplaza arriba", "sube", "subir"),
    "scroll_down": ("scroll abajo", "desplaza abajo", "baja", "bajar"),
    "toggle_pause": ("pausa", "pausar", "detente"),
    "resume": ("reanudar", "continua", "continuar", "activar", "despausar"),
    "escape": ("escape", "esc", "cancelar", "cancela", "atras"),
    "click": ("click", "clic", "pincha", "selecciona", "seleccionar"),
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_voice_command(text: str, *, require_wake_word: bool = True) -> str | None:
    """Map a transcript to an action id.

    By default we require a wake word (``FreeHands`` or ``Ntizar``) to reduce
    accidental activations. Pause/resume are allowed without wake word because
    they are safety controls.
    """
    norm = _normalise(text)
    if not norm:
        return None

    has_wake = any(w in norm for w in WAKE_WORDS)
    if has_wake:
        for wake in WAKE_WORDS:
            norm = norm.replace(wake, " ")
        norm = re.sub(r"\s+", " ", norm).strip()

    for action, phrases in COMMAND_PHRASES.items():
        if require_wake_word and not has_wake and action not in {"toggle_pause", "resume"}:
            continue
        if any(phrase in norm for phrase in phrases):
            return action
    return None


class VoiceListener:
    def __init__(
        self,
        language: str = "es",
        *,
        model_size: str = "tiny",
        chunk_seconds: float = 2.5,
        sample_rate: int = 16_000,
        require_wake_word: bool = True,
    ) -> None:
        self.language = language
        self.model_size = model_size
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.require_wake_word = require_wake_word
        self.transcripts: queue.Queue[str] = queue.Queue()
        self.commands: queue.Queue[VoiceCommand] = queue.Queue()
        self.errors: queue.Queue[str] = queue.Queue()
        self._audio: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> "VoiceListener":
        import sounddevice as sd

        if self._thread and self._thread.is_alive():
            return self

        def callback(indata, _frames, _time, status) -> None:
            if status:
                self.errors.put(str(status))
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                self._audio.put_nowait(mono)
            except queue.Full:
                pass

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=int(self.sample_rate * 0.5),
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._loop, name="FreeHandsVoice", daemon=True)
        self._thread.start()
        return self

    def drain_commands(self) -> list[VoiceCommand]:
        out: list[VoiceCommand] = []
        while True:
            try:
                out.append(self.commands.get_nowait())
            except queue.Empty:
                return out

    def drain_errors(self) -> list[str]:
        out: list[str] = []
        while True:
            try:
                out.append(self.errors.get_nowait())
            except queue.Empty:
                return out

    def _loop(self) -> None:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            needed = int(self.sample_rate * self.chunk_seconds)
            buf = np.empty((0,), dtype=np.float32)

            while not self._stop.is_set():
                try:
                    chunk = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf = np.concatenate([buf, chunk])
                if len(buf) < needed:
                    continue

                audio = buf[:needed]
                buf = buf[needed // 2:]

                if float(np.sqrt(np.mean(audio * audio))) < 0.006:
                    continue

                segments, info = model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if not text:
                    continue

                self.transcripts.put(text)
                action = parse_voice_command(text, require_wake_word=self.require_wake_word)
                if action:
                    confidence = getattr(info, "language_probability", 1.0) or 1.0
                    self.commands.put(VoiceCommand(text=text, action=action, confidence=float(confidence)))
        except Exception as exc:  # keep the rest of the app alive
            self.errors.put(str(exc))

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
