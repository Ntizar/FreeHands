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
    "double_click": ("double click", "doble click", "doble clic"),
    "right_click": ("right click", "context menu", "click derecho", "clic derecho", "boton derecho", "menu contextual"),
    "zoom_in": ("zoom in", "zoom mas", "acercar", "ampliar", "aumentar"),
    "zoom_out": ("zoom out", "zoom menos", "alejar", "reducir", "disminuir"),
    "scroll_up": ("scroll up", "page up", "scroll arriba", "desplaza arriba", "sube"),
    "scroll_down": ("scroll down", "page down", "scroll abajo", "desplaza abajo", "baja"),
    "toggle_pause": ("pause", "stop", "pausa", "pausar", "detente"),
    "resume": ("resume", "continue", "start", "activate", "reanudar", "continua", "continuar", "activar", "despausar"),
    "escape": ("escape", "esc", "cancel", "back", "cancelar", "cancela", "atras"),
    "click": ("click", "select", "clic", "pincha", "selecciona", "seleccionar"),
    # ── System commands (voice-only, no wake word needed — safety controls) ──
    "show_desktop": ("show desktop", "mostrar escritorio", "escritorio", "minimizar todo", "minimiza todo"),
    "screenshot": ("screenshot", "captura de pantalla", "foto pantalla", "captura pantalla"),
    "volume_up": ("volume up", "subir volumen", "volumen arriba", "mas volumen", "sube volumen", "volumen mas"),
    "volume_down": ("volume down", "bajar volumen", "volumen abajo", "menos volumen", "baja volumen", "volumen menos"),
    "volume_mute": ("mute", "silencio", "silenciar", "mudo", "muted", "quitar sonido"),
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SYSTEM_COMMANDS = {"show_desktop", "screenshot", "volume_up", "volume_down", "volume_mute"}
SAFETY_COMMANDS = {"toggle_pause", "resume", "show_desktop", "screenshot", "volume_up", "volume_down", "volume_mute"}


def parse_voice_command(
    text: str,
    *,
    require_wake_word: bool = True,
    wake_words: tuple[str, ...] = WAKE_WORDS,
) -> str | None:
    """Map a transcript to an action id.

    By default we require a wake word (``FreeHands`` or ``Ntizar``) to reduce
    accidental activations. Safety controls (pause/resume, system commands)
    are allowed without wake word.
    """
    norm = _normalise(text)
    if not norm:
        return None

    wake_words = tuple(_normalise(w) for w in wake_words if _normalise(w)) or WAKE_WORDS
    has_wake = any(w in norm for w in wake_words)
    if has_wake:
        for wake in wake_words:
            norm = norm.replace(wake, " ")
        norm = re.sub(r"\s+", " ", norm).strip()

    # Check system commands first (higher priority, no wake word needed)
    for action, phrases in COMMAND_PHRASES.items():
        if action in SYSTEM_COMMANDS:
            if any(phrase in norm for phrase in phrases):
                return action

    # Then check regular commands (wake word required unless safety control)
    for action, phrases in COMMAND_PHRASES.items():
        if action in SYSTEM_COMMANDS:
            continue
        if require_wake_word and not has_wake and action not in {"toggle_pause", "resume"}:
            continue
        if any(phrase in norm for phrase in phrases):
            return action
    return None


class VoiceListener:
    def __init__(
        self,
        language: str = "auto",
        *,
        model_size: str = "tiny",
        chunk_seconds: float = 2.5,
        sample_rate: int = 16_000,
        require_wake_word: bool = True,
        wake_words: tuple[str, ...] = WAKE_WORDS,
        backend: str = "faster_whisper",
        vosk_model_path: str | None = None,
    ) -> None:
        self.language = language
        self.model_size = model_size
        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate
        self.require_wake_word = require_wake_word
        self.wake_words = wake_words
        self.backend = backend
        self.vosk_model_path = vosk_model_path
        self.transcripts: queue.Queue[str] = queue.Queue()
        self.commands: queue.Queue[VoiceCommand] = queue.Queue()
        self.errors: queue.Queue[str] = queue.Queue()
        self._audio: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> "VoiceListener":
        import sounddevice as sd

        if self.backend == "vosk":
            return self._start_vosk()

        if self.backend != "faster_whisper":
            self.errors.put(
                f"Voice backend '{self.backend}' is experimental; using faster_whisper for now."
            )

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

                transcribe_kwargs = {
                    "beam_size": 1,
                    "vad_filter": True,
                    "condition_on_previous_text": False,
                }
                if self.language and self.language.lower() != "auto":
                    transcribe_kwargs["language"] = self.language
                segments, info = model.transcribe(audio, **transcribe_kwargs)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if not text:
                    continue

                self.transcripts.put(text)
                action = parse_voice_command(
                    text,
                    require_wake_word=self.require_wake_word,
                    wake_words=self.wake_words,
                )
                if action:
                    confidence = getattr(info, "language_probability", 1.0) or 1.0
                    self.commands.put(VoiceCommand(text=text, action=action, confidence=float(confidence)))
        except Exception as exc:  # keep the rest of the app alive
            self.errors.put(str(exc))

    def _start_vosk(self) -> "VoiceListener":
        """Start the Vosk offline ASR backend."""
        import sounddevice as sd

        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            self.errors.put(
                "Vosk backend selected but 'vosk' package is not installed. "
                "Install it with: pip install vosk"
            )
            return self

        if self.vosk_model_path:
            try:
                self._vosk_model = Model(self.vosk_model_path)
            except Exception as exc:
                self.errors.put(f"Vosk model load failed ({exc}); falling back to default.")
                self._vosk_model = Model()
        else:
            try:
                self._vosk_model = Model()
            except Exception as exc:
                self.errors.put(f"Vosk default model load failed ({exc}); falling back to faster_whisper.")
                return self

        SetLogLevel(-1)  # silence Vosk logs

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._vosk_callback,
            blocksize=int(self.sample_rate * 0.5),
        )
        self._stream.start()

        self._vosk_recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        self._vosk_recognizer.SetWords(True)

        self._thread = threading.Thread(target=self._vosk_loop, name="FreeHandsVosk", daemon=True)
        self._thread.start()
        return self

    def _vosk_callback(self, indata, _frames, _time, status) -> None:
        if status:
            self.errors.put(str(status))
        try:
            self._audio.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            pass

    def _vosk_loop(self) -> None:
        """Main loop for the Vosk backend."""
        try:
            while not self._stop.is_set():
                try:
                    chunk = self._audio.get(timeout=0.2)
                except queue.Empty:
                    continue

                if float(np.sqrt(np.mean(chunk * chunk))) < 0.006:
                    continue

                if self._vosk_recognizer.AcceptWaveform(chunk.tobytes()):
                    result = self._vosk_recognizer.Result()
                    # Parse JSON result
                    import json
                    try:
                        data = json.loads(result)
                        text = data.get("text", "").strip()
                    except (json.JSONDecodeError, KeyError):
                        text = ""

                    if not text:
                        continue

                    self.transcripts.put(text)
                    action = parse_voice_command(
                        text,
                        require_wake_word=self.require_wake_word,
                        wake_words=self.wake_words,
                    )
                    if action:
                        self.commands.put(VoiceCommand(text=text, action=action, confidence=1.0))
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
