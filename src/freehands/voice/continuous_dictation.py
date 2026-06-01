"""Continuous dictation engine — free-form voice-to-text.

Unlike the discrete-command VoiceListener, this module transcribes
arbitrary speech and writes the result into the currently focused
text field on the OS.  It is activated by a voice command (\"dictar\" /
\"escribir\") and terminated by \"parar dictado\" / \"stop dictado\" /
\"escapar\" or by the Escape key.

The engine runs on top of the same ASR backend (faster_whisper or
Vosk) as the command listener, reusing the audio stream so both
systems can share a single microphone.

Design decisions
----------------
- Transcription buffer is accumulated across short audio chunks
  (500 ms) to produce fluent, punctuated text.
- A silence detector (RMS < threshold for > 800 ms) auto-commits
  the current buffer as a space-separated phrase.
- Punctuation keywords are translated: \"coma\", \"punto", "nueva linea",
  "salto de linea", "signo de interrogacion", "signo de exclamacion".
- The dictation engine is stateful: idle / active / committing.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


# ── Whisper supported languages (98 languages, Whisper tiny-base) ──
# https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
WHISPER_LANGUAGES: dict[str, str] = {
    # European
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "pl": "Polish",
    "ro": "Romanian",
    "el": "Greek",
    "hu": "Hungarian",
    "cs": "Czech",
    "sk": "Slovak",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "da": "Danish",
    "fi": "Finnish",
    "sv": "Swedish",
    "no": "Norwegian",
    "tr": "Turkish",
    "he": "Hebrew",
    "is": "Icelandic",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ca": "Catalan",
    "gl": "Galician",
    "eu": "Basque",
    "ga": "Irish",
    "mt": "Maltese",
    "cy": "Welsh",
    "mk": "Macedonian",
    "sq": "Albanian",
    "bs": "Bosnian",
    "af": "Afrikaans",
    # Middle Eastern / South Asian
    "ar": "Arabic",
    "fa": "Persian (Farsi)",
    "ur": "Urdu",
    "hi": "Hindi",
    "bn": "Bengali",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ne": "Nepali",
    "si": "Sinhala",
    "my": "Burmese",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Filipino",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    # African
    "sw": "Swahili",
    "zu": "Zulu",
    "am": "Amharic",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "so": "Somali",
    "mg": "Malagasy",
    "ny": "Chichewa",
    "rw": "Kinyarwanda",
    "sn": "Shona",
    "mn": "Mongolian",
    "ka": "Georgian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "kk": "Kazakh",
    "ky": "Kyrgyz",
    "tg": "Tajik",
    "tk": "Turkmen",
    "uz": "Uzbek",
    # Southeast Asian / Pacific
    "jw": "Javanese",
    "su": "Sundanese",
    "haw": "Hawaiian",
    "mi": "Maori",
    # Other
    "la": "Latin",
    "lb": "Luxembourgish",
    "oc": "Occitan",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "ps": "Pashto",
    "po": "Polish",
    "tt": "Tatar",
    "yi": "Yiddish",
    "bo": "Tibetan",
    "as": "Assamese",
    "be": "Belarusian",
    "lo": "Lao",
    "km": "Khmer",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "sa": "Sanskrit",
}

# ── Punctuation keywords (ES + EN) ─────────────────────────────────
PUNCT_MAP: dict[str, str] = {
    # Spanish
    "coma": ",",
    "punto": ".",
    "nueva linea": "\n",
    "salto de linea": "\n",
    "signo de interrogacion": "?",
    "signo de exclamacion": "!",
    "interrogacion": "?",
    "exclamacion": "!",
    "abrir interrogacion": "?",
    "cerrar interrogacion": "?",
    "abrir exclamacion": "!",
    "cerrar exclamacion": "!",
    "parrafo": "\n\n",
    "guion": "-",
    "guion bajo": "_",
    "arroba": "@",
    "punto coma": ";",
    "dos puntos": ":",
    "parentesis": "(",
    "cerrar parentesis": ")",
    "abrir parentesis": "(",
    "corchetes": "[",
    "cerrar corchetes": "]",
    "abrir corchetes": "[",
    "llaves": "{",
    "cerrar llaves": "}",
    "abrir llaves": "{",
    # English
    "comma": ",",
    "period": ".",
    "dot": ".",
    "new line": "\n",
    "newline": "\n",
    "question mark": "?",
    "exclamation mark": "!",
    "question": "?",
    "exclamation": "!",
    "paragraph": "\n\n",
    "hyphen": "-",
    "at": "@",
    "semicolon": ";",
    "colon": ":",
    "open paren": "(",
    "close paren": ")",
    "open bracket": "[",
    "close bracket": "]",
    "open brace": "{",
    "close brace": "}",
}

# Commands that stop dictation
DICTATION_STOP_PHRASES: tuple[str, ...] = (
    "parar dictado",
    "parar dicta",
    "dejar de dictar",
    "terminar dictado",
    "terminar dicta",
    "dejar dictado",
)

class DictationState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMMITTING = "committing"  # flushing buffer before idle


@dataclass
class DictationConfig:
    """Configuration for the continuous dictation engine."""
    # ASR backend
    model_size: str = "tiny"
    language: str = "auto"
    backend: str = "faster_whisper"
    vosk_model_path: str | None = None

    # Audio
    sample_rate: int = 16_000
    chunk_seconds: float = 0.5  # 500 ms chunks for continuous mode
    silence_threshold: float = 0.006  # RMS below this = silence
    silence_cooldown: float = 0.8  # seconds of silence before auto-commit

    # Buffer
    max_buffer_length: int = 500  # chars before force-commit
    commit_interval: float = 2.0  # seconds between auto-commits

    # Punctuation auto-injection
    auto_punctuation: bool = True
    sentence_end_threshold: float = 0.5  # confidence threshold for sentence-ending

    # Voice typing mode (mejora #37)
    voice_typing_mode: bool = True  # True = full typing mode with language detection
    detected_language: str = ""  # Auto-detected language from first transcription

    # Callback
    on_text: Callable[[str], None] | None = None  # called with committed text


class ContinuousDictationEngine:
    """Voice-to-text engine for continuous dictation.

    Unlike VoiceListener (which maps speech to discrete commands),
    this engine accumulates transcribed text and writes it to the
    focused text field when dictation ends.

    Usage:
        engine = ContinuousDictationEngine()
        engine.start()
        # ... dictation is active ...
        engine.stop()  # flush buffer and return to idle
    """

    def __init__(self, config: DictationConfig | None = None) -> None:
        self.config = config or DictationConfig()
        self.state = DictationState.IDLE
        self._buffer: list[str] = []  # accumulated words
        self._full_text: str = ""  # committed text for this session
        self._last_speech_time: float = 0.0
        self._last_commit_time: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._audio: deque[np.ndarray] = deque(maxlen=32)
        self._lock = threading.Lock()

        # Stats
        self.total_chars_committed: int = 0
        self.dictation_sessions: int = 0
        self.detected_language: str = ""  # Auto-detected from first transcription

    # ── Public API ──────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state == DictationState.ACTIVE

    @property
    def buffer_text(self) -> str:
        with self._lock:
            return " ".join(self._buffer).strip()

    @property
    def session_text(self) -> str:
        with self._lock:
            return self._full_text

    def start(self) -> "ContinuousDictationEngine":
        """Start the dictation engine and begin listening."""
        if self._thread and self._thread.is_alive():
            return self

        self.state = DictationState.ACTIVE
        self._stop.clear()
        self._buffer.clear()
        self._last_commit_time = time.monotonic()

        # Import here to avoid hard dependency
        import sounddevice as sd

        def callback(indata, _frames, _time, status) -> None:
            if status:
                pass  # silently ignore audio status errors
            mono = np.asarray(indata[:, 0], dtype=np.float32).copy()
            try:
                self._audio.append(mono)
            except Exception:
                pass

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=int(self.config.sample_rate * 0.25),
        )
        self._stream.start()

        self._thread = threading.Thread(
            target=self._loop, name="FreeHandsDictation", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop dictation, flush buffer, return to idle."""
        self._stop.set()
        # Flush remaining buffer
        self._flush_buffer()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
        self.state = DictationState.IDLE

    def flush(self) -> str:
        """Force-flush the current buffer and return the text."""
        text = self._flush_buffer()
        return text

    def reset(self) -> None:
        """Clear the buffer and session text."""
        with self._lock:
            self._buffer.clear()
            self._full_text = ""

    # ── Internal loop ───────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            if self.config.backend == "vosk":
                self._vosk_loop()
            else:
                self._whisper_loop()
        except Exception as exc:
            print(f"[dictation] error: {exc}")
            self.state = DictationState.IDLE

    def _whisper_loop(self) -> None:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            self.config.model_size,
            device="cpu",
            compute_type="int8",
        )
        needed = int(self.config.sample_rate * self.config.chunk_seconds)
        buf = np.empty((0,), dtype=np.float32)

        while not self._stop.is_set():
            try:
                chunk = self._audio.popleft()
            except IndexError:
                self._stop.wait(timeout=0.1)
                continue

            buf = np.concatenate([buf, chunk])
            if len(buf) < needed:
                continue

            audio = buf[:needed]
            buf = buf[needed // 2:]

            # Silence detection
            rms = float(np.sqrt(np.mean(audio * audio)))
            if rms < self.config.silence_threshold:
                # Check if we should auto-commit due to silence
                if self._buffer and (time.monotonic() - self._last_speech_time
                                     > self.config.silence_cooldown):
                    self._auto_commit()
                continue

            self._last_speech_time = time.monotonic()

            transcribe_kwargs = {
                "beam_size": 1,
                "vad_filter": False,  # we do our own silence detection
                "condition_on_previous_text": True,
            }
            if self.config.language and self.config.language.lower() != "auto":
                transcribe_kwargs["language"] = self.config.language

            try:
                segments, info = model.transcribe(audio, **transcribe_kwargs)
                text = " ".join(seg.text.strip() for seg in segments).strip()
                # Auto-detect language on first transcription (mejora #37)
                if (self.config.voice_typing_mode
                        and not self.detected_language
                        and hasattr(info, 'language')
                        and info.language):
                    self.detected_language = info.language
                    lang_name = WHISPER_LANGUAGES.get(info.language, info.language)
                    print(f"[dictation] idioma detectado: {lang_name} ({info.language})")
            except Exception:
                continue

            if not text:
                continue

            self._process_transcript(text)

    def _vosk_loop(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError:
            self.state = DictationState.IDLE
            return

        if self.config.vosk_model_path:
            try:
                vosk_model = Model(self.config.vosk_model_path)
            except Exception:
                vosk_model = Model()
        else:
            try:
                vosk_model = Model()
            except Exception:
                self.state = DictationState.IDLE
                return

        SetLogLevel(-1)
        recognizer = KaldiRecognizer(vosk_model, self.config.sample_rate)
        recognizer.SetWords(True)

        while not self._stop.is_set():
            try:
                chunk = self._audio.popleft()
            except IndexError:
                self._stop.wait(timeout=0.1)
                continue

            rms = float(np.sqrt(np.mean(chunk * chunk)))
            if rms < self.config.silence_threshold:
                if self._buffer and (time.monotonic() - self._last_speech_time
                                     > self.config.silence_cooldown):
                    self._auto_commit()
                continue

            self._last_speech_time = time.monotonic()

            if recognizer.AcceptWaveform(chunk.tobytes()):
                result = recognizer.Result()
                import json
                try:
                    data = json.loads(result)
                    text = data.get("text", "").strip()
                except (json.JSONDecodeError, KeyError):
                    text = ""

                if not text:
                    continue

                self._process_transcript(text)

    def _process_transcript(self, text: str) -> None:
        """Process a transcript: check for stop phrases, punctuation, buffer."""
        norm = _normalise(text)

        # Check for stop phrases first
        for phrase in DICTATION_STOP_PHRASES:
            if phrase in norm:
                self._flush_buffer()
                return

        # Check for punctuation keywords
        processed = self._apply_punctuation(text, norm)

        with self._lock:
            # Split into words and add to buffer
            words = processed.split()
            self._buffer.extend(words)

            # Check buffer length limit
            buffer_str = " ".join(self._buffer)
            if len(buffer_str) >= self.config.max_buffer_length:
                self._flush_buffer()

            # Check auto-commit interval
            now = time.monotonic()
            if (now - self._last_commit_time > self.config.commit_interval
                    and self._buffer):
                self._flush_buffer()

    def _apply_punctuation(self, text: str, norm: str) -> str:
        """Replace punctuation keywords with actual punctuation marks."""
        result = text
        for keyword, punct in PUNCT_MAP.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            result = pattern.sub(punct, result)
        return result

    def _auto_commit(self) -> None:
        """Auto-commit buffer due to silence or timeout."""
        if not self._buffer:
            return
        self._flush_buffer()

    def _flush_buffer(self) -> str:
        """Flush the buffer, commit text, return it."""
        with self._lock:
            if not self._buffer:
                return ""

            text = " ".join(self._buffer).strip()
            self._buffer.clear()
            self._full_text += text + " "
            self.total_chars_committed += len(text)
            self._last_commit_time = time.monotonic()

        # Call the on_text callback if registered
        if self.config.on_text and text:
            try:
                self.config.on_text(text)
            except Exception as exc:
                print(f"[dictation] on_text callback error: {exc}")

        return text


# ── Helpers ─────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Normalise text for keyword matching."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9ñ\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
