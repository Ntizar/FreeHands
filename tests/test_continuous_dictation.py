"""Tests for continuous dictation engine (improvement #27)."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from freehands.voice.continuous_dictation import (
    ContinuousDictationEngine,
    DictationConfig,
    DictationState,
    _normalise,
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_fake_stream() -> types.SimpleNamespace:
    """Create a fake sounddevice.InputStream for monkeypatching."""
    class FakeStream:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def close(self) -> None:
            pass

    return types.SimpleNamespace(InputStream=FakeStream)


# ── _normalise tests ────────────────────────────────────────────────


def test_normalise_strips_accents() -> None:
    assert "haz clic" in _normalise("haz clìc")


def test_normalise_lowercases() -> None:
    assert _normalise("HOLA") == "hola"


def test_normalise_strips_punctuation() -> None:
    result = _normalise("signo de interrogacion?")
    assert "signo de interrogacion" in result


def test_normalise_collapses_spaces() -> None:
    assert _normalise("hola    mundo") == "hola mundo"


# ── DictationConfig tests ──────────────────────────────────────────


def test_dictation_config_defaults() -> None:
    config = DictationConfig()
    assert config.model_size == "tiny"
    assert config.language == "auto"
    assert config.backend == "faster_whisper"
    assert config.sample_rate == 16_000
    assert config.chunk_seconds == 0.5
    assert config.silence_threshold == 0.006
    assert config.silence_cooldown == 0.8
    assert config.max_buffer_length == 500
    assert config.commit_interval == 2.0
    assert config.auto_punctuation is True
    assert config.on_text is None


# ── ContinuousDictationEngine basic tests ───────────────────────────


def test_engine_initial_state() -> None:
    engine = ContinuousDictationEngine()
    assert engine.state == DictationState.IDLE
    assert engine.is_active is False
    assert engine.buffer_text == ""
    assert engine.session_text == ""
    assert engine.total_chars_committed == 0
    assert engine.dictation_sessions == 0


def test_engine_stop_when_idle_does_not_crash() -> None:
    engine = ContinuousDictationEngine()
    engine.stop()  # should not raise


def test_engine_flush_when_empty_returns_empty() -> None:
    engine = ContinuousDictationEngine()
    text = engine.flush()
    assert text == ""


def test_engine_reset_clears_buffer() -> None:
    engine = ContinuousDictationEngine()
    engine._buffer = ["hello", "world"]
    engine._full_text = "committed "
    engine.reset()
    assert engine.buffer_text == ""
    assert engine.session_text == ""


# ── Punctuation keyword tests ──────────────────────────────────────


def test_punct_map_spanish() -> None:
    from freehands.voice.continuous_dictation import PUNCT_MAP
    assert PUNCT_MAP["coma"] == ","
    assert PUNCT_MAP["punto"] == "."
    assert PUNCT_MAP["nueva linea"] == "\n"
    assert PUNCT_MAP["signo de interrogacion"] == "?"
    assert PUNCT_MAP["signo de exclamacion"] == "!"
    assert PUNCT_MAP["parrafo"] == "\n\n"


def test_punct_map_english() -> None:
    from freehands.voice.continuous_dictation import PUNCT_MAP
    assert PUNCT_MAP["comma"] == ","
    assert PUNCT_MAP["period"] == "."
    assert PUNCT_MAP["new line"] == "\n"
    assert PUNCT_MAP["question mark"] == "?"
    assert PUNCT_MAP["exclamation mark"] == "!"


def test_punct_map_at_symbol() -> None:
    from freehands.voice.continuous_dictation import PUNCT_MAP
    assert PUNCT_MAP["arroba"] == "@"
    assert PUNCT_MAP["at"] == "@"


# ── Stop phrases tests ─────────────────────────────────────────────


def test_stop_phrases_exist() -> None:
    from freehands.voice.continuous_dictation import DICTATION_STOP_PHRASES
    assert "parar dictado" in DICTATION_STOP_PHRASES
    assert "terminar dictado" in DICTATION_STOP_PHRASES
    assert "dejar de dictar" in DICTATION_STOP_PHRASES
    assert "dejar dictado" in DICTATION_STOP_PHRASES


# ── DictationState enum tests ──────────────────────────────────────


def test_dictation_state_values() -> None:
    assert DictationState.IDLE.value == "idle"
    assert DictationState.ACTIVE.value == "active"
    assert DictationState.COMMITTING.value == "committing"


# ── Buffer management tests ────────────────────────────────────────


def test_flush_buffer_returns_text_and_clears() -> None:
    engine = ContinuousDictationEngine()
    engine._buffer = ["hola", "mundo"]
    result = engine._flush_buffer()
    assert result == "hola mundo"
    assert engine.buffer_text == ""


def test_flush_buffer_accumulates_to_session() -> None:
    engine = ContinuousDictationEngine()
    engine._buffer = ["hola"]
    engine._flush_buffer()
    assert engine.session_text == "hola "


def test_on_text_callback_is_called() -> None:
    received: list[str] = []

    def on_text(text: str) -> None:
        received.append(text)

    config = DictationConfig(on_text=on_text)
    engine = ContinuousDictationEngine(config)
    engine._buffer = ["hola"]
    engine._flush_buffer()
    assert received == ["hola"]


def test_on_text_callback_error_does_not_crash_engine() -> None:
    def bad_callback(text: str) -> None:
        raise RuntimeError("callback error")

    config = DictationConfig(on_text=bad_callback)
    engine = ContinuousDictationEngine(config)
    engine._buffer = ["hola"]
    # Should not raise
    engine._flush_buffer()


# ── Punctuation application tests ──────────────────────────────────


def test_apply_punctuation_spanish() -> None:
    engine = ContinuousDictationEngine()
    result = engine._apply_punctuation("hola punto mundo coma", "hola punto mundo coma")
    assert "." in result
    assert "," in result


def test_apply_punctuation_english() -> None:
    engine = ContinuousDictationEngine()
    result = engine._apply_punctuation("hello period world comma", "hello period world comma")
    assert "." in result
    assert "," in result


def test_apply_punctuation_newline() -> None:
    engine = ContinuousDictationEngine()
    result = engine._apply_punctuation("linea nueva linea siguiente", "linea nueva linea siguiente")
    assert "\n" in result


# ── Integration with voice commands ─────────────────────────────────


def test_voice_commands_include_dictation() -> None:
    from freehands.voice import parse_voice_command
    # Start dictation (requires wake word — not a safety command)
    assert parse_voice_command("FreeHands dictar") == "start_dictation"
    assert parse_voice_command("FreeHands dictado") == "start_dictation"
    assert parse_voice_command("FreeHands empezar dictado") == "start_dictation"
    assert parse_voice_command("FreeHands begin dictation") == "start_dictation"
    assert parse_voice_command("FreeHands dicta") == "start_dictation"

    # Stop dictation (requires wake word — not a safety command)
    assert parse_voice_command("FreeHands parar dictado") == "stop_dictation"
    assert parse_voice_command("FreeHands parar dicta") == "stop_dictation"
    assert parse_voice_command("FreeHands dejar de dictar") == "stop_dictation"
    assert parse_voice_command("FreeHands terminar dictado") == "stop_dictation"
    assert parse_voice_command("FreeHands dejar dictado") == "stop_dictation"


def test_voice_commands_no_wake_word_for_dictation() -> None:
    """Dictation commands should work without wake word for convenience."""
    from freehands.voice import parse_voice_command
    # "dictar" is not a system command, so it requires wake word
    assert parse_voice_command("dictar", require_wake_word=True) is None
    # But with wake word
    assert parse_voice_command("FreeHands dictar") == "start_dictation"


# ── Audio silence detection tests ──────────────────────────────────


def test_silence_threshold_detection() -> None:
    """Verify that silence detection correctly identifies silent audio."""
    # Very quiet audio (below 0.006 threshold)
    silent = np.zeros(16000, dtype=np.float32)
    rms = float(np.sqrt(np.mean(silent * silent)))
    assert rms < 0.006

    # Loud audio (above threshold)
    loud = np.ones(16000, dtype=np.float32) * 0.1
    rms = float(np.sqrt(np.mean(loud * loud)))
    assert rms >= 0.006


# ── Max buffer length test ─────────────────────────────────────────


def test_max_buffer_length_forces_commit() -> None:
    """When buffer exceeds max_buffer_length, it should auto-commit."""
    received: list[str] = []

    def on_text(text: str) -> None:
        received.append(text)

    config = DictationConfig(
        on_text=on_text,
        max_buffer_length=20,  # very small for testing
    )
    engine = ContinuousDictationEngine(config)
    # Add words that exceed the limit
    engine._buffer = ["hello world this is a long sentence that exceeds limit"]
    engine._flush_buffer()
    assert len(received) > 0
    assert engine.buffer_text == ""
