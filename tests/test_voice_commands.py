from freehands.voice import VoiceListener, parse_voice_command


def test_voice_requires_wake_word_for_clicks() -> None:
    assert parse_voice_command("haz clic") is None
    assert parse_voice_command("FreeHands haz clic") == "click"
    assert parse_voice_command("Ntizar doble clic") == "double_click"


def test_voice_safety_controls_without_wake_word() -> None:
    assert parse_voice_command("pausa") == "toggle_pause"
    assert parse_voice_command("reanudar") == "resume"
    assert parse_voice_command("pause") == "toggle_pause"
    assert parse_voice_command("resume") == "resume"


def test_voice_english_action_synonyms() -> None:
    assert parse_voice_command("FreeHands click") == "click"
    assert parse_voice_command("Ntizar right click") == "right_click"
    assert parse_voice_command("FreeHands double click") == "double_click"
    assert parse_voice_command("Ntizar zoom in") == "zoom_in"
    assert parse_voice_command("Ntizar zoom out") == "zoom_out"
    assert parse_voice_command("FreeHands scroll down") == "scroll_down"
    assert parse_voice_command("FreeHands scroll up") == "scroll_up"
    assert parse_voice_command("FreeHands cancel") == "escape"


def test_voice_spanish_action_synonyms() -> None:
    assert parse_voice_command("FreeHands boton derecho") == "right_click"
    assert parse_voice_command("Ntizar zoom mas") == "zoom_in"
    assert parse_voice_command("Ntizar alejar") == "zoom_out"
    assert parse_voice_command("FreeHands scroll abajo") == "scroll_down"
    assert parse_voice_command("FreeHands sube") == "scroll_up"
    assert parse_voice_command("FreeHands cancelar") == "escape"


def test_voice_custom_wake_words() -> None:
    assert parse_voice_command("control clic", wake_words=("control",)) == "click"
    assert parse_voice_command("Ntizar clic", wake_words=("control",)) is None


def test_experimental_backend_keeps_faster_whisper_fallback_notice(monkeypatch) -> None:
    class FakeStream:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    import sys
    import types

    fake_sd = types.SimpleNamespace(InputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(VoiceListener, "_loop", lambda self: None)

    listener = VoiceListener(backend="vibevoice_asr")
    listener.start()
    try:
        assert listener.drain_errors() == [
            "Voice backend 'vibevoice_asr' is experimental; using faster_whisper for now."
        ]
    finally:
        listener.stop()
