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


def test_voice_system_commands_no_wake_word() -> None:
    """System commands (show desktop, screenshot, volume) work without wake word."""
    # Show desktop
    assert parse_voice_command("show desktop") == "show_desktop"
    assert parse_voice_command("mostrar escritorio") == "show_desktop"
    assert parse_voice_command("escritorio") == "show_desktop"
    assert parse_voice_command("minimizar todo") == "show_desktop"
    assert parse_voice_command("minimiza todo") == "show_desktop"

    # Screenshot
    assert parse_voice_command("screenshot") == "screenshot"
    assert parse_voice_command("captura de pantalla") == "screenshot"
    assert parse_voice_command("foto pantalla") == "screenshot"
    assert parse_voice_command("captura pantalla") == "screenshot"

    # Volume up
    assert parse_voice_command("volume up") == "volume_up"
    assert parse_voice_command("subir volumen") == "volume_up"
    assert parse_voice_command("volumen arriba") == "volume_up"
    assert parse_voice_command("mas volumen") == "volume_up"
    assert parse_voice_command("sube volumen") == "volume_up"

    # Volume down
    assert parse_voice_command("volume down") == "volume_down"
    assert parse_voice_command("bajar volumen") == "volume_down"
    assert parse_voice_command("volumen abajo") == "volume_down"
    assert parse_voice_command("menos volumen") == "volume_down"
    assert parse_voice_command("baja volumen") == "volume_down"

    # Mute
    assert parse_voice_command("mute") == "volume_mute"
    assert parse_voice_command("silencio") == "volume_mute"
    assert parse_voice_command("silenciar") == "volume_mute"
    assert parse_voice_command("mudo") == "volume_mute"
    assert parse_voice_command("muted") == "volume_mute"
    assert parse_voice_command("quitar sonido") == "volume_mute"


def test_voice_system_commands_with_wake_word() -> None:
    """System commands also work when wake word is present."""
    assert parse_voice_command("FreeHands subir volumen") == "volume_up"
    assert parse_voice_command("Ntizar captura de pantalla") == "screenshot"
    assert parse_voice_command("FreeHands mostrar escritorio") == "show_desktop"
    assert parse_voice_command("Ntizar silenciar") == "volume_mute"


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


def test_vosk_backend_requires_vosk_package(monkeypatch) -> None:
    """When Vosk is selected but not installed, an error is queued."""
    import sys
    import types

    # Make sounddevice available but block vosk import
    class FakeStream:
        def __init__(self, *args, **kwargs) -> None:
            pass
        def start(self) -> None:
            pass

    fake_sd = types.SimpleNamespace(InputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    # Make vosk import fail by patching sys.modules
    if "vosk" in sys.modules:
        del sys.modules["vosk"]
    # Block the actual vosk import
    class BlockVoskFinder:
        def find_module(self, name, path=None):
            if name == "vosk":
                return self
            return None
        def load_module(self, name):
            raise ImportError("No module named 'vosk'")

    finder = BlockVoskFinder()
    sys.meta_path.insert(0, finder)
    try:
        listener = VoiceListener(backend="vosk")
        listener.start()
        try:
            errors = listener.drain_errors()
            assert len(errors) == 1
            assert "vosk" in errors[0].lower()
        finally:
            listener.stop()
    finally:
        sys.meta_path.remove(finder)


def test_vosk_listener_accepts_model_path(monkeypatch) -> None:
    """The listener stores the vosk_model_path and passes it to Model()."""
    import sys
    import types

    class FakeStream:
        def __init__(self, *args, **kwargs) -> None:
            pass
        def start(self) -> None:
            pass

    fake_sd = types.SimpleNamespace(InputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    # Mock vosk imports — Model must be a callable class
    models_created = []

    class MockVoskModel:
        def __init__(self, path=None):
            models_created.append(path)

    class MockRecognizer:
        def __init__(self, model, sr):
            pass
        def SetWords(self, val=True):
            pass

    fake_vosk = types.SimpleNamespace(
        KaldiRecognizer=MockRecognizer,
        Model=MockVoskModel,
        SetLogLevel=lambda x: None,
    )
    monkeypatch.setitem(sys.modules, "vosk", fake_vosk)

    listener = VoiceListener(backend="vosk", vosk_model_path="/path/to/model")
    listener.start()
    try:
        assert listener.vosk_model_path == "/path/to/model"
        assert len(models_created) == 1
        assert models_created[0] == "/path/to/model"
    finally:
        listener.stop()


def test_vosk_backend_default_model_when_no_path(monkeypatch) -> None:
    """When no model path is given, the listener uses the default Model()."""
    import sys
    import types

    class FakeStream:
        def __init__(self, *args, **kwargs) -> None:
            pass
        def start(self) -> None:
            pass

    fake_sd = types.SimpleNamespace(InputStream=FakeStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    model_loaded = []

    class TrackingModel:
        def __init__(self, path=None):
            model_loaded.append(path)

    class MockRecognizer:
        def __init__(self, model, sr):
            pass
        def SetWords(self, val=True):
            pass

    fake_vosk = types.SimpleNamespace(
        KaldiRecognizer=MockRecognizer,
        Model=TrackingModel,
        SetLogLevel=lambda x: None,
    )
    monkeypatch.setitem(sys.modules, "vosk", fake_vosk)

    listener = VoiceListener(backend="vosk")
    listener.start()
    try:
        assert len(model_loaded) == 1
        assert model_loaded[0] is None  # default Model() with no path
    finally:
        listener.stop()
