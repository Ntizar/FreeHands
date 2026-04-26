from freehands.voice import parse_voice_command


def test_voice_requires_wake_word_for_clicks() -> None:
    assert parse_voice_command("haz clic") is None
    assert parse_voice_command("FreeHands haz clic") == "click"
    assert parse_voice_command("Ntizar doble clic") == "double_click"


def test_voice_safety_controls_without_wake_word() -> None:
    assert parse_voice_command("pausa") == "toggle_pause"
    assert parse_voice_command("reanudar") == "resume"


def test_voice_spanish_action_synonyms() -> None:
    assert parse_voice_command("FreeHands boton derecho") == "right_click"
    assert parse_voice_command("Ntizar zoom mas") == "zoom_in"
    assert parse_voice_command("Ntizar alejar") == "zoom_out"
    assert parse_voice_command("FreeHands scroll abajo") == "scroll_down"
    assert parse_voice_command("FreeHands sube") == "scroll_up"
    assert parse_voice_command("FreeHands cancelar") == "escape"
