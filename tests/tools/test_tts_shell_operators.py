"""Shell operators in spoken text: pauses and silence, never inserted words."""

import pytest

from tools.tts_text_normalize import pause_shell_operators_for_tts, prepare_spoken_text


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Run hermes --version && hermes doctor.", "Run hermes --version, hermes doctor."),
        ("Try make || make clean.", "Try make, make clean."),
        ("Use make test 2>&1 | tee out.", "Use make test; tee out."),
        ("Send it with echo hi >&2 now.", "Send it with echo hi now."),
        ("`make && make install`", "make, make install"),
    ],
)
def test_operators_become_pauses_or_silence(text, spoken):
    assert prepare_spoken_text(text) == spoken


def test_no_english_word_is_inserted_into_other_languages():
    assert prepare_spoken_text("Führe make && make install aus.") == "Führe make, make install aus."


@pytest.mark.parametrize(
    "text",
    ["Rock & roll.", "Q&A and AT&T stay.", "Set stt.local.model to medium.", "a&&b stays as it was"],
)
def test_text_without_spaced_operators_is_untouched(text):
    assert pause_shell_operators_for_tts(text) == text


def test_empty_table_cell_is_not_an_operator():
    table = "| a || b |\n|---|---|---|\n| 1 || 2 |"
    assert pause_shell_operators_for_tts(table) == table
