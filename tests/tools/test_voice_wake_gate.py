"""Pure tests for the Discord VC wake-name gate."""

from tools.voice_wake_gate import (
    apply_wake_gate,
    default_wake_names,
    normalize_wake_names,
    wake_gate_active,
)


def test_auto_off_for_one_human():
    ok, text = apply_wake_gate("what's the weather", policy="auto", human_count=1)
    assert ok is True
    assert text == "what's the weather"


def test_auto_drops_without_name_when_crowded():
    ok, _ = apply_wake_gate("what's the weather", policy="auto", human_count=2)
    assert ok is False


def test_auto_accepts_leading_name():
    ok, text = apply_wake_gate("Hermes what's the weather", policy="auto", human_count=3)
    assert ok is True
    assert text == "what's the weather"


def test_hey_hermes_prefix():
    ok, text = apply_wake_gate("hey Hermes, list the files", policy="auto", human_count=2)
    assert ok is True
    assert text == "list the files"


def test_trailing_name():
    ok, text = apply_wake_gate("list the files, Hermes", policy="auto", human_count=2)
    assert ok is True
    assert text == "list the files"


def test_name_only_still_accepted():
    ok, text = apply_wake_gate("Hermes", policy="auto", human_count=2)
    assert ok is True
    assert text == ""


def test_two_word_bot_name_preferred():
    ok, text = apply_wake_gate(
        "Hermes Agent check disk",
        policy="true",
        human_count=1,
        names=["Hermes Agent", "Hermes"],
    )
    assert ok is True
    assert text == "check disk"


def test_policy_false_never_gates():
    ok, text = apply_wake_gate("chatter", policy="false", human_count=9)
    assert ok is True
    assert text == "chatter"


def test_policy_true_always_gates():
    ok, _ = apply_wake_gate("chatter", policy="true", human_count=1)
    assert ok is False


def test_wake_gate_active_counts():
    assert wake_gate_active("auto", 0) is False
    assert wake_gate_active("auto", 1) is False
    assert wake_gate_active("auto", 2) is True
    assert wake_gate_active("true", 0) is True
    assert wake_gate_active("false", 8) is False


def test_normalize_caps_at_two_words_and_dedupes():
    assert normalize_wake_names(["  Hermes Agent bot ", "hermes agent", "Hermes"]) == [
        "Hermes Agent",
        "Hermes",
    ]


def test_default_wake_names_includes_hermes_and_bot():
    names = default_wake_names("Hermes Agent")
    assert "Hermes Agent" in names
    assert "Hermes" in names
