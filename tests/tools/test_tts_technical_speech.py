"""Dotted identifiers, emails, flags and shell operators in the TTS script."""
from tools.tts_technical_speech import speak_technical_tokens
from tools.tts_text_normalize import prepare_spoken_text


class TestDottedIdentifiers:
    def test_config_key_is_not_split_into_sentences(self):
        spoken = prepare_spoken_text("Set stt.local.model to medium.")
        assert "stt. local" not in spoken
        assert "dot" in spoken

    def test_hostname_is_not_split_into_sentences(self):
        spoken = prepare_spoken_text("Reachable at api.anthropic.com over TLS.")
        assert "api. anthropic" not in spoken

    def test_email_keeps_its_parts_together(self):
        spoken = prepare_spoken_text("Mail ops.team@example.com today.")
        assert "ops. team" not in spoken
        assert " at " in spoken


class TestShellOperators:
    def test_and_operator_does_not_stutter(self):
        # "&" expands to " and ", so "&&" reached the engine as "and and".
        spoken = prepare_spoken_text("Run hermes --version && hermes doctor.")
        assert "and and" not in spoken
        assert "and then" in spoken

    def test_or_operator_is_not_left_as_punctuation(self):
        spoken = prepare_spoken_text("Try foo || bar.")
        assert "||" not in spoken and ";;" not in spoken
        assert "or else" in spoken

    def test_single_ampersand_stays_prose(self):
        assert speak_technical_tokens("Tom & Jerry at R&D") == "Tom & Jerry at R&D"


class TestLongFlags:
    def test_every_word_of_a_long_flag_survives(self):
        spoken = prepare_spoken_text("Pass --dangerously-skip-permissions to it.")
        for word in ("dangerously", "skip", "permissions"):
            assert word in spoken


class TestPassThrough:
    def test_prose_units_and_ratios_are_unchanged(self):
        for text in (
            "The build is fine. It held. Nothing else.",
            "Costs went up 15% to $4.20 per unit, about 30 degrees C.",
            "Meet at 3.30 p.m. sharp, i.e. not late.",
            "Either and/or works, N/A otherwise, TCP/IP regardless.",
        ):
            assert speak_technical_tokens(text) == text

    def test_versions_and_addresses_are_left_for_the_engine(self):
        text = "Version v0.21.2 shipped and 10.0.3.14 responded."
        assert speak_technical_tokens(text) == text

    def test_filesystem_paths_are_left_to_the_existing_rules(self):
        # Paths are out of scope here; PR #89367 covers them.
        text = "Open /etc/hosts and ~/.config/app.toml"
        assert speak_technical_tokens(text) == text
