"""Tests for per-provider TTS input-character limits.

Replaces the old global ``MAX_TEXT_LENGTH = 4000`` cap that truncated every
provider at 4000 chars even though OpenAI allows 4096, xAI allows 15000,
MiniMax allows 10000, and ElevenLabs allows 5000-40000 depending on model.
"""

import json
from unittest.mock import patch

import pytest

from tools.tts_tool import (
    ELEVENLABS_MODEL_MAX_TEXT_LENGTH,
    FALLBACK_MAX_TEXT_LENGTH,
    PROVIDER_MAX_TEXT_LENGTH,
    _resolve_max_text_length,
)


class TestResolveMaxTextLength:
    def test_edge_default(self):
        assert _resolve_max_text_length("edge", {}) == PROVIDER_MAX_TEXT_LENGTH["edge"]

    def test_openai_default_is_4096(self):
        assert _resolve_max_text_length("openai", {}) == 4096

    def test_xai_default_is_15000(self):
        assert _resolve_max_text_length("xai", {}) == 15000

    def test_minimax_default_is_10000(self):
        assert _resolve_max_text_length("minimax", {}) == 10000

    def test_mistral_default(self):
        assert _resolve_max_text_length("mistral", {}) == PROVIDER_MAX_TEXT_LENGTH["mistral"]

    def test_gemini_default(self):
        assert _resolve_max_text_length("gemini", {}) == PROVIDER_MAX_TEXT_LENGTH["gemini"]

    def test_unknown_provider_falls_back(self):
        assert _resolve_max_text_length("does-not-exist", {}) == FALLBACK_MAX_TEXT_LENGTH

    def test_empty_provider_falls_back(self):
        assert _resolve_max_text_length("", {}) == FALLBACK_MAX_TEXT_LENGTH
        assert _resolve_max_text_length(None, {}) == FALLBACK_MAX_TEXT_LENGTH

    def test_case_insensitive(self):
        assert _resolve_max_text_length("OpenAI", {}) == 4096
        assert _resolve_max_text_length("  XAI  ", {}) == 15000

    # --- Overrides ---

    def test_override_wins(self):
        cfg = {"openai": {"max_text_length": 9999}}
        assert _resolve_max_text_length("openai", cfg) == 9999

    def test_override_zero_falls_through(self):
        # A broken/zero override must not disable truncation
        cfg = {"openai": {"max_text_length": 0}}
        assert _resolve_max_text_length("openai", cfg) == 4096

    def test_override_negative_falls_through(self):
        cfg = {"xai": {"max_text_length": -1}}
        assert _resolve_max_text_length("xai", cfg) == 15000

    def test_override_non_int_falls_through(self):
        cfg = {"minimax": {"max_text_length": "lots"}}
        assert _resolve_max_text_length("minimax", cfg) == 10000

    def test_override_bool_falls_through(self):
        # bool is technically an int; make sure we don't treat True as 1 char
        cfg = {"openai": {"max_text_length": True}}
        assert _resolve_max_text_length("openai", cfg) == 4096

    def test_missing_provider_section_uses_default(self):
        cfg = {"provider": "openai"}  # no "openai" key
        assert _resolve_max_text_length("openai", cfg) == 4096

    # --- ElevenLabs model-aware ---

    def test_elevenlabs_default_model_multilingual_v2(self):
        cfg = {"elevenlabs": {"model_id": "eleven_multilingual_v2"}}
        assert _resolve_max_text_length("elevenlabs", cfg) == 10000

    def test_elevenlabs_flash_v2_5_gets_40k(self):
        cfg = {"elevenlabs": {"model_id": "eleven_flash_v2_5"}}
        assert _resolve_max_text_length("elevenlabs", cfg) == 40000

    def test_elevenlabs_flash_v2_gets_30k(self):
        cfg = {"elevenlabs": {"model_id": "eleven_flash_v2"}}
        assert _resolve_max_text_length("elevenlabs", cfg) == 30000

    def test_elevenlabs_v3_gets_5k(self):
        cfg = {"elevenlabs": {"model_id": "eleven_v3"}}
        assert _resolve_max_text_length("elevenlabs", cfg) == 5000

    def test_elevenlabs_unknown_model_falls_back_to_provider_default(self):
        cfg = {"elevenlabs": {"model_id": "eleven_experimental_xyz"}}
        assert _resolve_max_text_length("elevenlabs", cfg) == PROVIDER_MAX_TEXT_LENGTH["elevenlabs"]

    def test_elevenlabs_override_beats_model_lookup(self):
        cfg = {"elevenlabs": {"model_id": "eleven_flash_v2_5", "max_text_length": 1000}}
        assert _resolve_max_text_length("elevenlabs", cfg) == 1000

    def test_elevenlabs_no_model_id_uses_default_model_mapping(self):
        # Falls back to DEFAULT_ELEVENLABS_MODEL_ID = eleven_multilingual_v2 -> 10000
        assert _resolve_max_text_length("elevenlabs", {}) == 10000

    def test_provider_config_not_a_dict(self):
        cfg = {"openai": "not-a-dict"}
        assert _resolve_max_text_length("openai", cfg) == 4096

    # --- Sanity: the table covers every provider listed in the schema ---

    def test_all_documented_providers_have_defaults(self):
        expected = {"edge", "openai", "xai", "minimax", "mistral",
                    "gemini", "elevenlabs", "neutts", "kittentts"}
        assert expected.issubset(PROVIDER_MAX_TEXT_LENGTH.keys())


class TestTextToSpeechToolTruncation:
    """End-to-end: verify the resolver actually drives the text_to_speech_tool
    truncation path rather than the old 4000-char global."""

    def test_openai_truncates_at_4096_not_4000(self, tmp_path, monkeypatch, caplog):
        import logging
        caplog.set_level(logging.WARNING, logger="tools.tts_tool")

        # 5000 chars -- over OpenAI's 4096 limit but under xAI's 15k
        text = "A" * 5000
        captured_text = {}

        def fake_openai(t, out, cfg):
            captured_text["text"] = t
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
        monkeypatch.setattr("tools.tts_tool._load_tts_config",
                            lambda: {"provider": "openai"})

        from tools.tts_tool import text_to_speech_tool
        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text=text, output_path=out))

        assert result["success"] is True
        # Should be truncated to 4096, not the old 4000
        assert len(captured_text["text"]) == 4096
        # And the warning should mention the provider
        assert any("openai" in rec.message.lower() for rec in caplog.records)

    def test_xai_accepts_much_longer_input(self, tmp_path, monkeypatch):
        # 12000 chars -- over old global 4000, under xAI's 15000
        text = "B" * 12000
        captured_text = {}

        def fake_xai(t, out, cfg):
            captured_text["text"] = t
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_xai_tts", fake_xai)
        monkeypatch.setattr("tools.tts_tool._load_tts_config",
                            lambda: {"provider": "xai"})

        from tools.tts_tool import text_to_speech_tool
        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text=text, output_path=out))

        assert result["success"] is True
        # xAI should accept the full 12000 chars
        assert len(captured_text["text"]) == 12000

    def test_user_override_is_respected(self, tmp_path, monkeypatch):
        # User says "cap openai at 100 chars" -- we must honor it
        text = "C" * 500
        captured_text = {}

        def fake_openai(t, out, cfg):
            captured_text["text"] = t
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
        monkeypatch.setattr("tools.tts_tool._load_tts_config",
                            lambda: {"provider": "openai",
                                     "openai": {"max_text_length": 100}})

        from tools.tts_tool import text_to_speech_tool
        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text=text, output_path=out))

        assert result["success"] is True
        assert len(captured_text["text"]) == 100
