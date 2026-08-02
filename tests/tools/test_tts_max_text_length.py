"""Tests for per-provider TTS input-character limits.

Replaces the old global ``MAX_TEXT_LENGTH = 4000`` cap that truncated every
provider at 4000 chars even though OpenAI allows 4096, xAI allows 15000,
MiniMax allows 10000, and ElevenLabs allows 5000-40000 depending on model.
"""

import json

import pytest


from tools.tts_tool import (
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


    # --- Overrides ---


    # --- ElevenLabs model-aware ---

    @pytest.mark.parametrize("configured_model", [None, "", "   "])
    def test_elevenlabs_sync_request_and_fallback_cap_share_nonempty_model_precedence(
        self,
        configured_model,
        tmp_path,
        monkeypatch,
    ):
        from tools import tts_tool

        captured_models = []

        class _TextToSpeech:
            @staticmethod
            def convert(**kwargs):
                captured_models.append(kwargs["model_id"])
                return iter([b"audio"])

        class _ElevenLabs:
            def __init__(self, **_kwargs):
                self.text_to_speech = _TextToSpeech()

        config = {"elevenlabs": {"model_id": configured_model}}
        monkeypatch.setattr(tts_tool, "_resolve_provider_key", lambda *_args: "test-key")
        monkeypatch.setattr(tts_tool, "_import_elevenlabs", lambda: _ElevenLabs)

        output_path = str(tmp_path / "sync.mp3")
        tts_tool._generate_elevenlabs("Fallback contract.", output_path, config)

        assert captured_models == [tts_tool.DEFAULT_ELEVENLABS_MODEL_ID]
        assert _resolve_max_text_length("elevenlabs", config) == 10000


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

        def fake_openai(t, out, cfg, **_kw):
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

        def fake_openai(t, out, cfg, **_kw):
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
