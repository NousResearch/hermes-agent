"""Tests for per-provider TTS input-character limits.

With long-form chunking, text exceeding the provider cap is split into
ordered chunks instead of silently truncated. Each chunk is synthesized
separately and the results are combined or delivered as multiple files.
"""

import json


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


    # --- Sanity: the table covers every provider listed in the schema ---

    def test_all_documented_providers_have_defaults(self):
        expected = {"edge", "openai", "xai", "minimax", "mistral",
                    "gemini", "elevenlabs", "neutts", "kittentts"}
        assert expected.issubset(PROVIDER_MAX_TEXT_LENGTH.keys())


class TestTextToSpeechToolChunking:
    """End-to-end: verify the resolver drives text_to_speech_tool to split
    per-request chunks rather than the old 4000-char global truncation."""

    def test_openai_chunks_at_4096_without_dropping_text(self, tmp_path, monkeypatch):
        # 5000 chars -- over OpenAI's 4096 limit but under xAI's 15k
        text = "A" * 5000
        captured_text = []

        def fake_openai(t, out, cfg, **_kw):
            captured_text.append(t)
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        def fake_combine(paths, output_path, *, voice_compatible=False):
            with open(output_path, "wb") as destination:
                for path in paths:
                    with open(path, "rb") as source:
                        destination.write(source.read())
            return output_path

        monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
        monkeypatch.setattr("tools.tts_tool._concat_audio_files", fake_combine)
        monkeypatch.setattr("tools.tts_tool._load_tts_config",
                            lambda: {"provider": "openai"})

        from tools.tts_tool import text_to_speech_tool
        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text=text, output_path=out))

        assert result["success"] is True
        assert [len(chunk) for chunk in captured_text] == [4096, 904]
        assert "".join(captured_text) == text
        assert result["chunk_count"] == 2

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
        # xAI should accept the full 12000 chars in a single chunk
        assert len(captured_text["text"]) == 12000

    def test_user_override_is_respected(self, tmp_path, monkeypatch):
        # User says "cap openai at 100 chars" -- we must honor it
        text = "C" * 500
        captured_text = []

        def fake_openai(t, out, cfg, **_kw):
            captured_text.append(t)
            with open(out, "wb") as f:
                f.write(b"\x00")
            return out

        def fake_combine(paths, output_path, *, voice_compatible=False):
            with open(output_path, "wb") as destination:
                for path in paths:
                    with open(path, "rb") as source:
                        destination.write(source.read())
            return output_path

        monkeypatch.setattr("tools.tts_tool._generate_openai_tts", fake_openai)
        monkeypatch.setattr("tools.tts_tool._concat_audio_files", fake_combine)
        monkeypatch.setattr("tools.tts_tool._load_tts_config",
                            lambda: {"provider": "openai",
                                     "openai": {"max_text_length": 100}})

        from tools.tts_tool import text_to_speech_tool
        out = str(tmp_path / "out.mp3")
        result = json.loads(text_to_speech_tool(text=text, output_path=out))

        assert result["success"] is True
        assert all(len(chunk) <= 100 for chunk in captured_text)
        assert "".join(captured_text) == text
