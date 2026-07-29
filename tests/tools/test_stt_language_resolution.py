"""Tests for the unified STT language resolution (_resolve_stt_language).

Class-level contract: EVERY provider resolves its language hint through one
helper with the order:

    stt.<provider>.language > stt.language (global) > HERMES_LOCAL_STT_LANGUAGE > None

Regression coverage for the "STT transcribes the wrong language" issue class
(#55551, #50181 and siblings):
  - xAI no longer silently forces "en" when nothing is configured
  - the global ``stt.language`` key reaches every provider
  - per-provider language still wins over the global key
"""

from unittest.mock import patch

import pytest

from tools.transcription_tools import _resolve_stt_language


@pytest.fixture(autouse=True)
def _clear_lang_env(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_STT_LANGUAGE", raising=False)


class TestResolveSttLanguage:
    def test_provider_language_wins(self):
        cfg = {"language": "en", "groq": {"language": "he"}}
        assert _resolve_stt_language("groq", cfg) == "he"

    def test_global_language_fallback(self):
        cfg = {"language": "hu", "groq": {}}
        assert _resolve_stt_language("groq", cfg) == "hu"

    def test_global_language_reaches_provider_without_section(self):
        cfg = {"language": "uk"}
        for provider in ("local", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra"):
            assert _resolve_stt_language(provider, cfg) == "uk", provider

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "de")
        assert _resolve_stt_language("openai", {}) == "de"

    def test_auto_detect_when_nothing_set(self):
        assert _resolve_stt_language("xai", {}) is None

    def test_blank_strings_are_skipped(self):
        cfg = {"language": "  ", "groq": {"language": ""}}
        assert _resolve_stt_language("groq", cfg) is None

    def test_extra_keys_alias(self):
        cfg = {"elevenlabs": {"language_code": "spa"}}
        assert _resolve_stt_language("elevenlabs", cfg, extra_keys=("language_code",)) == "spa"

    def test_null_provider_section(self):
        # YAML `stt.groq: null` must not crash
        cfg = {"groq": None, "language": "fr"}
        assert _resolve_stt_language("groq", cfg) == "fr"

    def test_value_is_stripped(self):
        cfg = {"language": " ja "}
        assert _resolve_stt_language("local", cfg) == "ja"


class TestXaiNoForcedEnglish:
    """xAI previously defaulted language to "en" — auto-detect must be the default."""

    def test_no_language_sent_by_default(self, tmp_path, monkeypatch):
        import tools.transcription_tools as tt
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"x")
        captured = {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"text": "hola", "language": "es", "duration": 1.0}

        import requests as _requests

        def fake_post(url, **kwargs):
            captured["data"] = kwargs.get("data")
            return _Resp()

        monkeypatch.setattr(_requests, "post", fake_post)
        with patch.object(tt, "_load_stt_config", return_value={}), \
             patch("tools.xai_http.resolve_xai_http_credentials",
                   return_value={"api_key": "xai-test", "base_url": "https://api.x.ai/v1"}):
            result = tt._transcribe_xai(str(audio), "grok-stt")

        assert result["success"] is True
        assert "language" not in (captured["data"] or {}), \
            "xAI must auto-detect when no language is configured (was forced to 'en')"

    def test_global_language_reaches_xai(self, tmp_path, monkeypatch):
        import tools.transcription_tools as tt
        audio = tmp_path / "a.ogg"
        audio.write_bytes(b"x")
        captured = {}

        class _Resp:
            status_code = 200

            @staticmethod
            def json():
                return {"text": "ok", "language": "he", "duration": 1.0}

        import requests as _requests

        monkeypatch.setattr(
            _requests, "post",
            lambda url, **kw: captured.update(data=kw.get("data")) or _Resp(),
        )
        with patch.object(tt, "_load_stt_config", return_value={"language": "he"}), \
             patch("tools.xai_http.resolve_xai_http_credentials",
                   return_value={"api_key": "xai-test", "base_url": "https://api.x.ai/v1"}):
            result = tt._transcribe_xai(str(audio), "grok-stt")

        assert result["success"] is True
        assert captured["data"]["language"] == "he"
