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
        # /v1/stt returns HTTP 400 for format=true without a language, so auto-detect drops it.
        assert "format" not in (captured["data"] or {}), \
            "format=true must not be sent without a language (xAI rejects it with HTTP 400)"

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


class TestExplicitlyEmptyLanguage:
    """``language: ""`` on a provider section means auto-detect, not "fall through" (#120100).

    The global ``stt.language`` defaults to "en"; before this, an operator who set
    ``stt.local.language: ""`` to opt back into Whisper auto-detection still got "en" pinned,
    because the empty string was skipped like an unset key.
    """

    def test_empty_provider_language_suppresses_global_default(self):
        cfg = {"language": "en", "local": {"language": ""}}
        assert _resolve_stt_language("local", cfg) is None

    def test_whitespace_only_provider_language_is_auto_detect(self):
        cfg = {"language": "en", "local": {"language": "   "}}
        assert _resolve_stt_language("local", cfg) is None

    def test_empty_alias_key_suppresses_global_default(self):
        cfg = {"language": "en", "local": {"language_code": ""}}
        assert _resolve_stt_language("local", cfg, extra_keys=("language_code",)) is None

    def test_absent_provider_language_still_falls_back_to_global(self):
        cfg = {"language": "en", "local": {}}
        assert _resolve_stt_language("local", cfg) == "en"

    def test_explicit_null_provider_language_falls_back_to_global(self):
        cfg = {"language": "en", "local": {"language": None}}
        assert _resolve_stt_language("local", cfg) == "en"

    def test_empty_provider_language_suppresses_env_default(self, monkeypatch):
        monkeypatch.setenv("HERMES_LOCAL_STT_LANGUAGE", "de")
        cfg = {"local": {"language": ""}}
        assert _resolve_stt_language("local", cfg) is None
