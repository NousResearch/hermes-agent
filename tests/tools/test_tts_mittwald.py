"""Tests for the mittwald AI Hosting TTS provider (Qwen3-TTS).

``_generate_mittwald_tts`` resolves credentials/voice/language and delegates to
``_generate_openai_tts``. The provider-specific part is the ``language`` field: mittwald
takes a word form ("German"), not the ISO code the shared OpenAI path would send as
``lang_code`` — a mismatch the endpoint answers with HTTP 400.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "test-key")
    yield


@pytest.fixture
def captured_speech(monkeypatch):
    """Capture the client kwargs and ``audio.speech.create`` payload."""
    captured: dict = {}

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None, **_kwargs):
            captured["api_key"] = api_key
            captured["base_url"] = base_url

            def _create(**kwargs):
                captured["create_kwargs"] = kwargs
                return MagicMock(stream_to_file=MagicMock())

            self.audio = MagicMock(speech=MagicMock(create=_create))

        def close(self):
            pass

    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: _FakeClient)
    return captured


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
    monkeypatch.delenv("MITTWALD_AI_API_KEY", raising=False)
    from tools.tts_tool import _generate_mittwald_tts
    with pytest.raises(ValueError, match="MITTWALD_LLM_API_KEY not set"):
        _generate_mittwald_tts("hi", str(tmp_path / "out.mp3"), {})


def test_defaults_are_a_hosted_model_and_a_real_voice(tmp_path, captured_speech):
    """``voice`` is mandatory server-side (no default), so the handler always sends one."""
    from tools.tts_tool import _generate_mittwald_tts
    from tools.tts_tool_openai import MITTWALD_TTS_VOICES

    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"), {})

    assert captured_speech["base_url"] == "https://llm.aihosting.mittwald.de/v1"
    assert captured_speech["api_key"] == "test-key"
    kwargs = captured_speech["create_kwargs"]
    assert kwargs["model"] == "Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert kwargs["voice"] in MITTWALD_TTS_VOICES
    assert kwargs["response_format"] == "mp3"
    # No language configured → no language field at all, so the server default applies.
    assert "extra_body" not in kwargs


def test_provider_wide_base_url_uses_the_env_accessor(monkeypatch, tmp_path, captured_speech):
    from tools import tts_tool
    from tools.tts_tool import _generate_mittwald_tts

    monkeypatch.setattr(
        tts_tool, "get_env_value",
        lambda name: "https://proxy.example/v1/" if name == "MITTWALD_BASE_URL" else None)

    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"), {})

    assert captured_speech["base_url"] == "https://proxy.example/v1"


def test_ai_api_key_alias_resolves_after_the_primary(monkeypatch, tmp_path, captured_speech):
    monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MITTWALD_AI_API_KEY", "alias-key")
    from tools.tts_tool import _generate_mittwald_tts

    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"), {})
    assert captured_speech["api_key"] == "alias-key"

    monkeypatch.setenv("MITTWALD_LLM_API_KEY", "primary-key")
    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"), {})

    assert captured_speech["api_key"] == "primary-key"


def test_iso_language_is_mapped_to_the_word_form(tmp_path, captured_speech):
    from tools.tts_tool import _generate_mittwald_tts

    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"),
                           {"mittwald": {"language": "de", "voice": "serena"}})

    kwargs = captured_speech["create_kwargs"]
    assert kwargs["voice"] == "serena"
    assert kwargs["extra_body"] == {"language": "German"}
    # The shared OpenAI spelling would be rejected here.
    assert "lang_code" not in kwargs["extra_body"]


def test_word_form_language_passes_through(tmp_path, captured_speech):
    from tools.tts_tool import _generate_mittwald_tts

    _generate_mittwald_tts("ni hao", str(tmp_path / "out.mp3"),
                           {"mittwald": {"language": "Sichuan_Dialect"}})

    assert captured_speech["create_kwargs"]["extra_body"] == {"language": "Sichuan_Dialect"}


def test_global_tts_language_applies_when_the_provider_block_is_silent(tmp_path, captured_speech):
    from tools.tts_tool import _generate_mittwald_tts

    _generate_mittwald_tts("hello", str(tmp_path / "out.mp3"), {"language": "en"})

    assert captured_speech["create_kwargs"]["extra_body"] == {"language": "English"}


def test_openai_language_config_never_leaks_in(tmp_path, captured_speech):
    """``tts.openai.language`` belongs to the OpenAI provider; sending its ``lang_code``
    shape to mittwald would be a 400."""
    from tools.tts_tool import _generate_mittwald_tts

    _generate_mittwald_tts("hallo", str(tmp_path / "out.mp3"),
                           {"openai": {"language": "de"}, "mittwald": {}})

    assert "extra_body" not in captured_speech["create_kwargs"]


class TestMittwaldLanguageMapping:
    @pytest.mark.parametrize("value,expected", [
        ("de", "German"), ("EN", "English"), ("zh", "Chinese"), ("pt", "Portuguese"),
        ("German", "German"), ("german", "German"), ("Beijing_Dialect", "Beijing_Dialect"),
        ("", None), (None, None), ("  ", None)])
    def test_mapping(self, value, expected):
        from tools.tts_tool_openai import _mittwald_tts_language

        assert _mittwald_tts_language(value) == expected

    def test_unknown_language_warns_but_is_still_sent(self, caplog):
        from tools.tts_tool_openai import _mittwald_tts_language

        with caplog.at_level("WARNING", logger="tools.tts_tool"):
            assert _mittwald_tts_language("Klingon") == "Klingon"
        assert "Klingon" in caplog.text

    def test_documented_values_are_all_accepted(self):
        from tools.tts_tool_openai import MITTWALD_TTS_LANGUAGES, _mittwald_tts_language

        for language in MITTWALD_TTS_LANGUAGES:
            assert _mittwald_tts_language(language) == language


class TestMittwaldWiring:
    def test_requirements_follow_the_explicit_provider(self, monkeypatch):
        from tools import tts_tool

        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "mittwald", "mittwald": {}})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
        assert tts_tool.check_tts_requirements() is True

    def test_requirements_fail_without_a_key(self, monkeypatch):
        from tools import tts_tool

        monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
        monkeypatch.delenv("MITTWALD_AI_API_KEY", raising=False)
        monkeypatch.setattr(tts_tool, "_load_tts_config",
                            lambda: {"provider": "mittwald", "mittwald": {}})
        monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
        assert tts_tool.check_tts_requirements() is False

    def test_name_is_reserved_for_the_builtin(self):
        from agent.tts_registry import _BUILTIN_NAMES
        from tools.tts_command_provider import BUILTIN_TTS_PROVIDERS

        assert "mittwald" in BUILTIN_TTS_PROVIDERS
        assert "mittwald" in _BUILTIN_NAMES

    def test_ogg_output_needs_no_ffmpeg(self):
        """``response_format=opus`` returns audio/ogg, so voice bubbles skip transcoding."""
        from tools.tts_tool import _NATIVE_OPUS_PROVIDERS

        assert "mittwald" in _NATIVE_OPUS_PROVIDERS


def test_requirements_accept_the_alias_key(monkeypatch):
    """The availability probe has to agree with the handler: a user who set only the alias
    would otherwise see the provider reported as unavailable."""
    from tools import tts_tool

    monkeypatch.delenv("MITTWALD_LLM_API_KEY", raising=False)
    monkeypatch.setenv("MITTWALD_AI_API_KEY", "alias-key")
    monkeypatch.setattr(tts_tool, "_load_tts_config",
                        lambda: {"provider": "mittwald", "mittwald": {}})
    monkeypatch.setattr(tts_tool, "_import_openai_client", lambda: object)
    assert tts_tool.check_tts_requirements() is True
