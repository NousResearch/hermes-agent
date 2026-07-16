"""Tests for TTS text preprocessing — _preprocess_tts_text and _get_tts_language.

Covers:
- Markdown stripping (delegated to _strip_markdown_for_tts)
- Localized emoji speech substitution (13 languages)
- Unicode emoji removal (residual emojis after speech map)
- Box-drawing / table pipe conversion
- MEDIA: tag removal
- Whitespace collapsing
- Language detection: explicit config, STT reuse, voice name extraction, fallback
- preprocess: false bypass
"""

import pytest
from unittest.mock import patch

from tools.tts_tool import (
    _preprocess_tts_text,
    _get_tts_language,
    _detect_lang_from_voice,
    _lookup_emoji_speech,
    _strip_markdown_for_tts,
)


# ============================================================================
# _preprocess_tts_text — basic normalization
# ============================================================================

class TestPreprocessBasic:
    def test_strips_bold(self):
        result = _preprocess_tts_text("This is **bold** text", lang="en")
        assert "bold" in result
        assert "**" not in result

    def test_strips_code_blocks(self):
        text = "Here:\n```python\nprint('hi')\n```\nDone."
        result = _preprocess_tts_text(text, lang="en")
        assert "print" not in result
        assert "Done" in result

    def test_strips_inline_code(self):
        result = _preprocess_tts_text("Run `pip install foo`", lang="en")
        assert "pip install foo" in result
        assert "`" not in result

    def test_strips_markdown_links(self):
        text = "See [the docs](https://example.com) for info"
        result = _preprocess_tts_text(text, lang="en")
        assert "the docs" in result
        assert "https://" not in result
        assert "[" not in result

    def test_strips_headers(self):
        result = _preprocess_tts_text("## Summary\nSome text", lang="en")
        assert "Summary" in result
        assert "##" not in result

    def test_removes_media_tags(self):
        text = "Here is the file MEDIA:/path/to/audio.mp3 done"
        result = _preprocess_tts_text(text, lang="en")
        assert "MEDIA:" not in result
        assert "/path/to/audio.mp3" not in result
        assert "done" in result

    def test_collapses_excess_whitespace(self):
        text = "Hello    world\n\n\n\n\nDone"
        result = _preprocess_tts_text(text, lang="en")
        assert "  " not in result  # no double spaces
        assert result.count("\n\n") == 0 or ". " in result

    def test_empty_string(self):
        assert _preprocess_tts_text("", lang="en") == ""

    def test_plain_text_unchanged(self):
        text = "Hello world this is a test"
        result = _preprocess_tts_text(text, lang="en")
        assert "Hello world" in result
        assert "test" in result


# ============================================================================
# Localized emoji speech substitution
# ============================================================================

class TestEmojiSpeech:
    def test_emoji_en(self):
        result = _preprocess_tts_text("✅ Done", lang="en")
        assert "done" in result.lower()
        assert "✅" not in result

    def test_emoji_ru(self):
        result = _preprocess_tts_text("✅ Готово", lang="ru")
        assert "готово" in result.lower()
        assert "✅" not in result

    def test_emoji_de(self):
        result = _preprocess_tts_text("✅ Fertig", lang="de")
        assert "fertig" in result.lower()
        assert "✅" not in result

    def test_emoji_ja(self):
        result = _preprocess_tts_text("✅ 完了", lang="ja")
        assert "完了" in result
        assert "✅" not in result

    def test_multiple_emojis(self):
        result = _preprocess_tts_text("✅ Updated to v0.8 ⬆️", lang="en")
        assert "done" in result.lower()
        assert "update" in result.lower()
        assert "✅" not in result
        assert "⬆️" not in result

    def test_multiple_emojis_ru(self):
        result = _preprocess_tts_text("✅ Обновлено до v0.8 ⬆️", lang="ru")
        assert "готово" in result.lower()
        assert "обновление" in result.lower()
        assert "✅" not in result
        assert "⬆️" not in result

    def test_warning_emoji_en(self):
        result = _preprocess_tts_text("⚠️ Caution needed", lang="en")
        assert "warning" in result.lower()
        assert "⚠️" not in result

    def test_warning_emoji_ru(self):
        result = _preprocess_tts_text("⚠️ Внимание", lang="ru")
        assert "внимание" in result.lower()
        assert "⚠️" not in result

    def test_silent_emoji_removed(self):
        """Emojis with no spoken equivalent are silently removed."""
        result = _preprocess_tts_text("Hello 👾 world", lang="en")
        assert "👾" not in result
        assert "Hello" in result
        assert "world" in result

    def test_unknown_emoji_removed(self):
        """Emojis not in the map at all are removed by the Unicode range regex."""
        result = _preprocess_tts_text("Test 🦄 done", lang="en")
        assert "🦄" not in result
        assert "Test" in result
        assert "done" in result


# ============================================================================
# Box-drawing / table pipes
# ============================================================================

class TestPseudoGraphics:
    def test_table_pipes_to_spaces(self):
        result = _preprocess_tts_text("| Name | Value |", lang="en")
        assert "|" not in result
        assert "Name" in result
        assert "Value" in result

    def test_box_drawing_chars_removed(self):
        text = "Line 1 ─── Line 2"
        result = _preprocess_tts_text(text, lang="en")
        assert "─" not in result
        assert "Line 1" in result
        assert "Line 2" in result


# ============================================================================
# Language detection: _get_tts_language
# ============================================================================

class TestGetTtsLanguage:
    def test_explicit_override(self):
        config = {"language": "ru", "provider": "edge", "edge": {"voice": "en-US-AriaNeural"}}
        assert _get_tts_language(config) == "ru"

    def test_explicit_override_with_region(self):
        config = {"language": "pt-BR", "provider": "edge"}
        assert _get_tts_language(config) == "pt"

    def test_falls_back_to_stt_language(self):
        """When no tts.language, reuses stt.local.language."""
        config = {"provider": "edge", "edge": {"voice": "en-US-AriaNeural"}}
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {"stt": {"local": {"language": "de"}}}
            assert _get_tts_language(config) == "de"

    def test_falls_back_to_voice_name_edge(self):
        """When no tts.language and no stt language, extracts from Edge voice."""
        config = {"provider": "edge", "edge": {"voice": "ru-RU-SvetlanaNeural"}}
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            assert _get_tts_language(config) == "ru"

    def test_falls_back_to_voice_name_openai(self):
        config = {"provider": "openai", "openai": {"voice": "en-US-AriaNeural"}}
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            assert _get_tts_language(config) == "en"

    def test_fallback_en(self):
        """No config, no STT, no voice → 'en'."""
        config = {"provider": "edge", "edge": {}}
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            assert _get_tts_language(config) == "en"

    def test_empty_language_string_falls_through(self):
        """tts.language: '' should NOT be used — should fall through to STT/voice."""
        config = {"language": "", "provider": "edge", "edge": {"voice": "ru-RU-SvetlanaNeural"}}
        with patch("hermes_cli.config.load_config") as mock_load:
            mock_load.return_value = {}
            assert _get_tts_language(config) == "ru"


# ============================================================================
# _detect_lang_from_voice
# ============================================================================

class TestDetectLangFromVoice:
    def test_edge_voice_ru(self):
        assert _detect_lang_from_voice("ru-RU-SvetlanaNeural") == "ru"

    def test_edge_voice_en(self):
        assert _detect_lang_from_voice("en-US-AriaNeural") == "en"

    def test_edge_voice_de(self):
        assert _detect_lang_from_voice("de-DE-KatjaNeural") == "de"

    def test_openai_short_name_defaults_en(self):
        assert _detect_lang_from_voice("alloy") == "en"

    def test_empty_string_defaults_en(self):
        assert _detect_lang_from_voice("") == "en"

    def test_none_defaults_en(self):
        assert _detect_lang_from_voice(None) == "en"


# ============================================================================
# _lookup_emoji_speech fallback chain
# ============================================================================

class TestLookupEmojiSpeech:
    def test_exact_lang_match(self):
        assert _lookup_emoji_speech("✅", "ru") == " готово "

    def test_fallback_to_en(self):
        """Unknown lang falls back to English."""
        assert _lookup_emoji_speech("✅", "xx") == " done "

    def test_silent_emoji(self):
        """Emojis marked as silent return just a space."""
        assert _lookup_emoji_speech("👾", "en") == " "

    def test_unknown_emoji(self):
        """Emojis not in the map return a space (removed)."""
        assert _lookup_emoji_speech("🦄", "en") == " "

    def test_lang_prefix_match(self):
        """pt-BR should match 'pt' in the speech map."""
        result = _lookup_emoji_speech("✅", "pt-BR")
        assert "feito" in result


# ============================================================================
# preprocess: false bypass
# ============================================================================

class TestPreprocessDisabled:
    def test_preprocess_false_still_strips_markdown(self):
        """When preprocess is false, text_to_speech_tool uses _strip_markdown_for_tts
        directly (tested elsewhere). Here we just verify _preprocess_tts_text
        itself still works — the toggle is applied at the call site."""
        # This is a sanity check that _preprocess_tts_text is callable
        result = _preprocess_tts_text("✅ Done", lang="en")
        assert "done" in result.lower()