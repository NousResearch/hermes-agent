"""Unit tests for the shared TTS text cleaner (tools/tts_text_normalize).

Covers the consolidated preprocessing pipeline: <think> reasoning blocks
(#34213), emoji strip (#13311/#18598), file-mutation verifier footer
(#40772), newline flattening for newline-sensitive providers (#9004), and
the wiring of the ONE shared cleaner into both the text_to_speech tool
path and the voice-mode paths.
"""

import json

from tools.tts_text_normalize import (
    flatten_newlines_for_payload,
    prepare_spoken_text,
    strip_nonspoken_blocks,
)


class TestThinkBlockStrip:
    def test_think_block_removed(self):
        raw = "<think>\nsecret reasoning here\n</think>\nThe answer is 42."
        spoken = prepare_spoken_text(raw)
        assert "secret reasoning" not in spoken
        assert "42" in spoken

    def test_think_block_with_attributes_removed(self):
        raw = "<think budget=high>chain of thought</think>Visible."
        spoken = prepare_spoken_text(raw)
        assert "chain of thought" not in spoken
        assert "Visible" in spoken

    def test_unterminated_think_block_removed(self):
        raw = "Answer first. <think>\ntruncated reasoning stream"
        spoken = prepare_spoken_text(raw)
        assert "truncated reasoning" not in spoken
        assert "Answer first" in spoken

    def test_multiple_think_blocks(self):
        raw = "<think>a</think>one<think>b</think> two"
        spoken = strip_nonspoken_blocks(raw)
        assert "a" not in spoken.replace("one", "").replace("two", "")
        assert "one" in spoken and "two" in spoken


class TestVerifierFooterStrip:
    FOOTER = (
        "⚠️ File-mutation verifier: 2 file(s) were NOT modified this turn "
        "despite any wording above that may suggest otherwise. Run `git "
        "status` or `read_file` to confirm.\n"
        "  • `tools/foo.py` — [patch] old_string not found\n"
        "  • `bar.md` — [write_file] failed"
    )

    def test_footer_removed(self):
        raw = "I fixed the file.\n\n" + self.FOOTER
        spoken = prepare_spoken_text(raw)
        assert "File-mutation verifier" not in spoken
        assert "NOT modified" not in spoken
        assert "fixed the file" in spoken


    def test_text_without_footer_untouched(self):
        raw = "Just a normal reply about files."
        assert strip_nonspoken_blocks(raw).strip() == raw


class TestEmojiStrip:
    def test_emoji_removed(self):
        spoken = prepare_spoken_text("Done! 🎉🚀 All tests pass ✅")
        assert "🎉" not in spoken
        assert "🚀" not in spoken
        assert "✅" not in spoken
        assert "All tests pass" in spoken


class TestNewlineFlattening:
    def test_no_newlines_in_output(self):
        raw = "First line\nSecond line\n\nThird paragraph"
        spoken = prepare_spoken_text(raw)
        assert "\n" not in spoken
        assert "First line" in spoken
        assert "Third paragraph" in spoken


    def test_existing_punctuation_not_doubled(self):
        out = flatten_newlines_for_payload("Alpha.\nBeta!")
        assert ".." not in out
        assert "Alpha." in out and "Beta!" in out


class TestIdentifierNormalization:
    """Machine identifiers become speakable references (conversation, not dictation)."""

    def test_opaque_voice_id_becomes_tail_anchor(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        out = normalize_identifiers_for_tts("Saved voice 6u6JbqKdaQy89ENzLSju to config.")
        assert "6u6JbqKdaQy89ENzLSju" not in out
        assert "string ending in" in out
        assert "l s j u" in out  # last four characters, char-spoken

    def test_long_hex_becomes_hash_tail(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        out = normalize_identifiers_for_tts("Commit 72a3277cd7abc is the culprit.")
        assert "72a3277cd7abc" not in out
        assert "hash ending in" in out

    def test_uuid_becomes_hash_tail(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        out = normalize_identifiers_for_tts("Page 550e8400-e29b-41d4-a716-446655440000 moved.")
        assert "550e8400" not in out
        assert "hash ending in 0 0 0 0" in out

    def test_file_name_extension_becomes_words(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        out = normalize_identifiers_for_tts("See 1234567.png in the folder.")
        assert "1234567" not in out
        assert "PNG file" in out

    def test_meaningful_file_stem_is_kept(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        out = normalize_identifiers_for_tts("Wrote report-2026.pdf for you.")
        assert "report-2026 PDF file" in out

    def test_underscore_identifier_gets_pause_not_underscore(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        assert normalize_identifiers_for_tts("Updated TTSS_2026 today.") == "Updated TTSS 2026 today."

    def test_normal_words_dates_versions_untouched(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        text = "Version 3.2.0 shipped in 2026. NASA2026 too."
        assert normalize_identifiers_for_tts(text) == text

    def test_mixed_case_word_without_digit_untouched(self):
        from tools.tts_text_normalize import normalize_identifiers_for_tts

        assert "internationalization" in normalize_identifiers_for_tts("internationalization")

    def test_wired_into_prepare_spoken_text(self):
        spoken = prepare_spoken_text("Saved voice 6u6JbqKdaQy89ENzLSju and wrote log_2026.")
        assert "6u6JbqKdaQy89ENzLSju" not in spoken
        assert "l s j u" in spoken
        assert "log 2026" in spoken


class TestSharedCleanerWiring:
    """The ONE cleaner must be applied on every TTS entry path."""

    def test_tool_path_strips_think_blocks(self):
        from tools.tts_text_normalize import _strip_markdown_for_tts

        cleaned = _strip_markdown_for_tts("<think>hidden</think>**Loud** and clear 🎉")
        assert "hidden" not in cleaned
        assert "**" not in cleaned
        assert "🎉" not in cleaned
        assert "Loud and clear" in cleaned

    def test_tool_rejects_text_empty_after_cleanup(self):
        from tools.tts_tool import text_to_speech_tool

        result = json.loads(text_to_speech_tool(text="<think>only reasoning</think>"))
        assert result["success"] is False

    def test_streaming_helper_uses_shared_cleaner(self):
        from tools.tts_text_normalize import _strip_markdown_for_tts

        cleaned = _strip_markdown_for_tts("Temp is 14°C today\nand rising")
        assert "degrees Celsius" in cleaned
        assert "\n" not in cleaned

    def test_gateway_prepare_tts_text_strips_think_blocks(self):
        from gateway.config import Platform, PlatformConfig
        from gateway.platforms.base import BasePlatformAdapter

        class _DummyAdapter(BasePlatformAdapter):
            def __init__(self):
                super().__init__(
                    PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM
                )

            async def connect(self):
                return True

            async def disconnect(self):
                pass

            async def send(self, chat_id, content, **kwargs):
                raise AssertionError("not used")

            async def get_chat_info(self, chat_id):
                return {"id": chat_id, "type": "dm"}

        adapter = _DummyAdapter()
        spoken = adapter.prepare_tts_text("<think>plan</think>Hello there")
        assert "plan" not in spoken
        assert "Hello there" in spoken
