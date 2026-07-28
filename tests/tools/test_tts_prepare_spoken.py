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

    def test_footer_bullets_removed(self):
        spoken = strip_nonspoken_blocks("Reply.\n" + self.FOOTER)
        assert "old_string" not in spoken
        assert "write_file" not in spoken

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

    def test_newlines_become_sentence_breaks(self):
        out = flatten_newlines_for_payload("Alpha\nBeta")
        assert out == "Alpha. Beta"

    def test_existing_punctuation_not_doubled(self):
        out = flatten_newlines_for_payload("Alpha.\nBeta!")
        assert ".." not in out
        assert "Alpha." in out and "Beta!" in out


class TestSharedCleanerWiring:
    """The ONE cleaner must be applied on every TTS entry path."""

    def test_tool_path_strips_think_blocks(self):
        from tools.tts_tool import _strip_markdown_for_tts

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
        from tools.tts_tool import _strip_markdown_for_tts

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
