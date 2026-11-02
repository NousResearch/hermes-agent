"""Unit tests for the shared TTS text cleaner (tools/tts_text_normalize).

Covers the consolidated preprocessing pipeline: <think> reasoning blocks
(#34213), emoji strip (#13311/#18598), file-mutation verifier footer
(#40772), newline flattening for newline-sensitive providers (#9004), and
the wiring of the ONE shared cleaner into both the text_to_speech tool
path and the voice-mode paths.
"""

import json

import pytest

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


class TestMoneyScale:
    """A scale after an amount is spoken as a word before the currency: $380B -> "380 billion dollars"."""

    @pytest.mark.parametrize(("raw", "spoken"), [
        ("max net $163 (was $380B paper)", "max net 163 dollars (was 380 billion dollars paper)"),
        ("cap is $1.5M", "cap is 1.5 million dollars"),
        ("about $50k", "about 50 thousand dollars"),
        ("budget $1,500K total", "budget 1,500 thousand dollars total"),
        ("valued at $3b", "valued at 3 billion dollars"),
        ("revenue of $2.3T", "revenue of 2.3 trillion dollars"),
        ("a $2bn fund", "a 2 billion dollars fund"),
        ("raised $5mm", "raised 5 million dollars"),
        ("$5M-$10M range", "5 million dollars-10 million dollars range"),
        ("raised $2.5 million last year", "raised 2.5 million dollars last year"),
        ("a $3 billion deal", "a 3 billion dollars deal"),
        ("a $2.5-million grant", "a 2.5 million dollars grant"),
        ("only $4 Thousand", "only 4 thousand dollars"),
    ])
    def test_dollar_amounts(self, raw, spoken):
        assert prepare_spoken_text(raw) == spoken

    @pytest.mark.parametrize(("raw", "spoken"), [
        ("US$2M deal", "2 million US dollars deal"),
        ("A$50k fee", "50 thousand Australian dollars fee"),
        ("NZ$3.5m budget", "3.5 million New Zealand dollars budget"),
        ("€2M grant", "2 million euros grant"),
        ("£3k fine", "3 thousand pounds fine"),
        ("€1.5 billion", "1.5 billion euros"),
    ])
    def test_other_currencies_keep_their_name(self, raw, spoken):
        assert prepare_spoken_text(raw) == spoken

    @pytest.mark.parametrize(("raw", "spoken"), [
        ("$10k/month", "10 thousand dollars per month"),
        ("$5M/year", "5 million dollars per year"),
    ])
    def test_rate_after_a_scale_reads_per(self, raw, spoken):
        assert prepare_spoken_text(raw) == spoken

    @pytest.mark.parametrize(("raw", "spoken"), [
        ("anchor $612.80 sane", "anchor 612.80 dollars sane"),
        ("$1,234.56 total", "1,234.56 dollars total"),
        ("$5/month", "5 dollars per month"),
        ("A$50, then", "50 Australian dollars, then"),
        ("costs $5 and the room is 3m wide", "costs 5 dollars and the room is 3 metres wide"),
        ("$20 k-pop", "20 dollars k-pop"),
        ("$5 t-shirt", "5 dollars t-shirt"),
    ])
    def test_plain_amounts_and_units_are_unchanged(self, raw, spoken):
        assert prepare_spoken_text(raw) == spoken

    @pytest.mark.parametrize("raw", ["was $380B", "US$2M deal", "$10k/month", "raised $2.5 million"])
    def test_second_pass_changes_nothing(self, raw):
        # Auto-TTS cleans the text in the gateway and again inside text_to_speech_tool.
        once = prepare_spoken_text(raw)
        assert prepare_spoken_text(once) == once

    def test_a_scale_word_on_the_next_line_is_not_part_of_the_amount(self):
        spoken = prepare_spoken_text("raised $5\nmillion people watched")
        assert "5 dollars" in spoken
        assert "million dollars" not in spoken


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
