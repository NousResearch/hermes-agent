"""Regression tests for tools.tts_tool._strip_markdown_for_tts().

Verifies that markdown emphasis stripping does NOT corrupt literal asterisks
in cron expressions, math, or other non-markdown content.

Mirrors the tests in tests/gateway/test_prepare_tts_text.py for the
sibling sanitizer used in the voice-reply gateway path.
"""

import pytest


class TestStripMarkdownForTts:
    """Test the _strip_markdown_for_tts asterisk-aware sanitization."""

    def _call(self, text: str) -> str:
        from tools.tts_tool import _strip_markdown_for_tts

        return _strip_markdown_for_tts(text)

    # --- Must strip: valid markdown emphasis ---

    def test_bold_stripped(self):
        assert self._call("**bold**") == "bold"

    def test_italic_stripped(self):
        assert self._call("*italic*") == "italic"

    def test_bold_with_surrounding_text(self):
        assert self._call("some **bold** text") == "some bold text"

    def test_italic_with_surrounding_text(self):
        assert self._call("some *italic* text") == "some italic text"

    def test_inline_code_stripped(self):
        assert self._call("`code`") == "code"

    def test_link_stripped_to_text(self):
        assert self._call("[click](https://example.com)") == "click"

    def test_header_stripped(self):
        assert self._call("## Header") == "Header"

    def test_list_item_stripped(self):
        assert self._call("- item") == "item"

    def test_code_block_stripped(self):
        assert self._call("```code```") == ""

    def test_url_stripped(self):
        assert self._call("visit https://example.com now") == "visit  now"

    # --- Must NOT strip: literal asterisks ---

    def test_cron_expression_preserved(self):
        assert self._call("cron: 30 8 * * 1-5") == "cron: 30 8 * * 1-5"

    def test_cron_every_minute_preserved(self):
        # _MD_LIST_ITEM strips leading "* " (valid markdown list marker)
        # so test with non-line-start context
        assert self._call("every minute: * * * * *") == "every minute: * * * * *"

    def test_math_multiplication_preserved(self):
        assert self._call("5 * 3 * 2 = 30") == "5 * 3 * 2 = 30"

    def test_single_multiplication_preserved(self):
        assert self._call("5 * 3 = 15") == "5 * 3 = 15"

    def test_asterisks_with_only_whitespace_inside(self):
        # _MD_LIST_ITEM strips leading "* " — test mid-line to isolate emphasis fix
        assert self._call("x * * * y") == "x * * * y"

    def test_literal_asterisk_between_words(self):
        assert self._call("foo * bar * baz") == "foo * bar * baz"

    def test_mixed_markdown_and_literal(self):
        result = self._call("**bold** and *italic* vs 5 * 3 = 15")
        assert result == "bold and italic vs 5 * 3 = 15"

    def test_empty_emphasis_not_stripped(self):
        assert self._call("**") == "**"
