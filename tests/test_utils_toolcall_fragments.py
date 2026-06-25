"""Tests for strip_partial_toolcall_fragments (live per-delta #14251 follow-up).

Qwen3-class models sometimes emit a partial ``<tool_call>``/``<function_call>``
XML opener as text content, then switch to native ``tool_calls`` mid-token.
That leaves a stray content delta such as ``"ool_call>"`` which must never reach
user-facing output.  The helper scrubs these leaked opener *tails*; its regex
mirrors the production fix exactly (single compiled pattern for search + sub, so
there is no search/sub asymmetry).
"""

from utils import strip_partial_toolcall_fragments


class TestLeakedFragmentsAreStripped:
    """The realistic leaked opener tails must be removed."""

    def test_pure_tool_call_fragment_becomes_empty(self):
        # The canonical bug: "<tool_call>" leaked as text, truncated to "ool_call>".
        assert strip_partial_toolcall_fragments("ool_call>") == ""

    def test_single_o_tool_call_fragment(self):
        assert strip_partial_toolcall_fragments("ol_call>") == ""

    def test_plural_tool_calls_fragment(self):
        assert strip_partial_toolcall_fragments("ool_calls>") == ""

    def test_function_call_fragment(self):
        # The "f" of "<function_call>" arrives in an earlier delta; the leaked
        # tail is "unction_call>".  This is exactly what the bug produces.
        assert strip_partial_toolcall_fragments("unction_call>") == ""

    def test_function_calls_plural_fragment(self):
        assert strip_partial_toolcall_fragments("unction_calls>") == ""

    def test_fragment_after_prose_keeps_prose(self):
        # Mixed delta: real prose + a trailing leaked fragment.  Only the
        # fragment is removed; the prose is preserved verbatim.
        assert strip_partial_toolcall_fragments("hello ool_call>") == "hello "

    def test_fragment_is_case_insensitive(self):
        assert strip_partial_toolcall_fragments("OOL_CALL>") == ""


class TestNormalContentUnchanged:
    """Legitimate prose must pass through byte-for-byte."""

    def test_plain_prose_unchanged(self):
        text = "Let me help you with that."
        assert strip_partial_toolcall_fragments(text) is text

    def test_prose_mentioning_tool_call_words_unchanged(self):
        # No trailing '>' tail, so nothing matches — prose about tool calls is safe.
        text = "I'll make a tool call to read the file."
        assert strip_partial_toolcall_fragments(text) == text

    def test_empty_string_unchanged(self):
        assert strip_partial_toolcall_fragments("") == ""

    def test_none_passes_through(self):
        # Defensive: callers may hand us None; it must not raise.
        assert strip_partial_toolcall_fragments(None) is None

    def test_word_starting_with_l_call_is_not_a_fragment(self):
        # The pattern requires at least one leading 'o' (o{1,2}l_call), so a bare
        # "l_call>" is NOT treated as a tool-call tail and is left untouched.
        # This documents the exact (production-proven) regex behavior.
        assert strip_partial_toolcall_fragments("l_call>") == "l_call>"
