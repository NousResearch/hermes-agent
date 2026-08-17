"""Tests for whitespace-visualized mismatch diagnosis in patch no-match hints."""

import json

import pytest

from tools.fuzzy_match import (
    _visualize_whitespace,
    find_closest_lines,
    fuzzy_find_and_replace,
)


class TestVisualizeWhitespace:
    def test_spaces_and_tabs_visualized(self):
        assert _visualize_whitespace("\t    x = 1") == "→····x = 1"

    def test_interior_whitespace_untouched(self):
        assert _visualize_whitespace("  a  b") == "··a  b"

    def test_no_leading_ws(self):
        assert _visualize_whitespace("plain") == "plain"


class TestWhitespaceDiagnosis:
    def test_whitespace_shaped_miss_shows_both_lines(self):
        # File uses tabs; the model sends 4 spaces AND a wrong second line so
        # every fuzzy strategy misses (content mismatch), but line 1 is a
        # whitespace-only difference worth diagnosing.
        content = "def f():\n\tvalue = compute_thing()\n\treturn value\n"
        old = "    value = compute_thing()\n    return wrong_name"
        hint = find_closest_lines(old, content)
        assert "Whitespace difference detected" in hint
        assert "→value = compute_thing()" in hint
        assert "····value = compute_thing()" in hint

    def test_content_miss_has_no_whitespace_block(self):
        content = "alpha = 1\nbeta = 2\n"
        old = "alpha = 999"
        hint = find_closest_lines(old, content)
        assert "Whitespace difference detected" not in hint

    def test_exact_line_present_no_ws_block(self):
        # anchor matches raw -> no whitespace diagnosis needed
        content = "  x = 1\n  y = 2\n"
        old = "  x = 1\n  z = 3"
        hint = find_closest_lines(old, content)
        assert "Whitespace difference detected" not in hint


class TestEndToEndHint:
    def test_no_match_error_carries_ws_diagnosis(self):
        # Second line's CONTENT differs (wrong call name) so all 9 fuzzy
        # strategies genuinely miss; first line differs only in whitespace,
        # so the hint should include the whitespace diagnosis.
        content = "class A:\n\tdef run(self):\n\t\tstart_engine()\n\t\treturn 1\n"
        old = "    def run(self):\n        boot_engine_wrong()\n        return 2"
        new = "    def run(self):\n        start_engine()\n        return 3"
        _c, count, _s, err = fuzzy_find_and_replace(content, old, new)
        assert count == 0
        from tools.fuzzy_match import format_no_match_hint
        hint = format_no_match_hint(err, count, old, content)
        assert "Did you mean" in hint
        assert "Whitespace difference detected" in hint
