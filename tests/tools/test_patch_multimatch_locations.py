"""Tests for multi-match location listing in patch ambiguity errors."""

import json

from tools.fuzzy_match import fuzzy_find_and_replace, _format_match_locations


class TestFormatMatchLocations:
    def test_lists_line_numbers_and_snippets(self):
        content = "alpha = 1\nvalue = compute()\nbeta = 2\nvalue = compute()\n"
        matches = [(10, 27), (38, 55)]
        out = _format_match_locations(content, matches)
        assert "L2: value = compute()" in out
        assert "L4: value = compute()" in out

    def test_caps_at_five_with_overflow_note(self):
        line = "x = do_thing()\n"
        content = line * 9
        matches = []
        pos = 0
        for _ in range(9):
            matches.append((pos, pos + len(line) - 1))
            pos += len(line)
        out = _format_match_locations(content, matches)
        assert out.count("L") == 5
        assert "... and 4 more" in out

    def test_long_lines_truncated(self):
        content = "y = " + "z" * 200 + "\n"
        out = _format_match_locations(content, [(0, 5)])
        assert "..." in out
        assert len(out.splitlines()[0]) < 100


class TestMultiMatchErrorIncludesLocations:
    def test_ambiguous_replace_lists_locations(self):
        content = (
            "def block_a(v):\n    value = value + 1\n    return v\n\n"
            "def block_b(v):\n    value = value + 1\n    return v\n"
        )
        _new, count, _strategy, error = fuzzy_find_and_replace(
            content, "    value = value + 1", "    value = value + 2"
        )
        assert count == 0
        assert "Found 2 matches" in error
        assert "L2:" in error
        assert "L6:" in error

    def test_replace_all_unaffected(self):
        content = "a = old_value_here\nb = old_value_here\n"
        new, count, _strategy, error = fuzzy_find_and_replace(
            content, "old_value_here", "new_value_here", replace_all=True
        )
        assert error is None
        assert count == 2
        assert "old_value_here" not in new

    def test_unique_match_unaffected(self):
        content = "only_one = special_marker_line\nother = thing\n"
        new, count, _strategy, error = fuzzy_find_and_replace(
            content, "special_marker_line", "replacement_marker"
        )
        assert error is None
        assert count == 1
