"""Tests for multi-match location listing in patch ambiguity errors."""

from tools.fuzzy_match import fuzzy_find_and_replace, _format_match_locations

class TestFormatMatchLocations:

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
