"""Tests for the fuzzy matching module."""

from tools.fuzzy_match import fuzzy_find_and_replace


class TestExactMatch:
    def test_single_replacement(self):
        content = "hello world"
        new, count, _, err = fuzzy_find_and_replace(content, "hello", "hi")
        assert err is None
        assert count == 1
        assert new == "hi world"

    def test_no_match(self):
        content = "hello world"
        new, count, _, err = fuzzy_find_and_replace(content, "xyz", "abc")
        assert count == 0
        assert err is not None
        assert new == content

    def test_empty_old_string(self):
        new, count, _, err = fuzzy_find_and_replace("abc", "", "x")
        assert count == 0
        assert err is not None

    def test_identical_strings(self):
        new, count, _, err = fuzzy_find_and_replace("abc", "abc", "abc")
        assert count == 0
        assert "identical" in err

    def test_multiline_exact(self):
        content = "line1\nline2\nline3"
        new, count, _, err = fuzzy_find_and_replace(content, "line1\nline2", "replaced")
        assert err is None
        assert count == 1
        assert new == "replaced\nline3"


class TestWhitespaceDifference:
    def test_extra_spaces_match(self):
        content = "def  foo(  x,  y  ):"
        new, count, _, err = fuzzy_find_and_replace(content, "def foo( x, y ):", "def bar(x, y):")
        assert count == 1
        assert "bar" in new


class TestIndentDifference:
    def test_different_indentation(self):
        content = "    def foo():\n        pass"
        new, count, _, err = fuzzy_find_and_replace(content, "def foo():\n    pass", "def bar():\n    return 1")
        assert count == 1
        assert "bar" in new


class TestReplaceAll:
    def test_multiple_matches_without_flag_errors(self):
        content = "aaa bbb aaa"
        new, count, _, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=False)
        assert count == 0
        assert "Found 2 matches" in err

    def test_multiple_matches_with_flag(self):
        content = "aaa bbb aaa"
        new, count, _, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=True)
        assert err is None
        assert count == 2
        assert new == "ccc bbb ccc"


class TestUnicodeNormalized:
    """Tests for the unicode_normalized strategy (Bug 5)."""

    def test_em_dash_matched(self):
        """Em-dash in content should match ASCII '--' in pattern."""
        content = "return value\u2014fallback"
        new, count, strategy, err = fuzzy_find_and_replace(
            content, "return value--fallback", "return value or fallback"
        )
        assert count == 1, f"Expected match via unicode_normalized, got err={err}"
        assert strategy == "unicode_normalized"
        assert "return value or fallback" in new

    def test_smart_quotes_matched(self):
        """Smart double quotes in content should match straight quotes in pattern."""
        content = 'print(\u201chello\u201d)'
        new, count, strategy, err = fuzzy_find_and_replace(
            content, 'print("hello")', 'print("world")'
        )
        assert count == 1, f"Expected match via unicode_normalized, got err={err}"
        assert "world" in new

    def test_no_unicode_skips_strategy(self):
        """When content and pattern have no Unicode variants, strategy is skipped."""
        content = "hello world"
        # Should match via exact, not unicode_normalized
        new, count, strategy, err = fuzzy_find_and_replace(content, "hello", "hi")
        assert count == 1
        assert strategy == "exact"


class TestBlockAnchorThreshold:
    """Tests for the raised block_anchor threshold (Bug 4)."""

    def test_high_similarity_matches(self):
        """A block with >50% middle similarity should match."""
        content = "def foo():\n    x = 1\n    y = 2\n    return x + y\n"
        pattern = "def foo():\n    x = 1\n    y = 9\n    return x + y"
        new, count, strategy, err = fuzzy_find_and_replace(content, pattern, "def foo():\n    return 0\n")
        # Should match via block_anchor or earlier strategy
        assert count == 1

    def test_completely_different_middle_does_not_match(self):
        """A block where only first+last lines match but middle is completely different
        should NOT match under the raised 0.50 threshold."""
        content = (
            "class Foo:\n"
            "    completely = 'unrelated'\n"
            "    content = 'here'\n"
            "    nothing = 'in common'\n"
            "    pass\n"
        )
        # Pattern has same first/last lines but completely different middle
        pattern = (
            "class Foo:\n"
            "    x = 1\n"
            "    y = 2\n"
            "    z = 3\n"
            "    pass"
        )
        new, count, strategy, err = fuzzy_find_and_replace(content, pattern, "replaced")
        # With threshold=0.50, this near-zero-similarity middle should not match
        assert count == 0, (
            f"Block with unrelated middle should not match under threshold=0.50, "
            f"but matched via strategy={strategy}"
        )


class TestStrategyNameSurfaced:
    """Tests for the strategy name in the 4-tuple return (Bug 6)."""

    def test_exact_strategy_name(self):
        new, count, strategy, err = fuzzy_find_and_replace("hello", "hello", "world")
        assert strategy == "exact"
        assert count == 1

    def test_failed_match_returns_none_strategy(self):
        new, count, strategy, err = fuzzy_find_and_replace("hello", "xyz", "world")
        assert count == 0
        assert strategy is None


class TestEscapeDriftGuard:
    """Tests for the escape-drift guard that catches bash/JSON serialization
    artifacts where an apostrophe gets prefixed with a spurious backslash
    in tool-call transport.
    """

    def test_drift_blocked_apostrophe(self):
        """File has ', old_string and new_string both have \\' — classic
        tool-call drift. Guard must block with a helpful error instead of
        writing \\' literals into source code."""
        content = "x = \"hello there\"\n"
        # Simulate transport-corrupted old_string and new_string where an
        # apostrophe-like context got prefixed with a backslash. The content
        # itself has no apostrophe, but both strings do — matching via
        # whitespace/anchor strategies would otherwise succeed.
        old_string = "x = \"hello there\" # don\\'t edit\n"
        new_string = "x = \"hi there\" # don\\'t edit\n"
        # This particular pair won't match anything, so it exits via
        # no-match path. Build a case where a non-exact strategy DOES match.
        content = "line\n    x = 1\nline"
        old_string = "line\n  x = \\'a\\'\nline"
        new_string = "line\n  x = \\'b\\'\nline"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert count == 0
        assert err is not None and "Escape-drift" in err
        assert "backslash" in err.lower()
        assert new == content  # file untouched

    def test_drift_blocked_double_quote(self):
        """Same idea but with \\" drift instead of \\'."""
        content = 'line\n    x = 1\nline'
        old_string = 'line\n  x = \\"a\\"\nline'
        new_string = 'line\n  x = \\"b\\"\nline'
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert count == 0
        assert err is not None and "Escape-drift" in err

    def test_drift_allowed_when_file_genuinely_has_backslash_escapes(self):
        """If the file already contains \\' (e.g. inside an existing escaped
        string), the model is legitimately preserving it. Guard must NOT
        fire."""
        content = "line\n  x = \\'a\\'\nline"
        old_string = "line\n  x = \\'a\\'\nline"
        new_string = "line\n  x = \\'b\\'\nline"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None
        assert count == 1
        assert "\\'b\\'" in new

    def test_drift_allowed_on_exact_match(self):
        """Exact matches bypass the drift guard entirely — if the file
        really contains the exact bytes old_string specified, it's not
        drift."""
        content = "hello \\'world\\'"
        new, count, strategy, err = fuzzy_find_and_replace(
            content, "hello \\'world\\'", "hello \\'there\\'"
        )
        assert err is None
        assert count == 1
        assert strategy == "exact"

    def test_drift_allowed_when_adding_escaped_strings(self):
        """Model is adding new content with \\' that wasn't in the original.
        old_string has no \\', so guard doesn't fire."""
        content = "line1\nline2\nline3"
        old_string = "line1\nline2\nline3"
        new_string = "line1\nprint(\\'added\\')\nline2\nline3"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None
        assert count == 1
        assert "\\'added\\'" in new

    def test_no_drift_check_when_new_string_lacks_suspect_chars(self):
        """Fast-path: if new_string has no \\' or \\", guard must not
        fire even on fuzzy match."""
        content = "def foo():\n    pass"  # extra space ignored by line_trimmed
        old_string = "def foo():\n  pass"
        new_string = "def bar():\n  return 1"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None
        assert count == 1


class TestFindClosestLines:
    def setup_method(self):
        from tools.fuzzy_match import find_closest_lines
        self.find_closest_lines = find_closest_lines

    def test_finds_similar_line(self):
        content = "def foo():\n    pass\ndef bar():\n    return 1\n"
        result = self.find_closest_lines("def baz():", content)
        assert "def foo" in result or "def bar" in result

    def test_returns_empty_for_no_match(self):
        content = "completely different content here"
        result = self.find_closest_lines("xyzzy_no_match_possible_!!!", content)
        assert result == ""

    def test_returns_empty_for_empty_inputs(self):
        assert self.find_closest_lines("", "some content") == ""
        assert self.find_closest_lines("old string", "") == ""

    def test_includes_context_lines(self):
        content = "line1\nline2\ndef target():\n    pass\nline5\n"
        result = self.find_closest_lines("def target():", content)
        assert "target" in result

    def test_includes_line_numbers(self):
        content = "line1\nline2\ndef foo():\n    pass\n"
        result = self.find_closest_lines("def foo():", content)
        # Should include line numbers in format "N| content"
        assert "|" in result

