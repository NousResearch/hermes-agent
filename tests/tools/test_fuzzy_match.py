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
        assert err is not None
