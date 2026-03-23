"""Tests for the fuzzy matching module."""

from tools.fuzzy_match import fuzzy_find_and_replace


class TestExactMatch:
    def test_single_replacement(self):
        content = "hello world"
        new, count, err = fuzzy_find_and_replace(content, "hello", "hi")
        assert err is None
        assert count == 1
        assert new == "hi world"

    def test_no_match(self):
        content = "hello world"
        new, count, err = fuzzy_find_and_replace(content, "xyz", "abc")
        assert count == 0
        assert err is not None
        assert new == content

    def test_empty_old_string(self):
        new, count, err = fuzzy_find_and_replace("abc", "", "x")
        assert count == 0
        assert err is not None

    def test_identical_strings(self):
        new, count, err = fuzzy_find_and_replace("abc", "abc", "abc")
        assert count == 0
        assert "identical" in err

    def test_multiline_exact(self):
        content = "line1\nline2\nline3"
        new, count, err = fuzzy_find_and_replace(content, "line1\nline2", "replaced")
        assert err is None
        assert count == 1
        assert new == "replaced\nline3"


class TestWhitespaceDifference:
    def test_extra_spaces_match(self):
        content = "def  foo(  x,  y  ):"
        new, count, err = fuzzy_find_and_replace(content, "def foo( x, y ):", "def bar(x, y):")
        assert count == 1
        assert "bar" in new


class TestIndentDifference:
    def test_different_indentation(self):
        content = "    def foo():\n        pass"
        new, count, err = fuzzy_find_and_replace(content, "def foo():\n    pass", "def bar():\n    return 1")
        assert count == 1
        assert "bar" in new


class TestReplaceAll:
    def test_multiple_matches_without_flag_errors(self):
        content = "aaa bbb aaa"
        new, count, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=False)
        assert count == 0
        assert "Found 2 matches" in err

    def test_multiple_matches_with_flag(self):
        content = "aaa bbb aaa"
        new, count, err = fuzzy_find_and_replace(content, "aaa", "ccc", replace_all=True)
        assert err is None
        assert count == 2
        assert new == "ccc bbb ccc"


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
