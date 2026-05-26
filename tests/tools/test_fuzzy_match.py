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


class TestIndentationPreservation:
    """When a non-exact strategy matches, ``new_string`` should be re-indented
    so it lands at the file's actual indent depth — not at whatever indent the
    LLM happened to send in the tool args.  Without this fix the file gets a
    silently-broken indent level that may even still parse but is logically
    wrong."""

    def test_unindented_input_reindented_to_match_file(self):
        # File: 8-space-indented method body inside a class.
        content = (
            "class Calculator:\n"
            "    def add(self, a, b):\n"
            "        result = a + b\n"
            "        return result\n"
        )
        # LLM sends zero-indent old/new — common bug from frontier models
        # that "remember" code instead of reading it.
        old = "result = a + b\nreturn result"
        new = "result = a + b\nresult *= 2\nreturn result"
        out, count, strategy, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and count == 1
        assert strategy != "exact"  # must have gone through a fuzzy strategy
        # Every replaced line should be at 8-space indent.
        for marker in ("result = a + b", "result *= 2", "return result"):
            line = next(line for line in out.split("\n") if marker in line)
            indent = len(line) - len(line.lstrip())
            assert indent == 8, f"Expected 8-space indent for {marker!r}, got {indent}: {line!r}"
        # Resulting file must still be valid Python.
        import ast
        ast.parse(out)

    def test_dedent_at_start_anchors_to_file_base(self):
        # File: 2-space-indented function body.  LLM sends zero-indent
        # old/new where new_string contains a dedent (the new structure
        # adds a top-level class wrapper).  After re-indent, every line
        # of new_string should be anchored to the file's 2-space base.
        content = "  return 1\n  return 2\n"
        old = "return 1\nreturn 2"  # zero-indent — forces line_trimmed
        new = "class X:\n  return 99\n  return 100"
        out, count, strategy, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and count == 1
        assert strategy != "exact"
        lines = out.split("\n")
        # 'class X:' anchored to file's 2-space base.
        assert lines[0] == "  class X:", repr(lines[0])
        # Indented body lines lift to 4-space (file base + LLM's +2).
        assert lines[1] == "    return 99", repr(lines[1])
        assert lines[2] == "    return 100", repr(lines[2])

    def test_exact_match_no_reindent(self):
        # Exact strategy should be a pure passthrough — no shift logic
        # should touch the result.
        content = "    def foo():\n        return 1\n"
        old = "    def foo():\n        return 1"
        new = "    def foo():\n        return 2"
        out, count, strategy, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and strategy == "exact"
        assert out == "    def foo():\n        return 2\n"

    def test_llm_zero_indent_shifts_to_file_two_space(self):
        # LLM sent zero-indent old/new; file has 2-space indent.  The
        # re-indent shifts the whole replacement so 'def x()' lands at
        # 2-space and the body keeps its relative +2 from new_string.
        content = "  def x():\n    return 1\n"
        old = "def x():\n  return 1"
        new = "def x():\n  return 99"
        out, count, _, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and count == 1
        lines = out.strip("\n").split("\n")
        assert lines[0] == "  def x():"
        assert lines[1] == "    return 99"

    def test_indent_already_matches_passthrough(self):
        # When old_string's base indent already equals file_region's base
        # indent, _reindent_replacement returns new_string unchanged.
        # Verify with whitespace_normalized strategy (collapsed spaces).
        content = "  def  x(  ):\n    return 1\n"
        old = "  def x():\n    return 1"  # same base indent (2), different inner whitespace
        new = "  def x():\n    return 42"
        out, count, strategy, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and count == 1
        assert strategy != "exact"  # non-exact strategy matched
        # Body retains its 4-space indent (passthrough — no shift).
        assert "    return 42" in out

    def test_blank_lines_left_alone(self):
        # Blank lines in new_string should keep whatever whitespace they
        # had — we never strip or pad them.
        content = "    a = 1\n    b = 2\n"
        old = "a = 1\nb = 2"
        new = "a = 1\n\nb = 99"
        out, count, _, err = fuzzy_find_and_replace(content, old, new)
        assert err is None and count == 1
        # blank line is preserved (empty), indented lines anchored.
        lines = out.split("\n")
        assert lines[0] == "    a = 1"
        assert lines[1] == ""
        assert lines[2] == "    b = 99"


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

    def test_overlapping_matches_replace_all(self):
        """Overlapping occurrences must not produce double-counted matches.

        In "aaaa" the substring "aa" occurs at offsets 0 and 2 (non-overlapping).
        Before the fix, the scan advanced by 1 instead of len(pattern), yielding
        3 overlapping matches and a corrupt result.
        """
        new, count, _, err = fuzzy_find_and_replace("aaaa", "aa", "b", replace_all=True)
        assert err is None
        assert count == 2
        assert new == "bb"

    def test_overlapping_matches_without_flag_errors(self):
        """Overlapping occurrences should still report the non-overlapping count."""
        new, count, _, err = fuzzy_find_and_replace("aaaa", "aa", "b", replace_all=False)
        assert count == 0
        assert "2 matches" in err

    def test_overlapping_single_char_pattern(self):
        """Single-char pattern: 'aaa' replacing 'a' with 'b' must give 'bbb'."""
        new, count, _, err = fuzzy_find_and_replace("aaa", "a", "b", replace_all=True)
        assert err is None
        assert count == 3
        assert new == "bbb"

    def test_overlapping_with_surrounding_content(self):
        """Overlapping region embedded in larger content — non-overlapping parts
        must be preserved exactly."""
        content = "prefix aaaa suffix"
        new, count, _, err = fuzzy_find_and_replace(content, "aa", "b", replace_all=True)
        assert err is None
        assert count == 2
        assert new == "prefix bb suffix"

    def test_overlapping_multiline(self):
        """Overlapping short string across multi-line code content."""
        content = "x = 0\nx = 0\nx = 0\n"
        new, count, _, err = fuzzy_find_and_replace(
            content, "x = 0\n", "y = 1\n", replace_all=True
        )
        assert err is None
        assert count == 3
        assert new == "y = 1\ny = 1\ny = 1\n"

    def test_no_false_overlaps_when_pattern_is_longer(self):
        """When the pattern is longer than any possible self-overlap, there is
        nothing to regress — this documents the benign case."""
        new, count, _, err = fuzzy_find_and_replace(
            "foo bar foo bar", "foo bar", "baz", replace_all=True
        )
        assert err is None
        assert count == 2
        assert new == "baz baz"


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


class TestFormatNoMatchHint:
    """Gating tests for format_no_match_hint — the shared helper that decides
    whether a 'Did you mean?' snippet should be appended to an error.
    """

    def setup_method(self):
        from tools.fuzzy_match import format_no_match_hint
        self.fmt = format_no_match_hint

    def test_fires_on_could_not_find_with_match(self):
        """Classic no-match: similar content exists → hint fires."""
        content = "def foo():\n    pass\ndef bar():\n    pass\n"
        result = self.fmt(
            "Could not find a match for old_string in the file",
            0, "def baz():", content,
        )
        assert "Did you mean" in result
        assert "foo" in result or "bar" in result

    def test_silent_on_ambiguous_match_error(self):
        """'Found N matches' is not a missing-match failure — no hint."""
        content = "aaa bbb aaa\n"
        result = self.fmt(
            "Found 2 matches for old_string. Provide more context to make it unique, or use replace_all=True.",
            0, "aaa", content,
        )
        assert result == ""

    def test_silent_on_escape_drift_error(self):
        """Escape-drift errors are intentional blocks — hint would mislead."""
        content = "x = 1\n"
        result = self.fmt(
            "Escape-drift detected: old_string and new_string contain the literal sequence '\\\\''...",
            0, "x = \\'1\\'", content,
        )
        assert result == ""

    def test_silent_on_identical_strings(self):
        """old_string == new_string — hint irrelevant."""
        result = self.fmt(
            "old_string and new_string are identical",
            0, "foo", "foo bar\n",
        )
        assert result == ""

    def test_silent_when_match_count_nonzero(self):
        """If match succeeded, we shouldn't be in the error path — defense in depth."""
        result = self.fmt(
            "Could not find a match for old_string in the file",
            1, "foo", "foo bar\n",
        )
        assert result == ""

    def test_silent_on_none_error(self):
        """No error at all — no hint."""
        result = self.fmt(None, 0, "foo", "bar\n")
        assert result == ""

    def test_silent_when_no_similar_content(self):
        """Even for a valid no-match error, skip hint when nothing similar exists."""
        result = self.fmt(
            "Could not find a match for old_string in the file",
            0, "totally_unique_xyzzy_qux", "abc\nxyz\n",
        )
        assert result == ""


class TestEscapeNormalizedNewString:
    """Regression tests for unescaping common sequences in new_string when
    the matched region of the file contains real control characters.

    Issue #33733: LLMs overwhelmingly represent tabs as the two-character
    sequence ``\\t`` (backslash + t) in JSON tool-call arguments. When the
    file already contains real tab bytes (0x09), writing new_string
    verbatim leaves literal ``\\t`` characters and corrupts the file.

    The fix unescapes ``\\t`` -> tab and ``\\r`` -> CR in new_string when
    the matched file region actually contains those control characters,
    regardless of which match strategy fired. ``\\n`` is excluded because
    newlines serialize correctly through JSON.
    """

    def test_tab_in_new_string_unescaped_under_escape_normalized(self):
        """File has real tab, model sends literal \\t in BOTH old and new.

        Match strategy is ``escape_normalized``.
        """
        content = "def hello():\n\tprint(\"before\")\n"
        old_string = "def hello():\n\\tprint(\"before\")\n"
        new_string = "def hello():\n\\tprint(\"after\")\n"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        assert strategy == "escape_normalized"
        assert "\tprint(\"after\")" in new
        assert "\\t" not in new

    def test_tab_in_new_string_unescaped_under_exact(self):
        """File has real tab, old_string has real tab too (matches via
        ``exact``), but new_string still arrives with literal ``\\t``.

        This is the issue's headline reproduction — the previous fix that
        gated on ``strategy_name == "escape_normalized"`` missed this case.
        """
        content = "def hello():\n\tprint(\"before\")\n"
        old_string = "\tprint(\"before\")"           # real tab
        new_string = "\\tprint(\"after\")"           # literal backslash + t
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        assert strategy == "exact"
        assert "\tprint(\"after\")" in new
        assert "\\t" not in new

    def test_carriage_return_in_new_string_unescaped(self):
        """File has real CR, model sends literal \\r in new_string."""
        content = "line1\r\nline2\r\n"
        old_string = "line1\\r\\nline2\\r\\n"
        new_string = "replaced\\r\\n"
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        assert strategy == "escape_normalized"
        assert "replaced\r" in new

    def test_newline_in_new_string_NOT_unescaped(self):
        """``\\n`` is intentionally left alone — newlines serialize correctly
        through JSON, and unescaping would corrupt source-code escape
        sequences far more often than help.
        """
        content = "line1\nline2\n"
        old_string = "line1\nline2"
        new_string = "alpha\\nbeta"                 # literal backslash + n
        new, count, _, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        # The literal two-character sequence ``\n`` must survive verbatim.
        assert "alpha\\nbeta" in new
        # And there should be no real newline added where ``\\n`` sat.
        assert "alpha\nbeta" not in new

    def test_mixed_tab_and_newline_only_tab_unescaped(self):
        """When new_string contains both \\t and \\n, only \\t is converted."""
        content = "def foo():\n\tpass\n"
        old_string = "def foo():\n\tpass\n"
        new_string = "def bar():\\n\\treturn 1\\n"
        new, count, _, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        # \t -> real tab
        assert "\treturn 1" in new
        assert "\\t" not in new
        # \n preserved as literal backslash-n
        assert "\\n" in new

    def test_exact_match_preserves_literal_backslash_t_in_string_literal(self):
        """If the matched region of the file does NOT contain a real tab,
        new_string's literal ``\\t`` is preserved — the file genuinely uses
        a backslash-t sequence (e.g. a Python source line ``sep = "\\t"``).
        """
        content = 'sep = "\\t"\n'                   # source contains backslash + t
        old_string = 'sep = "\\t"\n'
        new_string = 'sep = "\\tab"\n'              # still backslash + t literal
        new, count, strategy, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None, f"Unexpected error: {err}"
        assert count == 1
        assert strategy == "exact"
        # File still has the literal two-char ``\t`` — no tab byte injected.
        assert 'sep = "\\tab"' in new
        assert "\t" not in new

    def test_no_escape_sequences_passthrough(self):
        """When new_string has no \\t or \\r, the helper is a no-op."""
        content = "def foo():\n    return 1\n"
        old_string = "def foo():\n    return 1\n"
        new_string = "def foo():\n    return 2\n"
        new, count, _, err = fuzzy_find_and_replace(content, old_string, new_string)
        assert err is None
        assert count == 1
        assert "return 2" in new

class TestMapNormalizedNoOverlap:
    """Regression tests for trailing-whitespace expansion causing overlapping
    matches in _map_normalized_positions.  The expansion was removed because
    it greedily consumed spaces/tabs beyond the matched region, making
    adjacent matches overlap and corrupting files under replace_all=True.
    """

    def test_whitespace_normalized_no_trailing_spillover(self):
        """Adjacent whitespace-normalized matches must not overlap in original space."""
        content = "a  a  a  a"
        new, count, strat, err = fuzzy_find_and_replace(
            content, "a a", "BB", replace_all=True
        )
        assert err is None, err
        assert count == 2
        assert new == "BB  BB"

    def test_single_match_preserves_trailing_spaces(self):
        """Trailing spaces after a matched region must NOT be swallowed."""
        content = "foo   bar baz"
        new, count, strat, err = fuzzy_find_and_replace(
            content, "foo bar", "X Y"
        )
        assert err is None, err
        assert count == 1
        assert new == "X Y baz"

    def test_full_string_match_via_whitespace_normalized(self):
        """When old_string spans the entire content, result is just new_string."""
        content = "hello   world"
        new, count, strat, err = fuzzy_find_and_replace(
            content, "hello world", "hi there"
        )
        assert err is None, err
        assert count == 1
        assert new == "hi there"

    def test_trailing_spaces_in_normalized_match(self):
        """Trailing spaces in original must be fully captured by the mapping.

        Regression: the char-mapping loop checked == before space-collapsing,
        so when original ends with multiple spaces and normalized ends with one,
        the first space consumed the norm position and the rest fell into
        Fill-remaining, mapping past the normalized end.  The result was that
        trailing spaces were only partially replaced.
        """
        content = "hello    world    "  # 4 spaces mid, 4 trailing
        new, count, strat, err = fuzzy_find_and_replace(
            content, "hello world ", "hi"
        )
        assert err is None, err
        assert count == 1
        assert new == "hi"

    def test_trailing_tabs_in_normalized_match(self):
        """Trailing tabs must also be fully captured."""
        content = "hello    world\t\t\t"  # 4 spaces mid, 3 tabs trailing
        new, count, strat, err = fuzzy_find_and_replace(
            content, "hello world\t", "hi"
        )
        assert err is None, err
        assert count == 1
        assert new == "hi"

    def test_match_prefix_only_preserves_rest(self):
        """Partial match (prefix) must leave unmatched trailing content intact."""
        content = "a    b    c"
        new, count, strat, err = fuzzy_find_and_replace(
            content, "a b", "X"
        )
        assert err is None, err
        assert count == 1
        assert new == "X    c"

    def test_match_suffix_only_preserves_leading(self):
        """Partial match (suffix) must leave leading content intact."""
        content = "a    b    c"
        new, count, strat, err = fuzzy_find_and_replace(
            content, "b c", "Y"
        )
        assert err is None, err
        assert count == 1
        assert new == "a    Y"

    def test_replace_all_with_trailing_spaces_no_overlap(self):
        """replace_all must not overlap when matches have trailing spaces."""
        content = "foo   bar   foo   bar   "
        new, count, strat, err = fuzzy_find_and_replace(
            content, "foo bar ", "X", replace_all=True
        )
        assert err is None, err
        assert count == 2
        assert new == "XX"

    def test_multiline_trailing_spaces(self):
        """Multiline match with trailing spaces on last line."""
        content = "def  foo():\n    return  42  "
        new, count, strat, err = fuzzy_find_and_replace(
            content, "def foo():\n    return 42  ", "X"
        )
        assert err is None, err
        assert count == 1
        assert new == "X"



