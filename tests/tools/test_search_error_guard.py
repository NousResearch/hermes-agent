"""Regression tests for the rg/grep error guard in content search.

The guard in ``_search_with_rg`` / ``_search_with_grep`` had two defects on
``origin/main`` (see PR replacing #39710):

1. **Unreachable on a hard error.** Both methods pipe the search through
   ``| head`` with no ``pipefail``, so the pipeline reported head's exit code
   (0), masking rg/grep's error code (2). The guard never fired, and the
   error text — merged into stdout by ``_exec`` (``stderr=subprocess.STDOUT``)
   — was parsed as bogus match lines instead of being surfaced.

2. **Would have nuked partial results if it ever did fire.** A broad
   ``exit_code == 2`` check discards real matches whenever rg/grep also hit a
   non-fatal error (e.g. one unreadable file in a tree that otherwise
   matched), which both tools signal with exit 2.

The fix adds ``set -o pipefail`` so the real exit code propagates, splits
tool diagnostics from match output by *shape*, and only surfaces an error
when exit==2 AND no usable match payload remains.

These tests drive the real methods through the real local terminal backend.
"""

import os
import shutil

import pytest

from tools.file_operations import (
    ShellFileOperations,
    _pattern_has_regex_newline,
    _split_tool_diagnostics,
)
from tools.environments.local import LocalEnvironment


def _ops(root):
    return ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))


@pytest.fixture
def match_tree(tmp_path):
    """A tree with several files all containing 'needle'."""
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text(f"needle line {i}\n")
    return tmp_path


@pytest.fixture
def partial_error_tree(tmp_path):
    """A tree with matches plus one unreadable file (forces exit 2 + matches)."""
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text(f"needle line {i}\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    locked = sub / "locked.txt"
    locked.write_text("needle in locked\n")
    os.chmod(locked, 0o000)
    yield tmp_path
    os.chmod(locked, 0o755)  # let pytest clean up tmp_path


# Run every test once per available backend method.
_METHODS = ["_search_with_grep"]
if shutil.which("rg"):
    _METHODS.append("_search_with_rg")


def _search(ops, method, pattern, path, **kw):
    fn = getattr(ops, method)
    return fn(pattern, str(path), kw.get("file_glob"), kw.get("limit", 50),
              kw.get("offset", 0), kw.get("output_mode", "content"),
              kw.get("context", 0))


@pytest.mark.parametrize("method", _METHODS)
class TestSearchErrorGuard:
    def test_happy_path_returns_matches(self, method, match_tree):
        res = _search(_ops(match_tree), method, "needle", match_tree)
        assert res.error is None
        assert len(res.matches) == 5

    def test_hard_error_is_surfaced(self, method, match_tree):
        # An invalid regex makes rg/grep exit 2 with only diagnostics in
        # stdout. The guard MUST surface it — not return empty matches.
        res = _search(_ops(match_tree), method, "[", match_tree)
        assert res.error is not None, "search error was silently swallowed"
        assert "Search failed" in res.error
        assert not res.matches

    def test_partial_error_keeps_matches(self, method, partial_error_tree):
        # rg/grep exit 2 because of the unreadable file, but the readable
        # files matched. Those matches must be preserved, not discarded.
        res = _search(_ops(partial_error_tree), method, "needle", partial_error_tree)
        assert res.error is None, f"partial error wrongly surfaced: {res.error!r}"
        assert len(res.matches) >= 4

    def test_no_match_is_empty_not_error(self, method, match_tree):
        res = _search(_ops(match_tree), method, "zzznomatchzzz", match_tree)
        assert res.error is None
        assert not res.matches

    def test_truncation_no_false_error(self, method, tmp_path):
        # head truncates a large result set. With pipefail, grep exits 141
        # (SIGPIPE) on truncation; the strict `== 2` guard must ignore it.
        big = tmp_path / "big.txt"
        big.write_text("".join(f"needle {i}\n" for i in range(3000)))
        res = _search(_ops(tmp_path), method, "needle", tmp_path, limit=5)
        assert res.error is None, f"truncated success wrongly errored: {res.error!r}"
        assert len(res.matches) == 5

    def test_files_only_excludes_diagnostics(self, method, partial_error_tree):
        # files_only mode must not list a diagnostic line as a fake file path.
        res = _search(_ops(partial_error_tree), method, "needle",
                      partial_error_tree, output_mode="files_only")
        assert res.error is None
        assert res.files, "expected matching files"
        assert all("Permission denied" not in f and "locked.txt" not in f
                   for f in res.files), f"diagnostic leaked into files: {res.files}"

    def test_count_mode_with_partial_error(self, method, partial_error_tree):
        res = _search(_ops(partial_error_tree), method, "needle",
                      partial_error_tree, output_mode="count")
        assert res.error is None
        assert res.total_count >= 4


class TestSearchContentNewlineWarning:
    def test_odd_backslash_n_is_detected_as_regex_newline(self):
        assert _pattern_has_regex_newline(r"needle\n")
        assert _pattern_has_regex_newline(r"needle\\\n")

    def test_even_backslash_n_is_literal_and_not_detected(self):
        assert not _pattern_has_regex_newline(r"needle\\n")
        assert not _pattern_has_regex_newline(r"needle\\\\n")

    def test_zero_matches_with_regex_newline_adds_warning_not_error(self, match_tree):
        res = _ops(match_tree).search(
            r"absent\npattern",
            path=str(match_tree),
            target="content",
            context=2,
        )

        assert res.error is None
        assert res.total_count == 0
        assert res.warning is not None
        assert "0 results found" in res.warning
        assert "-U/--multiline" in res.warning

    def test_actual_newline_pattern_adds_warning_not_error(self, match_tree):
        res = _ops(match_tree).search(
            "absent\npattern",
            path=str(match_tree),
            target="content",
        )

        assert res.error is None
        assert res.total_count == 0
        assert res.warning is not None

    def test_search_with_matching_alternative_and_regex_newline_warns(self, match_tree):
        res = _ops(match_tree).search(
            r"needle|absent\npattern",
            path=str(match_tree),
            target="content",
        )

        assert res.error is None
        assert res.total_count == 0
        assert res.warning is not None

    def test_literal_backslash_n_pattern_does_not_warn(self, match_tree):
        res = _ops(match_tree).search(
            r"absent\\npattern",
            path=str(match_tree),
            target="content",
        )

        assert res.error is None
        assert res.total_count == 0
        assert res.warning is None


class TestSplitToolDiagnostics:
    """Unit coverage for the shape-based diagnostic/payload splitter."""

    def test_pure_error_has_empty_payload(self):
        out = "rg: regex parse error:\n    (?:[)\n       ^\nerror: unclosed character class\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert payload.strip() == ""
        assert "regex parse error" in diagnostics

    def test_partial_error_separates_matches(self):
        out = ("rg: sub/locked.txt: Permission denied (os error 13)\n"
               "a.txt:1:needle here\nb.txt:2:needle there\n")
        diagnostics, payload = _split_tool_diagnostics(out)
        assert "Permission denied" in diagnostics
        assert "a.txt:1:needle here" in payload
        assert "b.txt:2:needle there" in payload
        assert "Permission denied" not in payload

    def test_files_only_is_payload(self):
        diagnostics, payload = _split_tool_diagnostics("src/a.py\nsrc/b.py\n")
        assert diagnostics == ""
        assert payload == "src/a.py\nsrc/b.py"

    def test_count_lines_are_payload(self):
        diagnostics, payload = _split_tool_diagnostics("src/a.py:3\nsrc/b.py:1\n")
        assert diagnostics == ""
        assert "src/a.py:3" in payload

    def test_context_lines_and_separator_are_payload(self):
        out = "a.py:5:hit\na.py-6-after\n--\nb.py:9:hit\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert diagnostics == ""
        assert "--" in payload
        assert "a.py-6-after" in payload


@pytest.mark.parametrize("method", _METHODS)
class TestSearchPatternFlagInjection:
    """The search_files pattern must never be parsed as an rg/grep flag.

    ``_escape_shell_arg`` single-quotes the pattern, which blocks *shell*
    injection but does nothing about *argument* injection: a token like
    ``--pre=/bin/sh`` is still handed to ripgrep, which parses it as a flag.
    ``--pre=CMD`` makes rg run CMD as a preprocessor on every searched file —
    arbitrary command execution driven entirely by the model/prompt-controlled
    pattern. The fix inserts a literal ``--`` end-of-options marker before the
    pattern so everything after it is a positional operand.
    """

    def test_leading_dash_pattern_does_not_execute_a_command(
        self, method, tmp_path
    ):
        # A real, executable preprocessor that drops a sentinel file if it ever
        # runs. On vulnerable code rg's `--pre` would invoke it; with the `--`
        # guard the token is just a search string and the sentinel never appears.
        sentinel = tmp_path / "rg_pwned.txt"
        pre = tmp_path / "pre.sh"
        pre.write_text(
            "#!/bin/sh\n"
            f"echo pwned > {sentinel}\n"
            'cat "$1"\n'
        )
        os.chmod(pre, 0o755)

        # Give the search a real file to scan so rg/grep actually walk the tree.
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "f.txt").write_text("needle here\n")

        injection = f"--pre={pre}"
        res = _search(_ops(tree), method, injection, tree)

        assert not sentinel.exists(), (
            "search_files pattern was executed as an rg/grep flag — "
            "argument injection / RCE is not contained"
        )
        # The pattern is a flag-shaped literal that matches nothing in the tree,
        # so this is a clean empty result, never a swallowed flag error.
        assert res.error is None
        assert not res.matches

    def test_leading_dash_pattern_is_searched_literally(self, method, tmp_path):
        # When the dash-prefixed token genuinely appears in a file it must be
        # found as ordinary text rather than consumed as an option.
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "hit.txt").write_text("config: --pre=value\n")

        res = _search(_ops(tree), method, "--pre=value", tree)

        assert res.error is None
        assert len(res.matches) == 1
        assert res.matches[0].path.endswith("hit.txt")
