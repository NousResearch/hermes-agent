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


    def test_count_mode_with_partial_error(self, method, partial_error_tree):
        res = _search(_ops(partial_error_tree), method, "needle",
                      partial_error_tree, output_mode="count")
        assert res.error is None
        assert res.total_count >= 4


class TestSearchContentNewlineWarning:
    def test_odd_backslash_n_is_detected_as_regex_newline(self):
        assert _pattern_has_regex_newline(r"needle\n")
        assert _pattern_has_regex_newline(r"needle\\\n")


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


    def test_context_lines_and_separator_are_payload(self):
        out = "a.py:5:hit\na.py-6-after\n--\nb.py:9:hit\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert diagnostics == ""
        assert "--" in payload
        assert "a.py-6-after" in payload

    def test_files_only_line_with_spaces_is_payload(self):
        """A files_only path may contain spaces (#91698).

        The old whitespace-free class classified every spaced path as a tool
        diagnostic, silently dropping it from search_files output.
        """
        out = "plain.md\nmy vault/note with space.md\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert diagnostics == ""
        assert "my vault/note with space.md" in payload
        assert "plain.md" in payload

    def test_error_prefix_line_stays_diagnostics_after_space_relaxation(self):
        """rg's trailing "error: ..." line must not leak into payload (#91698)."""
        out = "plain.md\nerror: unclosed character class\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert "error: unclosed character class" in diagnostics
        assert payload.strip() == "plain.md"

    def test_capitalized_and_warning_diagnostic_lines_stay_diagnostics(self):
        """The diagnostic lookaheads are case-insensitive and also cover rg's
        "warning: ..." lines — a capitalized wrapper build's ``Error:`` must
        not land in payload as a phantom result file (review on #91702)."""
        out = "plain.md\nError: unclosed character class\nwarning: skipped binary file\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert "Error: unclosed character class" in diagnostics
        assert "warning: skipped binary file" in diagnostics
        assert payload.strip() == "plain.md"

    def test_spaced_path_match_line_classifies_as_payload(self):
        """The relaxed first alternative also carries spaced paths in
        match/count mode — pin it so a future "tighten for files_only"
        edit cannot silently re-break match mode (review on #91702)."""
        out = "my vault/note.md:3:needle\n"
        diagnostics, payload = _split_tool_diagnostics(out)
        assert diagnostics == ""
        assert "my vault/note.md:3:needle" in payload


@pytest.fixture
def spaced_tree(tmp_path):
    """A tree whose matches all live under a space-containing directory."""
    vault = tmp_path / "Obsidian Vault"
    vault.mkdir()
    (vault / "note with space.md").write_text("needle in vault\n")
    (tmp_path / "plain.md").write_text("needle plain\n")
    return tmp_path


@pytest.mark.parametrize("method", _METHODS)
class TestFilesOnlySpacedPaths:
    """Regression tests for #91698 — files_only must not drop spaced paths."""

    def test_files_only_lists_spaced_paths(self, method, spaced_tree):
        res = _search(_ops(spaced_tree), method, "needle", spaced_tree,
                      output_mode="files_only")
        assert res.error is None
        assert res.total_count == 2
        assert any("note with space.md" in f for f in res.files)
        assert any(f.endswith("plain.md") for f in res.files)

    def test_files_only_all_spaced_directory_not_zero(self, method, tmp_path):
        vault = tmp_path / "all spaced"
        vault.mkdir()
        for i in range(3):
            (vault / f"note {i}.md").write_text("needle\n")
        res = _search(_ops(tmp_path), method, "needle", tmp_path,
                      output_mode="files_only")
        assert res.error is None
        assert res.total_count == 3
        assert len(res.files) == 3
