"""Regression for #85795 — search_files fails on any pattern starting with
"-": missing "--" end-of-options separator before the pattern argument,
in both the rg and grep search paths.

`_escape_shell_arg` correctly single-quotes the pattern (a shell
word-splitting concern), but that does nothing to stop `rg`/`grep`
themselves from parsing a leading "-" as an option flag -- only a `--`
separator immediately before the pattern does that.
"""

from __future__ import annotations

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture
def markdown_checklist_tree(tmp_path):
    """A note vault containing the exact reported repro content."""
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "Project.md").write_text("- [ ] pending task\n")
    (notes / "Diff.md").write_text("-foo removed line\n")
    return tmp_path


def _grep_ops(root, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")
    return ops


def _rg_ops(root, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(cwd=str(root)), cwd=str(root))
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "rg")
    return ops


class TestGrepSearchLeadingDashPattern:
    """The real grep fallback must not treat a leading-"-" pattern as a
    flag -- exercised end-to-end (real grep subprocess, no mocking)."""

    def test_leading_dash_pattern_does_not_error(
        self, markdown_checklist_tree, monkeypatch
    ):
        result = _grep_ops(markdown_checklist_tree, monkeypatch).search(
            "-foo", path=".", target="content"
        )
        assert result.error is None, (
            f"a pattern starting with '-' must not be parsed as a grep "
            f"flag: {result.error!r}"
        )
        assert result.total_count == 1
        assert any("Diff.md" in match.path for match in result.matches)

    def test_leading_dash_pattern_no_longer_reports_invalid_option(
        self, markdown_checklist_tree, monkeypatch
    ):
        """The exact reported symptom: grep's own 'invalid option' error
        text must not appear -- confirms the parsing failure itself is
        gone, independent of whether every character in a given pattern
        happens to match under grep's regex semantics."""
        result = _grep_ops(markdown_checklist_tree, monkeypatch).search(
            "- [ ]", path=".", target="content"
        )
        assert result.error is None
        assert "invalid option" not in str(result.error)


class TestRgCommandIncludesEndOfOptionsSeparator:
    """rg isn't installed in this sandbox, so verify the constructed
    command string directly rather than running rg -- the `--` must sit
    immediately before the (quoted) pattern argument."""

    def test_rg_command_includes_separator_before_pattern(
        self, markdown_checklist_tree, monkeypatch
    ):
        ops = _rg_ops(markdown_checklist_tree, monkeypatch)
        captured = {}

        class _FakeResult:
            exit_code = 1  # rg's "no matches" code -- a clean, error-free run
            stdout = ""

        def _fake_exec(command, cwd=None, timeout=None, **kwargs):
            captured["command"] = command
            return _FakeResult()

        monkeypatch.setattr(ops, "_exec", _fake_exec)

        ops.search("-foo", path=".", target="content")

        assert "command" in captured, "rg path was not exercised"
        assert " -- '-foo'" in captured["command"], (
            f"the -- separator must appear immediately before the quoted "
            f"pattern: {captured['command']!r}"
        )
        # And it must come strictly before the pattern, not after it or
        # before the path only.
        sep_idx = captured["command"].index(" -- ")
        pattern_idx = captured["command"].index("'-foo'")
        assert sep_idx < pattern_idx


class TestGrepCommandIncludesEndOfOptionsSeparator:
    """Same direct command-construction check for the grep path, so the
    separator's presence is verified independent of grep's own regex
    interpretation of any particular pattern."""

    def test_grep_command_includes_separator_before_pattern(
        self, markdown_checklist_tree, monkeypatch
    ):
        ops = _grep_ops(markdown_checklist_tree, monkeypatch)
        captured = {}

        class _FakeResult:
            exit_code = 1
            stdout = ""

        def _fake_exec(command, cwd=None, timeout=None, **kwargs):
            captured["command"] = command
            return _FakeResult()

        monkeypatch.setattr(ops, "_exec", _fake_exec)

        ops.search("-foo", path=".", target="content")

        assert "command" in captured, "grep path was not exercised"
        assert " -- '-foo'" in captured["command"], (
            f"the -- separator must appear immediately before the quoted "
            f"pattern: {captured['command']!r}"
        )
        # Positional check (review of #85798, point 3), robust to any
        # future change in _escape_shell_arg's exact quoting format --
        # not just this specific quoted-string shape.
        sep_idx = captured["command"].index(" -- ")
        pattern_idx = captured["command"].index("'-foo'")
        assert sep_idx < pattern_idx


class TestZeroMatchProbeIncludesEndOfOptionsSeparator:
    """_zero_match_probe() runs up to three of its own separate rg
    invocations (case-insensitive retry, hidden/gitignored retry, fixed-
    string retry) as a "why did this return zero matches" hint when the
    main search finds nothing. These were NOT cited in the original issue
    report but share the exact same missing-separator bug -- each builds
    its own "rg ... {pattern} {path}" command string independently of
    _search_with_rg."""

    def test_all_three_probe_invocations_include_separator(
        self, markdown_checklist_tree, monkeypatch
    ):
        ops = _rg_ops(markdown_checklist_tree, monkeypatch)
        captured_commands = []

        class _FakeResult:
            exit_code = 1
            stdout = ""

        def _fake_exec(command, cwd=None, timeout=None, **kwargs):
            captured_commands.append(command)
            return _FakeResult()

        monkeypatch.setattr(ops, "_exec", _fake_exec)

        # A pattern with a regex metacharacter (".") so all three probe
        # branches run: case-insensitive, hidden/gitignored, AND the
        # fixed-string retry (gated on re.search(r"[.\[\](){}?*+^$\\|]",
        # pattern)).
        ops._zero_match_probe("-fo.o", path=".", file_glob=None)

        rg_commands = [c for c in captured_commands if c.startswith("rg ")]
        assert len(rg_commands) == 3, (
            f"expected all three probe branches to run, got "
            f"{len(rg_commands)}: {rg_commands}"
        )
        for cmd in rg_commands:
            assert " -- '-fo.o'" in cmd, (
                f"probe invocation missing the -- separator before the "
                f"pattern: {cmd!r}"
            )
