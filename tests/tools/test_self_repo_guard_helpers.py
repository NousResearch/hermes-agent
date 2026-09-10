"""Test coverage for tools/self_repo_guard.py — path resolution and
git mutation detection helpers. Zero prior coverage for these helpers.

All tests use tmp_path — no real repo is touched.
"""

import pytest
from pathlib import Path

from tools.self_repo_guard import (
    _is_within,
    _resolve,
    _executable_name,
    _mutates_worktree,
    _next_positional,
)


class TestIsWithin:
    def test_path_inside_root(self, tmp_path):
        root = tmp_path / "repo"
        child = root / "src" / "file.py"
        assert _is_within(child, root) is True

    def test_path_outside_root(self, tmp_path):
        root = tmp_path / "repo"
        outside = tmp_path / "other" / "file.py"
        assert _is_within(outside, root) is False

    def test_path_is_root_itself(self, tmp_path):
        assert _is_within(tmp_path, tmp_path) is True


class TestResolve:
    def test_resolves_relative(self, tmp_path):
        result = _resolve("src/file.py", tmp_path)
        assert result == tmp_path / "src" / "file.py"

    def test_resolves_absolute(self, tmp_path):
        result = _resolve(str(tmp_path / "x.py"), tmp_path)
        assert result == tmp_path / "x.py"


class TestExecutableName:
    def test_extracts_name(self):
        assert _executable_name("/usr/bin/git") == "git"

    def test_windows_path(self):
        assert _executable_name("C:\\Program Files\\Git\\cmd\\git.exe") == "git.exe" or \
               _executable_name("C:\\Program Files\\Git\\cmd\\git.exe") == "git"


class TestMutatesWorktree:
    @pytest.mark.parametrize("sub", ["push", "commit", "merge", "rebase", "reset", "checkout"])
    def test_mutating_subcommands(self, sub):
        assert _mutates_worktree(sub, []) is True

    @pytest.mark.parametrize("sub", ["status", "log", "diff", "branch", "remote"])
    def test_non_mutating_subcommands(self, sub):
        assert _mutates_worktree(sub, []) is False


class TestNextPositional:
    def test_skips_flags(self):
        args = ["--force", "main"]
        assert _next_positional(args, 0) == 1

    def test_no_positional(self):
        args = ["--force"]
        assert _next_positional(args, 0) is None
