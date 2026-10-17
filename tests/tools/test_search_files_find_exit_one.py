"""Fallback-find exit-1 handling in filename search (#107403).

GNU find exits 1 when any directory could not be descended into (EACCES on an
unreadable sibling) while still printing every match it did find. The fallback
path used to treat that as a total failure and discard the payload; it must
return the partial results instead, and keep failing closed when exit 1 comes
with an empty payload (a bad root, not a partial traversal).
"""

from unittest.mock import MagicMock

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_common import ExecuteResult


def _find_only_ops() -> ShellFileOperations:
    """Force the fallback path even where rg is installed."""
    ops = ShellFileOperations(LocalEnvironment("/"))
    ops._has_command = lambda command: command == "find"
    return ops


class TestFindExitOneWithMatches:
    def test_partial_traversal_returns_matches_with_warning(self, monkeypatch):
        ops = _find_only_ops()
        match = "/tmp/repo/readable/probe-target.txt"
        exec_result = ExecuteResult(exit_code=1, stdout=f"{match}\n")
        monkeypatch.setattr(ops, "_exec", MagicMock(return_value=exec_result))

        result = ops._search_files("probe-target*", "/tmp/repo", limit=50, offset=0)

        assert result.error is None
        assert result.files == [match]
        assert result.total_count == 1
        assert result.warning is not None
        assert "could not be traversed" in result.warning

    def test_partial_traversal_modified_order_returns_matches(self, monkeypatch):
        ops = _find_only_ops()
        match = "/tmp/repo/readable/probe-target.txt"
        exec_result = ExecuteResult(
            exit_code=1, stdout=f"1696000000.0000000000 {match}\n"
        )
        monkeypatch.setattr(ops, "_exec", MagicMock(return_value=exec_result))

        result = ops._search_files(
            "probe-target*", "/tmp/repo", limit=50, offset=0, order="modified"
        )

        assert result.error is None
        assert result.files == [match]
        assert result.warning is not None


class TestFindExitOneWithoutMatches:
    def test_bad_root_still_fails_closed(self, monkeypatch):
        ops = _find_only_ops()
        monkeypatch.setattr(
            ops, "_exec", MagicMock(return_value=ExecuteResult(exit_code=1, stdout=""))
        )

        result = ops._search_files(
            "probe-target*", "/tmp/missing-root", limit=50, offset=0
        )

        assert (
            result.error == "File search failed while running bounded find traversal."
        )
        assert result.files == []
        assert result.warning is None

    def test_modified_order_without_printf_output_still_fails_closed(self, monkeypatch):
        ops = _find_only_ops()
        monkeypatch.setattr(
            ops, "_exec", MagicMock(return_value=ExecuteResult(exit_code=1, stdout=""))
        )

        result = ops._search_files(
            "probe-target*", "/tmp/repo", limit=50, offset=0, order="modified"
        )

        assert result.error is not None
        assert "-printf" in result.error


class TestFindCleanExit:
    def test_exit_zero_has_no_warning(self, monkeypatch):
        ops = _find_only_ops()
        match = "/tmp/repo/readable/probe-target.txt"
        monkeypatch.setattr(
            ops,
            "_exec",
            MagicMock(return_value=ExecuteResult(exit_code=0, stdout=f"{match}\n")),
        )

        result = ops._search_files("probe-target*", "/tmp/repo", limit=50, offset=0)

        assert result.error is None
        assert result.files == [match]
        assert result.warning is None
