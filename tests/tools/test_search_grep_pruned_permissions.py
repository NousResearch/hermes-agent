"""Live regression tests: the pruned grep fallback reports unreadable subtrees.

With ``rg`` unavailable, a search rooted under a dot-directory takes the
``find ... -prune -exec grep {} +`` lane. That lane used to discard stderr
(``2>/dev/null``), so a readable match next to an unreadable subtree came back
as a confident, warning-free result (find exits 1 with a nonempty payload and
its ``find: '<dir>': Permission denied`` diagnostic never reached the parser).

Contract: the matches (or counts) are returned AND the skipped path is named in
``warning``; clean searches stay warning-free.
"""

import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations
from tools.file_operations_search import (
    _extract_unreadable_paths,
    _is_permission_only_error,
    _split_tool_diagnostics,
)

_LIVE = [
    pytest.mark.platforms("posix"),
    pytest.mark.skipif(
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root reads chmod 000 dirs",
    ),
]


def live(func):
    """Real find/grep on a real mode-000 directory (POSIX, non-root only)."""
    for mark in reversed(_LIVE):
        func = mark(func)
    return func


def _grep_only_ops(root, monkeypatch):
    ops = ShellFileOperations(LocalEnvironment(str(root)), cwd=str(root))
    # rg absent, grep present: forces the grep lane, and a hidden root forces
    # its pruned (find ... -exec grep) variant.
    monkeypatch.setattr(ops, "_has_command", lambda command: command == "grep")
    return ops


@pytest.fixture
def hidden_root(tmp_path):
    root = tmp_path / ".hidden-root"
    (root / "open").mkdir(parents=True)
    (root / "open" / "a.txt").write_text("needle one\n")
    return root


@pytest.fixture
def locked_dir(hidden_root):
    locked = hidden_root / "locked"
    locked.mkdir()
    (locked / "b.txt").write_text("needle two\n")
    os.chmod(locked, 0o000)
    try:
        yield locked
    finally:
        os.chmod(locked, 0o755)


@live
def test_content_mode_partial_results_name_unreadable_subtree(
        hidden_root, locked_dir, monkeypatch):
    result = _grep_only_ops(hidden_root, monkeypatch).search(
        "needle", path=str(hidden_root), target="content")

    assert result.error is None
    assert [m.path for m in result.matches] == [str(hidden_root / "open" / "a.txt")]
    assert "unreadable path" in (result.warning or "")
    assert str(locked_dir) in (result.warning or "")


@live
def test_count_mode_partial_results_name_unreadable_subtree(
        hidden_root, locked_dir, monkeypatch):
    result = _grep_only_ops(hidden_root, monkeypatch).search(
        "needle", path=str(hidden_root), target="content", output_mode="count")

    assert result.error is None
    assert result.counts == {str(hidden_root / "open" / "a.txt"): 1}
    assert "unreadable path" in (result.warning or "")
    assert str(locked_dir) in (result.warning or "")


@live
def test_no_match_beside_unreadable_subtree_still_warns(
        hidden_root, locked_dir, monkeypatch):
    result = _grep_only_ops(hidden_root, monkeypatch).search(
        "absent-token", path=str(hidden_root), target="content")

    assert result.error is None
    assert result.total_count == 0
    assert str(locked_dir) in (result.warning or "")


@live
@pytest.mark.parametrize("output_mode", ["content", "count"])
def test_clean_match_has_no_warning(hidden_root, monkeypatch, output_mode):
    result = _grep_only_ops(hidden_root, monkeypatch).search(
        "needle", path=str(hidden_root), target="content", output_mode=output_mode)

    assert result.error is None
    assert result.total_count == 1
    assert result.warning is None


@live
@pytest.mark.parametrize("output_mode", ["content", "count"])
def test_clean_no_match_has_no_warning(hidden_root, monkeypatch, output_mode):
    result = _grep_only_ops(hidden_root, monkeypatch).search(
        "absent-token", path=str(hidden_root), target="content", output_mode=output_mode)

    assert result.error is None
    assert result.total_count == 0
    assert result.warning is None


@pytest.mark.parametrize("quoted", ["\u2018/r/locked\u2019", "'/r/locked'"])
def test_find_permission_diagnostic_is_recognized_in_both_quote_styles(quoted):
    line = f"find: {quoted}: Permission denied"

    assert _is_permission_only_error(line)
    assert _extract_unreadable_paths(line) == ["/r/locked"]
    assert _split_tool_diagnostics(f"{line}\n/r/a.txt:1:needle") == (
        line, "/r/a.txt:1:needle")


def test_unrecognized_find_diagnostic_is_not_a_permission_error():
    assert not _is_permission_only_error("find: \u2018/r\u2019: No such file or directory")
