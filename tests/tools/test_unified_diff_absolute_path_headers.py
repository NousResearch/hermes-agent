"""Absolute paths must not join the a//b// diff prefixes into a double slash (#134718).

Every unified-diff producer concatenates the git-style ``a/``/``b/`` prefix onto a
path that may already be absolute, so tool output, write-approval previews and the
session DB accumulated ``a//home/...`` headers that read as a bogus second root.
Each producer now joins a relative label.
"""

import types

from agent.display import LocalEditSnapshot, _diff_from_snapshot
from tools.file_operations import ShellFileOperations
from tools.patch_parser import (
    Hunk,
    HunkLine,
    OperationType,
    PatchOperation,
    _apply_add,
    _unified_diff,
)
from tools.write_approval import skill_pending_diff


def test_patch_parser_unified_diff_strips_the_leading_slash():
    diff = _unified_diff("/home/u/.local/bin/dictate", "one\n", "two\n")
    assert "--- a/home/u/.local/bin/dictate" in diff
    assert "+++ b/home/u/.local/bin/dictate" in diff
    assert "a//" not in diff and "b//" not in diff


def test_v4a_add_file_header_is_single_slashed():
    op = PatchOperation(
        operation=OperationType.ADD,
        file_path="/home/u/new-file.txt",
        hunks=[Hunk(lines=[HunkLine(prefix="+", content="hello")])],
    )
    file_ops = types.SimpleNamespace(
        read_file_raw=lambda _p: types.SimpleNamespace(
            error="not found", not_found=True
        ),
        write_file=lambda _p, _c: types.SimpleNamespace(error=None),
    )
    ok, diff, *_ = _apply_add(op, file_ops)
    assert ok
    assert "+++ b/home/u/new-file.txt" in diff
    assert "b//" not in diff


def test_file_operations_unified_diff_strips_the_leading_slash():
    # _unified_diff never touches self: call it unbound on the implementing class.
    diff = ShellFileOperations._unified_diff(
        None, "one\n", "two\n", "/home/u/notes.txt"
    )
    assert "--- a/home/u/notes.txt" in diff
    assert "+++ b/home/u/notes.txt" in diff
    assert "a//" not in diff and "b//" not in diff


def test_display_snapshot_diff_strips_the_leading_slash(tmp_path):
    target = tmp_path / "dictate"
    target.write_text("two\n")
    snap = LocalEditSnapshot(paths=[target], before={str(target): "one\n"})
    diff = _diff_from_snapshot(snap)
    assert diff is not None
    assert diff.startswith("--- a/")  # relative label, never --- a//
    assert "+++ b/" in diff
    assert "a//" not in diff and "b//" not in diff


def test_write_approval_diff_header_is_single_slashed():
    record = {
        "payload": {
            "action": "write_file",
            "name": "demo",
            "file_path": "/home/u/skill-file.py",
            "file_content": "two\n",
        }
    }
    staged = {"demo": {"/home/u/skill-file.py": "one\n"}}
    diff = skill_pending_diff(record, staged)
    assert "--- a/home/u/skill-file.py" in diff
    assert "+++ b/home/u/skill-file.py" in diff
    assert "a//" not in diff and "b//" not in diff
