"""Behavior tests for ``ShellFileOperations.delete_file`` (real LocalEnvironment shell).

delete_file's python snippet contract: a regular file is unlinked; a directory
is refused (``is a directory``), never silently removed — no recursive delete
exists on this interface.
"""

import os
import sys

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations


@pytest.fixture()
def ops(tmp_path):
    env = LocalEnvironment(cwd=str(tmp_path))
    return ShellFileOperations(env, cwd=str(tmp_path))


def test_delete_file_removes_regular_file(ops, tmp_path):
    target = tmp_path / "doomed.txt"
    target.write_text("bye", encoding="utf-8")

    result = ops.delete_file(str(target))

    assert result.error is None
    assert not target.exists()


def test_delete_file_refuses_directory(ops, tmp_path):
    target = tmp_path / "dir"
    target.mkdir()
    (target / "keep.txt").write_text("still here", encoding="utf-8")

    result = ops.delete_file(str(target))

    assert result.error is not None
    assert "is a directory" in result.error
    assert target.exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", ["delete", "delete_trailing_slash", "delete_dot", "move"])
@pytest.mark.parametrize("layout", ["beside_runtime_venv", "link_in_credential_dir"])
def test_delete_and_move_guard_the_entry_itself(ops, tmp_path, monkeypatch, op, layout):
    """Delete/Move vet the directory entry (parent resolved, leaf kept): a plain file whose
    directory merely CONTAINS the runtime venv (``~/notes.txt``) stays deletable/movable,
    while a link directly inside a credential dir is refused even though it points outside.
    A trailing separator or ``.`` (``.ssh/link/``, ``.ssh/link/.``) must not empty the leaf
    and skip the entry check."""
    home, outside = tmp_path / "home", tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    if layout == "beside_runtime_venv":
        (home / ".venv").mkdir(parents=True)
        monkeypatch.setattr(sys, "prefix", str(home / ".venv"))
        entry = home / "plain.txt"
        entry.write_text("x", encoding="utf-8")
    else:
        (home / ".ssh").mkdir(parents=True)
        entry = home / ".ssh" / "link"
        entry.symlink_to(outside)
    moved = home / "moved.txt"

    if op == "move":
        result = ops.move_file(str(entry), str(moved))
    else:
        result = ops.delete_file(str(entry) + {"delete_trailing_slash": "/", "delete_dot": "/."}.get(op, ""))

    if layout == "beside_runtime_venv":
        assert result.error is None, result.error
        assert not os.path.lexists(entry) and moved.exists() == (op == "move")
    else:
        assert result.error and "protected" in result.error, result.error
        assert entry.is_symlink() and not os.path.lexists(moved)
    assert outside.read_text(encoding="utf-8-sig") == "keep"
