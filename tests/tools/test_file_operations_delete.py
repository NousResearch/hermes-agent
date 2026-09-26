"""Behavior tests for ``ShellFileOperations.delete_file`` (real LocalEnvironment shell).

delete_file's python snippet contract: a regular file is unlinked; a directory
is refused (``is a directory``), never silently removed — no recursive delete
exists on this interface.
"""

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
