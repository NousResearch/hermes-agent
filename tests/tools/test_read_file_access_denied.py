"""A path the process cannot stat is "Permission denied", never "File not found".

Regression for #138090: the read probes mapped EACCES/EPERM to the missing
sentinel, so the model was told an existing file was absent.
"""

import os

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

READ_OPS = ["read_file", "read_file_raw", "read_file_bytes"]


@pytest.fixture(params=["1", "0"], ids=["native", "shell"])
def file_ops(request, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", request.param)
    return ShellFileOperations(LocalEnvironment(cwd=str(tmp_path), timeout=15))


@pytest.fixture
def locked_dir(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "notes.txt").write_text("x\n")
    locked.chmod(0o000)
    yield locked
    locked.chmod(0o700)


@pytest.fixture
def locked_file(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores file permissions")
    locked = tmp_path / "secret.txt"
    locked.write_text("secret\n")
    locked.chmod(0o000)
    yield locked
    locked.chmod(0o600)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_unsearchable_parent_is_permission_denied(file_ops, locked_dir, op):
    result = getattr(file_ops, op)(str(locked_dir / "notes.txt"))
    assert result.error and "Permission denied" in result.error
    assert not result.not_found


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_unreadable_file_is_permission_denied(file_ops, locked_file, op):
    result = getattr(file_ops, op)(str(locked_file))
    assert result.error and "Permission denied" in result.error
    assert not result.not_found
    assert not result.content


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_absent_path_is_still_not_found(file_ops, tmp_path, op):
    result = getattr(file_ops, op)(str(tmp_path / "nope.txt"))
    assert result.error and "File not found" in result.error
    assert result.not_found
