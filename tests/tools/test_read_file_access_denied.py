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


@pytest.fixture
def sealed_link(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "secret.txt").write_text("secret\n")
    link = tmp_path / "link.txt"
    link.symlink_to(sealed / "secret.txt")
    sealed.chmod(0o000)
    yield link
    sealed.chmod(0o700)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_symlink_into_unsearchable_dir_is_permission_denied(file_ops, sealed_link, op):
    result = getattr(file_ops, op)(str(sealed_link))
    assert result.error and "Permission denied" in result.error
    assert "not a regular file" not in result.error
    assert not result.not_found


@pytest.fixture
def chained_sealed_link(sealed_link, tmp_path):
    chained = tmp_path / "chained.txt"
    chained.symlink_to(sealed_link.name)
    return chained


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_symlink_chain_into_unsearchable_dir_is_permission_denied(file_ops, chained_sealed_link, op):
    result = getattr(file_ops, op)(str(chained_sealed_link))
    assert result.error and "Permission denied" in result.error
    assert not result.not_found


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("op", READ_OPS)
def test_dangling_symlink_stays_not_regular(file_ops, tmp_path, op):
    link = tmp_path / "dangling.txt"
    link.symlink_to(tmp_path / "gone.txt")
    result = getattr(file_ops, op)(str(link))
    assert result.error and "not a regular file" in result.error
    assert not result.not_found


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


@pytest.mark.platforms("posix")
def test_backend_binary_write_under_unsearchable_dir_names_the_denial(locked_dir, monkeypatch):
    from tools import file_tools
    from tools.file_tools_write_guards import _check_binary_document_write

    monkeypatch.setattr(file_tools, "_file_ops_uses_host_paths", lambda _ops: False)
    error = _check_binary_document_write(str(locked_dir / "doc.pdf"), "denied-pdf")
    assert error and "permission denied" in error
    assert "retry once" not in error


@pytest.mark.platforms("posix")
def test_write_file_on_unreadable_file_names_the_denial_not_read_first(locked_file):
    """write_file must not tell the model to read a file read_file cannot read (refusal loop)."""
    import json

    from tools.file_tools import write_file_tool

    result = json.loads(write_file_tool(str(locked_file), "new\n", task_id="denied-write"))
    assert "permission denied" in result["error"]
    assert "Read the file" not in result["error"]
    assert "call write_file again" not in result["error"]
    locked_file.chmod(0o600)
    assert locked_file.read_text() == "secret\n"
    result = json.loads(write_file_tool(str(locked_file), "new\n", task_id="denied-write"))
    assert "call write_file again" in result["error"]
