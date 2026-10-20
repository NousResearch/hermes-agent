"""File reads distinguish inaccessible paths from missing paths."""

import os

import pytest

from tools.file_operations import ShellFileOperations
from tests.tools.test_file_operations import make_real_subprocess_env


_PERMISSION_TEST_UNSUPPORTED = (
    os.name == "nt" or getattr(os, "geteuid", lambda: 1)() == 0
)


@pytest.mark.skipif(
    _PERMISSION_TEST_UNSUPPORTED,
    reason="chmod permissions require a non-root POSIX user",
)
def test_read_file_reports_permission_denied_for_file_in_unsearchable_directory(
    tmp_path,
):
    parent = tmp_path / "sealed"
    parent.mkdir()
    path = parent / "secret.txt"
    path.write_text("secret\n")
    parent.chmod(0)
    try:
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))

        result = ops.read_file(str(path))

        assert result.error is not None
        assert "permission" in result.error.lower()
        assert "exists" in result.error.lower()
        assert result.not_found is not True
    finally:
        parent.chmod(0o700)


def test_read_file_keeps_genuinely_missing_path_as_not_found(tmp_path):
    ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))

    result = ops.read_file(str(tmp_path / "missing.txt"))

    assert result.error is not None
    assert result.not_found is True


@pytest.mark.skipif(
    _PERMISSION_TEST_UNSUPPORTED,
    reason="chmod permissions require a non-root POSIX user",
)
def test_read_file_reports_permission_denied_for_symlink_to_unreadable_file(tmp_path):
    parent = tmp_path / "sealed"
    parent.mkdir()
    target = parent / "secret.txt"
    target.write_text("secret\n")
    link = tmp_path / "secret-link.txt"
    link.symlink_to(target)
    parent.chmod(0)
    try:
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))

        result = ops.read_file(str(link))

        assert result.error is not None
        assert "permission" in result.error.lower()
        assert "exists" in result.error.lower()
        assert result.not_found is not True
    finally:
        parent.chmod(0o700)


@pytest.mark.skipif(
    _PERMISSION_TEST_UNSUPPORTED,
    reason="chmod permissions require a non-root POSIX user",
)
def test_read_file_reports_permission_denied_for_non_readable_file(tmp_path):
    path = tmp_path / "secret.txt"
    path.write_text("secret\n")
    path.chmod(0)
    try:
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))

        result = ops.read_file(str(path))

        assert result.error is not None
        assert "permission" in result.error.lower()
        assert "exists" in result.error.lower()
        assert result.not_found is not True
    finally:
        path.chmod(0o600)


@pytest.mark.skipif(
    _PERMISSION_TEST_UNSUPPORTED,
    reason="chmod permissions require a non-root POSIX user",
)
def test_read_file_raw_reports_permission_denied_for_unsearchable_directory(tmp_path):
    parent = tmp_path / "sealed"
    parent.mkdir()
    path = parent / "secret.txt"
    path.write_text("secret\n")
    parent.chmod(0)
    try:
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))

        result = ops.read_file_raw(str(path))

        assert result.error is not None
        assert "permission" in result.error.lower()
        assert "exists" in result.error.lower()
        assert result.not_found is not True
    finally:
        parent.chmod(0o700)


@pytest.mark.skipif(
    _PERMISSION_TEST_UNSUPPORTED,
    reason="chmod permissions require a non-root POSIX user",
)
def test_native_read_reports_permission_denied_for_unsearchable_directory(
    tmp_path, monkeypatch
):
    parent = tmp_path / "sealed"
    parent.mkdir()
    path = parent / "secret.txt"
    path.write_text("secret\n")
    parent.chmod(0)
    try:
        ops = ShellFileOperations(make_real_subprocess_env(str(tmp_path)))
        monkeypatch.setattr(ops, "_native_read_enabled", lambda: True)

        result = ops.read_file(str(path))

        assert result.error is not None
        assert "permission" in result.error.lower()
        assert "exists" in result.error.lower()
        assert result.not_found is not True
    finally:
        parent.chmod(0o700)
