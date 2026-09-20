"""``utils.rmtree_readonly`` removes trees that ``shutil.rmtree`` refuses.

Git marks loose object files read-only on Windows (``WinError 5``), and package
installs arrive as read-only trees on POSIX, so every cleanup path that deletes a
clone needs the retry.  Regression coverage for #117170, #117176 and #117179.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import utils
from utils import rmtree_readonly


def _read_only_object_dir(root: Path) -> Path:
    """A clone-shaped tree whose loose object and its directory are read-only."""
    obj_dir = root / ".git" / "objects" / "4b"
    obj_dir.mkdir(parents=True)
    obj = obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904"
    obj.write_text("blob", encoding="utf-8")
    obj.chmod(0o444)
    obj_dir.chmod(0o555)  # POSIX unlink needs a writable parent
    return obj_dir


def test_removes_tree_with_read_only_object(tmp_path):
    root = tmp_path / "plugins" / "demo"
    obj_dir = _read_only_object_dir(root)
    assert not (obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904").stat().st_mode & stat.S_IWUSR

    rmtree_readonly(root)

    assert not root.exists()


@pytest.mark.windows_only
def test_removes_read_only_file_in_writable_directory(tmp_path):
    """The Git-for-Windows shape: the file is read-only, its directory is writable."""
    root = tmp_path / "plugins" / "demo"
    obj_dir = root / ".git" / "objects" / "4b"
    obj_dir.mkdir(parents=True)
    (obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904").write_text("blob", encoding="utf-8")
    (obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904").chmod(stat.S_IREAD)

    rmtree_readonly(root)

    assert not root.exists()


def _stub_rmtree(monkeypatch, exc: OSError) -> list:
    """Replace ``shutil.rmtree`` so the wrapper sees *exc* for every attempt."""
    attempts: list = []

    def _fake(path, **kwargs):
        attempts.append((path, kwargs))
        raise exc

    monkeypatch.setattr(utils.shutil, "rmtree", _fake)
    return attempts


def test_ignore_errors_swallows_a_persistent_permission_failure(tmp_path, monkeypatch):
    _stub_rmtree(monkeypatch, PermissionError(13, "Access is denied", str(tmp_path)))

    rmtree_readonly(tmp_path, ignore_errors=True)


def test_permission_failure_still_raises_without_ignore_errors(tmp_path, monkeypatch):
    _stub_rmtree(monkeypatch, PermissionError(13, "Access is denied", str(tmp_path)))

    with pytest.raises(PermissionError):
        rmtree_readonly(tmp_path)


def test_non_permission_failures_propagate(tmp_path, monkeypatch):
    """Only ``PermissionError`` is retried — everything else keeps rmtree semantics."""
    attempts = _stub_rmtree(monkeypatch, OSError(39, "Directory not empty", str(tmp_path)))

    with pytest.raises(OSError) as excinfo:
        rmtree_readonly(tmp_path)

    assert excinfo.value.errno == 39
    assert len(attempts) == 1  # no second attempt for a non-permission failure
