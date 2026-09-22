"""``utils.rmtree_readonly`` removes trees that ``shutil.rmtree`` refuses.

Git marks loose object files read-only on Windows (``WinError 5``), and package
installs arrive as read-only trees on POSIX, so every cleanup path that deletes a
clone needs the retry.  Regression coverage for #117170, #117176, #117179 and #117184.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import utils
from utils import rmtree_readonly, unlink_readonly


def _read_only_object_dir(root: Path) -> Path:
    """A clone-shaped tree whose loose object and its directory are read-only."""
    obj_dir = root / ".git" / "objects" / "4b"
    obj_dir.mkdir(parents=True)
    obj = obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904"
    obj.write_text("blob", encoding="utf-8")
    obj.chmod(0o444)
    obj_dir.chmod(0o555)  # POSIX unlink needs a writable parent
    return obj_dir


def test_checkpoint_clear_all_removes_tree_with_read_only_object(tmp_path):
    """Driven through a production call site (#117170): ``checkpoint_manager.clear_all`` must
    reach ``rmtree_readonly`` — a bare ``shutil.rmtree`` there reports ``deleted=False``."""
    from tools.checkpoint_manager import clear_all

    root = tmp_path / "checkpoints"
    obj_dir = _read_only_object_dir(root)
    assert not (obj_dir / "825dc642cb6eb9a060e54bf8d69288fbee4904").stat().st_mode & stat.S_IWUSR

    out = clear_all(root)

    assert out["deleted"] is True
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


def test_unlink_readonly_recovers_file_and_parent_permissions(tmp_path, monkeypatch):
    """Single-file replacement uses the same permission recovery as tree removal."""
    parent = tmp_path / "profile" / "tools"
    parent.mkdir(parents=True)
    victim = parent / "helper.py"
    victim.write_text("old", encoding="utf-8")
    victim.chmod(stat.S_IREAD)
    parent.chmod(stat.S_IREAD | stat.S_IEXEC)
    original_unlink = os.unlink
    attempts = 0

    def fail_once(path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("read-only entry")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(utils.os, "unlink", fail_once)

    unlink_readonly(victim)

    assert attempts == 2
    assert not victim.exists()
    parent.chmod(stat.S_IRWXU)


@pytest.mark.require_symlinks
def test_unlink_readonly_symlink_does_not_chmod_external_target(tmp_path, monkeypatch):
    """A failed link unlink may repair its parent but never the external target."""
    external = tmp_path / "external"
    external.write_text("keep", encoding="utf-8")
    external.chmod(stat.S_IREAD)
    original_mode = stat.S_IMODE(external.stat().st_mode)
    parent = tmp_path / "profile"
    parent.mkdir()
    link = parent / "outside"
    link.symlink_to(external)
    parent.chmod(stat.S_IREAD | stat.S_IEXEC)
    original_unlink = os.unlink
    attempts = 0

    def fail_once(path, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("read-only parent")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(utils.os, "unlink", fail_once)

    unlink_readonly(link)

    assert attempts == 2
    assert not link.exists()
    assert external.read_text(encoding="utf-8") == "keep"
    assert stat.S_IMODE(external.stat().st_mode) == original_mode
    parent.chmod(stat.S_IRWXU)
    external.chmod(stat.S_IRWXU)


@pytest.mark.require_symlinks
@pytest.mark.skipif(os.name != "posix", reason="POSIX parent modes drive unlink permission")
def test_read_only_parent_symlink_does_not_chmod_external_target(tmp_path, monkeypatch):
    """Permission recovery may make the parent writable, but must never follow a child link."""
    external = tmp_path / "external"
    external.write_text("keep", encoding="utf-8")
    external.chmod(0o400)
    original_mode = stat.S_IMODE(external.stat().st_mode)

    root = tmp_path / "clone"
    root.mkdir()
    (root / "outside").symlink_to(external)
    root.chmod(0o555)

    def fail_unlink_once(path, **kwargs):
        callback = kwargs["onexc"]
        callback(os.unlink, str(root / "outside"), PermissionError("read-only parent"))
        os.rmdir(path)

    monkeypatch.setattr(utils.shutil, "rmtree", fail_unlink_once)

    rmtree_readonly(root)

    assert not root.exists()
    assert external.read_text(encoding="utf-8") == "keep"
    assert stat.S_IMODE(external.stat().st_mode) == original_mode


def test_typeerror_during_removal_is_not_misread_as_api_fallback(tmp_path, monkeypatch):
    """A callback TypeError is a real failure, not a signal to rerun deletion."""
    attempts: list = []

    def _fake(path, **kwargs):
        attempts.append((path, kwargs))
        raise TypeError("callback bug")

    monkeypatch.setattr(utils.shutil, "rmtree", _fake)

    with pytest.raises(TypeError, match="callback bug"):
        rmtree_readonly(tmp_path)

    assert len(attempts) == 1


def test_non_permission_failures_propagate(tmp_path, monkeypatch):
    """Only ``PermissionError`` is retried — everything else keeps rmtree semantics."""
    attempts: list = []

    def _fake(path, **kwargs):
        attempts.append(path)
        raise OSError(39, "Directory not empty", str(path))

    monkeypatch.setattr(utils.shutil, "rmtree", _fake)

    with pytest.raises(OSError) as excinfo:
        rmtree_readonly(tmp_path)

    assert excinfo.value.errno == 39
    assert len(attempts) == 1  # no second attempt for a non-permission failure
