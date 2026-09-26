"""A timestamp-preserving skill replacement must reach the remote execution tree."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.environments.file_sync import FileSyncManager, iter_sync_files


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("replacement", [
    "rename",
    pytest.param("copy", marks=pytest.mark.platforms("linux", "macos")),
])
def test_replaced_skill_is_uploaded_without_resending_unchanged_files(
    tmp_path, monkeypatch, bulk, replacement,
):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = home / "skills" / "example" / "script.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"print('old')\n")
    remote_root = tmp_path / "remote"
    uploads = []

    def upload(local, remote):
        destination = remote_root / remote.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, destination)
        uploads.append(remote)

    def upload_bulk(files):
        for local, remote in files:
            upload(local, remote)

    manager = FileSyncManager(
        get_files_fn=iter_sync_files, upload_fn=upload, delete_fn=lambda paths: None,
        bulk_upload_fn=upload_bulk if bulk else None,
    )
    manager.sync(force=True)
    remote_path = "/root/.hermes/skills/example/script.py"
    remote_script = remote_root / remote_path.lstrip("/")
    assert subprocess.check_output([sys.executable, str(remote_script)]).strip() == b"old"
    sent = uploads.count(remote_path)
    manager.sync(force=True)
    assert uploads.count(remote_path) == sent

    # Restores/package extracts commonly retain mtimes. Both versions have the
    # same length, but a fresh inode (or POSIX ctime on in-place copy) distinguishes them.
    before = source.stat()
    restored = tmp_path / "restored.py"
    restored.write_bytes(b"print('new')\n")
    os.utime(restored, ns=(before.st_atime_ns, before.st_mtime_ns))
    if replacement == "rename":
        os.replace(restored, source)
    else:
        shutil.copy2(restored, source)
    assert (source.stat().st_mtime_ns, source.stat().st_size) == (before.st_mtime_ns, before.st_size)

    manager.sync(force=True)
    assert subprocess.check_output([sys.executable, str(remote_script)]).strip() == b"new"
    assert uploads.count(remote_path) == sent + 1
    manager.sync(force=True)
    assert uploads.count(remote_path) == sent + 1
