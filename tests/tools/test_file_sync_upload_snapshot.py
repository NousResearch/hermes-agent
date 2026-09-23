"""Upload hashes must describe the bytes received by the remote sandbox."""

import hashlib
import tarfile
from pathlib import Path

import pytest

from tools.environments.file_sync import FileSyncManager


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("edit_before_read", [False, True])
def test_host_save_during_upload_survives_unchanged_remote(
    tmp_path, monkeypatch, bulk, edit_before_read,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    host = tmp_path / "home" / "skills" / "example" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_bytes(b"original skill")
    remote_path = "/root/.hermes/skills/example/SKILL.md"
    remote = tmp_path / "remote.md"

    def upload(source, destination):
        assert destination == remote_path
        if edit_before_read:
            host.write_bytes(b"saved while upload is starting")
        remote.write_bytes(Path(source).read_bytes())
        host.write_bytes(b"new local version saved before upload acknowledgement")

    def download(destination):
        with tarfile.open(destination, "w") as archive:
            archive.add(remote, arcname=remote_path.lstrip("/"))

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host), remote_path)],
        upload_fn=upload,
        bulk_upload_fn=(lambda files: [upload(*pair) for pair in files]) if bulk else None,
        delete_fn=lambda paths: None,
        bulk_download_fn=download,
    )
    manager.sync(force=True)
    saved = host.read_bytes()
    assert manager._pushed_hashes[remote_path] == hashlib.sha256(remote.read_bytes()).hexdigest()
    manager.sync_back()
    assert host.read_bytes() == saved

    # A subsequent cycle still detects and uploads the newer local version.
    manager._upload_fn = lambda source, destination: remote.write_bytes(Path(source).read_bytes())
    manager._bulk_upload_fn = None
    manager.sync(force=True)
    assert remote.read_bytes() == saved
