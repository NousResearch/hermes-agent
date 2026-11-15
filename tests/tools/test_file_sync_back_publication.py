"""A failed sync-back must leave complete host files for readers and retries."""

import errno
import os
import shutil
import tarfile
from pathlib import Path

import pytest

from tools.environments import file_sync


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(file_sync, "_sleep", lambda _: None)
    host = home / "skills" / "example" / "script.py"
    host.parent.mkdir(parents=True)
    host.write_bytes(b"original complete script\n")
    (host.parent / "anchor.py").write_bytes(b"unchanged local anchor")
    remote = tmp_path / "remote" / "script.py"
    remote.parent.mkdir()
    remote_path = "/root/.hermes/skills/example/script.py"
    observations = []

    def download(destination):
        observations.append(host.read_bytes() if host.exists() else None)
        with tarfile.open(destination, "w") as archive:
            archive.add(remote, arcname=remote_path.lstrip("/"))

    def upload(source, destination):
        shutil.copy2(source, remote.parent / Path(destination).name)

    manager = file_sync.FileSyncManager(
        get_files_fn=lambda: [(str(p), "/root/.hermes/skills/example/" + p.name)
                             for p in host.parent.glob("*.py")],
        upload_fn=upload,
        delete_fn=lambda paths: None,
        bulk_download_fn=download,
    )
    manager.sync(force=True)
    remote.write_bytes(b"complete revised remote script\n")
    return manager, host, remote, observations


@pytest.mark.parametrize("new_file", [False, True])
@pytest.mark.parametrize("fault", ["copy", "replace"])
def test_failed_publication_preserves_host(transfer, monkeypatch, new_file, fault, caplog):
    manager, host, remote, observations = transfer
    original = host.read_bytes()
    if new_file:
        host.unlink()
    original = None if new_file else original
    real_copy = shutil.copyfile
    real_replace = os.replace
    failures = []

    def copy(source, destination, **kwargs):
        if Path(source).read_bytes() == remote.read_bytes() and fault == "copy":
            Path(destination).write_bytes(b"partial")
            failures.append(destination)
            raise OSError(errno.ENOSPC, "injected copy interruption")
        return real_copy(source, destination, **kwargs)

    def replace(source, destination):
        if Path(destination) == host and fault == "replace":
            failures.append(destination)
            raise PermissionError(errno.EACCES, "injected publication refusal")
        return real_replace(source, destination)

    monkeypatch.setattr(shutil, "copyfile", copy)
    monkeypatch.setattr(os, "replace", replace)
    manager.sync_back()

    assert (host.read_bytes() if host.exists() else None) == original
    assert observations and all(value == original for value in observations)
    assert len(failures) == len(observations) > 1
    assert "attempts failed" in caplog.text
    assert set(host.parent.iterdir()) == ({host} if host.exists() else set()) | {host.parent / "anchor.py"}


@pytest.mark.parametrize("interrupt_once", [False, True])
@pytest.mark.parametrize("linked", [False, pytest.param(True, marks=pytest.mark.platforms("linux", "macos"))])
def test_complete_publication_preserves_links_and_retries(transfer, monkeypatch, interrupt_once, linked):
    manager, host, remote, observations = transfer
    original = host.read_bytes()
    target = host
    if linked:
        target = host.parent / "actual.py"
        host.rename(target)
        host.symlink_to(target.name)
    remote.chmod(0o755)
    os.utime(remote, (1_700_000_000, 1_700_000_000))
    real_copy = shutil.copyfile
    interruptions = []

    def copy(source, destination, **kwargs):
        if interrupt_once and not interruptions and Path(source).read_bytes() == remote.read_bytes():
            Path(destination).write_bytes(b"partial")
            interruptions.append(destination)
            raise OSError(errno.ENOSPC, "injected transient copy interruption")
        return real_copy(source, destination, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", copy)
    manager.sync_back()

    assert host.read_bytes() == target.read_bytes() == remote.read_bytes()
    assert all(value == original for value in observations)
    assert len(observations) == (2 if interrupt_once else 1)
    assert host.is_symlink() == linked
    assert target.stat().st_mtime == remote.stat().st_mtime
    if os.name == "posix":
        assert target.stat().st_mode & 0o777 == remote.stat().st_mode & 0o777
    assert set(host.parent.iterdir()) == {host, target, host.parent / "anchor.py"}
