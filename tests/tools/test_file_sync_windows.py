"""Cross-platform file-sync locking contracts."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.environments import file_sync
from tools.environments.file_sync import FileSyncManager, _open_owned_sync_lock


def test_created_lock_junction_open_restore_writes_no_external_bytes(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "locks" / ".sync.lock"
    lock_path.parent.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external = external_dir / lock_path.name
    real_open = file_sync.os.open
    real_remove = file_sync._remove_created_sync_lock
    external_before_cleanup = []

    def junction_open(path, flags, *args, **kwargs):
        if Path(path) == lock_path and flags & os.O_EXCL:
            fd = real_open(external, flags, *args, **kwargs)
            # Model restoring the original parent/path after the redirected open.
            lock_path.write_bytes(b"human replacement")
            return fd
        return real_open(path, flags, *args, **kwargs)

    def observe_remove(path, expected):
        if Path(path) == external:
            external_before_cleanup.append(external.read_bytes())
        return real_remove(path, expected)

    monkeypatch.setattr(file_sync.os, "open", junction_open)
    monkeypatch.setattr(file_sync, "_opened_lock_path", lambda _fd: external)
    monkeypatch.setattr(file_sync, "_remove_created_sync_lock", observe_remove)

    with pytest.raises(OSError, match="provenance changed"):
        _open_owned_sync_lock(lock_path)

    assert external_before_cleanup == [b""]
    assert not external.exists()
    assert lock_path.read_bytes() == b"human replacement"


def test_created_lock_cleanup_preserves_nonempty_opened_object(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "locks" / ".sync.lock"
    lock_path.parent.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    external = external_dir / lock_path.name
    real_open = file_sync.os.open

    def junction_open(path, flags, *args, **kwargs):
        if Path(path) == lock_path and flags & os.O_EXCL:
            fd = real_open(external, flags, *args, **kwargs)
            os.write(fd, b"foreign")
            lock_path.write_bytes(b"human replacement")
            return fd
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_sync.os, "open", junction_open)
    monkeypatch.setattr(file_sync, "_opened_lock_path", lambda _fd: external)

    with pytest.raises(OSError, match="provenance changed"):
        _open_owned_sync_lock(lock_path)

    assert external.read_bytes() == b"foreign"
    assert lock_path.read_bytes() == b"human replacement"


def test_created_lock_race_winner_is_opened_without_replacement(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "locks" / ".sync.lock"
    real_open = file_sync.os.open
    injected = False

    def race_open(path, flags, *args, **kwargs):
        nonlocal injected
        if Path(path) == lock_path and flags & os.O_EXCL and not injected:
            injected = True
            lock_path.write_bytes(b"human winner")
            raise FileExistsError(str(lock_path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(file_sync.os, "open", race_open)

    with _open_owned_sync_lock(lock_path):
        pass

    assert lock_path.read_bytes() == b"human winner"


def test_created_lock_fstat_failure_closes_descriptor(tmp_path, monkeypatch):
    lock_path = tmp_path / "locks" / ".sync.lock"
    real_open = file_sync.os.open
    real_fstat = file_sync.os.fstat
    opened_fd = None

    def tracked_open(path, flags, *args, **kwargs):
        nonlocal opened_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == lock_path:
            opened_fd = fd
        return fd

    def failed_fstat(fd):
        if fd == opened_fd:
            raise OSError("injected fstat failure")
        return real_fstat(fd)

    monkeypatch.setattr(file_sync.os, "open", tracked_open)
    monkeypatch.setattr(file_sync.os, "fstat", failed_fstat)

    with pytest.raises(OSError, match="injected fstat failure"):
        _open_owned_sync_lock(lock_path)

    assert opened_fd is not None
    with pytest.raises(OSError):
        real_fstat(opened_fd)


def test_created_lock_fdopen_failure_closes_and_removes_empty_file(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "locks" / ".sync.lock"
    real_open = file_sync.os.open
    real_fstat = file_sync.os.fstat
    opened_fd = None

    def tracked_open(path, flags, *args, **kwargs):
        nonlocal opened_fd
        fd = real_open(path, flags, *args, **kwargs)
        if Path(path) == lock_path:
            opened_fd = fd
        return fd

    monkeypatch.setattr(file_sync.os, "open", tracked_open)
    monkeypatch.setattr(
        file_sync.os,
        "fdopen",
        MagicMock(side_effect=OSError("injected fdopen failure")),
    )

    with pytest.raises(OSError, match="injected fdopen failure"):
        _open_owned_sync_lock(lock_path)

    assert opened_fd is not None
    with pytest.raises(OSError):
        real_fstat(opened_fd)
    assert not lock_path.exists()


def test_sync_back_uses_windows_lock_when_fcntl_is_unavailable(tmp_path, monkeypatch):
    """Windows sync-back must serialize instead of silently skipping the lock."""
    lock_calls = []
    real_get_osfhandle = getattr(file_sync.msvcrt, "get_osfhandle", lambda fd: fd)
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: lock_calls.append((fd, mode, size)),
        get_osfhandle=real_get_osfhandle,
    )
    monkeypatch.setattr(file_sync, "fcntl", None)
    monkeypatch.setattr(file_sync, "msvcrt", fake_msvcrt)

    host_file = tmp_path / "skill.md"
    host_file.write_text("host")
    mapping = [(str(host_file), "/root/.hermes/skill.md")]
    manager = FileSyncManager(
        get_files_fn=lambda: mapping,
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=lambda destination: None,
    )
    manager._sync_back_impl = MagicMock()
    manager._sync_back_locked(tmp_path / ".hermes" / ".sync.lock")

    assert [mode for _fd, mode, _size in lock_calls] == [
        fake_msvcrt.LK_LOCK,
        fake_msvcrt.LK_UNLCK,
    ]


def test_sync_back_refuses_hardlinked_lock_without_modifying_target(
    tmp_path, monkeypatch
):
    external = tmp_path / "operator.txt"
    external.write_bytes(b"")
    lock_path = tmp_path / ".hermes" / ".sync.lock"
    lock_path.parent.mkdir()
    try:
        lock_path.hardlink_to(external)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    monkeypatch.setattr(file_sync, "fcntl", None)
    monkeypatch.setattr(
        file_sync,
        "msvcrt",
        SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=MagicMock()),
    )
    manager = FileSyncManager(
        get_files_fn=lambda: [],
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=lambda destination: None,
    )
    manager._sync_back_impl = MagicMock()

    with pytest.raises(OSError, match="not exclusively owned"):
        manager._sync_back_locked(lock_path)

    assert external.read_bytes() == b""
    manager._sync_back_impl.assert_not_called()
