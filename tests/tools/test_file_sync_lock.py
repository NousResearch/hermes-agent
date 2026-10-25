"""Sync-back lock lifetime and contention, independently collectible on Windows."""
import errno
import logging
import multiprocessing
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.environments.file_sync as sync


def _manager():
    manager = sync.FileSyncManager(lambda: [], lambda *args: None, lambda *args: None)
    manager._sync_back_impl = MagicMock()
    return manager


@pytest.mark.parametrize("busy", [False, True])
def test_windows_sync_lock_uses_nonblocking_contention_retry(tmp_path, monkeypatch, busy):
    fake = MagicMock(LK_NBLCK=1, LK_UNLCK=2)
    calls = []

    def lock(fd, mode, size):
        calls.append((mode, size, os.lseek(fd, 0, os.SEEK_CUR)))
        if busy and len(calls) == 1:
            raise OSError(errno.EACCES, "owned by another process")

    fake.locking.side_effect = lock
    monkeypatch.setattr(sync, "fcntl", None)
    monkeypatch.setattr(sync, "msvcrt", fake)
    sleep = MagicMock()
    monkeypatch.setattr(sync, "_sleep", sleep)
    manager = _manager()
    manager._sync_back_locked(tmp_path / ".sync.lock")
    assert calls == [(1, 1, 0)] * (2 if busy else 1) + [(2, 1, 0)]
    assert sleep.call_args_list == ([((0.1,), {})] if busy else [])
    manager._sync_back_impl.assert_called_once_with()


@pytest.mark.parametrize("failure", ["acquire", "sync"])
def test_windows_sync_lock_releases_only_after_acquisition(tmp_path, monkeypatch, failure):
    fake = MagicMock(LK_NBLCK=1, LK_UNLCK=2)
    captured = []

    def lock(fd, mode, size):
        captured.append(fd)
        if failure == "acquire":
            raise OSError(errno.EBADF, "unusable lock")

    fake.locking.side_effect = lock
    monkeypatch.setattr(sync, "fcntl", None)
    monkeypatch.setattr(sync, "msvcrt", fake)
    manager = _manager()
    manager._sync_back_impl.side_effect = RuntimeError("download failed")
    with pytest.raises(OSError if failure == "acquire" else RuntimeError):
        manager._sync_back_locked(tmp_path / ".sync.lock")
    assert [c.args[1] for c in fake.locking.call_args_list] == ([1] if failure == "acquire" else [1, 2])
    assert manager._sync_back_impl.call_count == (failure == "sync")
    with pytest.raises(OSError):
        os.fstat(captured[0])


def test_sync_back_without_locking_module_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(sync, "fcntl", None)
    monkeypatch.setattr(sync, "msvcrt", None)
    manager = _manager()
    with caplog.at_level(logging.WARNING, logger=sync.__name__):
        manager._sync_back_locked(tmp_path / ".sync.lock")
    manager._sync_back_impl.assert_called_once_with()
    assert "without serialization" in caplog.text


def _sync_lock_process(lock_path, ready, entered, release):
    manager = sync.FileSyncManager(lambda: [], lambda *args: None, lambda *args: None)

    def apply():
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test holder was not released")

    manager._sync_back_impl = apply
    ready.set()
    manager._sync_back_locked(Path(lock_path))


def test_concurrent_sync_back_processes_serialize(tmp_path):
    """Exercise the host's real locking implementation, including Windows in CI."""
    ctx = multiprocessing.get_context("spawn")
    ready, entered, release = [[ctx.Event() for _ in range(2)] for _ in range(3)]
    workers = [ctx.Process(target=_sync_lock_process, args=(
        str(tmp_path / ".sync.lock"), ready[i], entered[i], release[i])) for i in range(2)]
    try:
        workers[0].start()
        assert entered[0].wait(10)
        workers[1].start()
        assert ready[1].wait(10)
        assert not entered[1].wait(0.2)
        release[0].set()
        assert entered[1].wait(10)
    finally:
        for event in release:
            event.set()
        for worker in workers:
            if worker.pid is not None:
                worker.join(5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(5)
    assert [worker.exitcode for worker in workers] == [0, 0]


@pytest.mark.parametrize("pre_existing", [False, True])
def test_windows_lock_file_has_a_byte_range_to_lock(tmp_path, monkeypatch, pre_existing):
    """A zero-length lock file gives msvcrt nothing to lock (see agent/vault_store.py)."""
    fake = MagicMock(LK_NBLCK=1, LK_UNLCK=2)
    monkeypatch.setattr(sync, "fcntl", None)
    monkeypatch.setattr(sync, "msvcrt", fake)
    lock_path = tmp_path / ".sync.lock"
    if pre_existing:
        lock_path.write_bytes(b"")
    _manager()._sync_back_locked(lock_path)
    assert lock_path.stat().st_size == 1
    assert fake.locking.call_args_list[0].args[1] == fake.LK_NBLCK
