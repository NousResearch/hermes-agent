"""Cross-platform file-sync locking contracts."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.environments import file_sync
from tools.environments.file_sync import FileSyncManager


def test_sync_back_uses_windows_lock_when_fcntl_is_unavailable(tmp_path, monkeypatch):
    """Windows sync-back must serialize instead of silently skipping the lock."""
    lock_calls = []
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=lambda fd, mode, size: lock_calls.append((fd, mode, size)),
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
