"""Tests for FileSyncManager.sync_back() — pull remote changes to host."""

import io
import logging
import os
import signal
import tarfile
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

fcntl = pytest.importorskip("fcntl")

from tools.environments.file_sync import (
    FileSyncManager,
    _sha256_file,
    _SYNC_BACK_BACKOFF,
    _SYNC_BACK_MAX_RETRIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tar(files: dict[str, bytes], dest: Path):
    """Write a tar archive containing the given arcname->content pairs."""
    with tarfile.open(dest, "w") as tar:
        for arcname, content in files.items():
            info = tarfile.TarInfo(name=arcname)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


def _make_download_fn(files: dict[str, bytes]):
    """Return a bulk_download_fn that writes a tar of the given files."""
    def download(dest: Path):
        _make_tar(files, dest)
    return download


def _sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes (for test convenience)."""
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _write_file(path: Path, content: bytes) -> str:
    """Write bytes to *path*, creating parents, and return the string path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _make_manager(
    tmp_path: Path,
    file_mapping: list[tuple[str, str]] | None = None,
    bulk_download_fn=None,
    seed_pushed_state: bool = True,
) -> FileSyncManager:
    """Create a FileSyncManager wired for testing.

    *file_mapping* is a list of (host_path, remote_path) tuples that
    ``get_files_fn`` returns.  If *None* an empty list is used.

    When *seed_pushed_state* is True (default), populate ``_pushed_hashes``
    from the mapping so sync_back doesn't early-return on the "nothing
    previously pushed" guard. Set False to test the noop path.
    """
    mapping = file_mapping or []
    mgr = FileSyncManager(
        get_files_fn=lambda: mapping,
        upload_fn=MagicMock(),
        delete_fn=MagicMock(),
        bulk_download_fn=bulk_download_fn,
    )
    if seed_pushed_state:
        # Seed _pushed_hashes so sync_back's "nothing previously pushed"
        # guard does not early-return. Populate from the mapping when we
        # can; otherwise drop a sentinel entry.
        for host_path, remote_path in mapping:
            if os.path.exists(host_path):
                mgr._pushed_hashes[remote_path] = _sha256_file(host_path)
            else:
                mgr._pushed_hashes[remote_path] = "0" * 64
        if not mgr._pushed_hashes:
            mgr._pushed_hashes["/_sentinel"] = "0" * 64
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncBackNoop:
    """sync_back() is a no-op when there is no download function."""

    def test_sync_back_noop_without_download_fn(self, tmp_path):
        mgr = _make_manager(tmp_path, bulk_download_fn=None)
        # Should return immediately without error
        mgr.sync_back(hermes_home=tmp_path / ".hermes")
        # Nothing to assert beyond "no exception raised"


class TestSyncBackNoChanges:
    """When all remote files match pushed hashes, nothing is applied."""

    def test_sync_back_no_changes(self, tmp_path):
        host_file = tmp_path / "host" / "cred.json"
        host_content = b'{"key": "val"}'
        _write_file(host_file, host_content)

        remote_path = "/root/.hermes/cred.json"
        mapping = [(str(host_file), remote_path)]

        # Remote tar contains the same content as was pushed
        download_fn = _make_download_fn({
            "root/.hermes/cred.json": host_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        # Simulate that we already pushed this file with this hash
        mgr._pushed_hashes[remote_path] = _sha256_bytes(host_content)

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # Host file should be unchanged (same content, same bytes)
        assert host_file.read_bytes() == host_content


class TestSyncBackAppliesChanged:
    """Remote file differs from pushed version -- gets copied to host."""

    def test_sync_back_applies_changed_file(self, tmp_path):
        host_file = tmp_path / "host" / "skill.py"
        original_content = b"print('v1')"
        _write_file(host_file, original_content)

        remote_path = "/root/.hermes/skill.py"
        mapping = [(str(host_file), remote_path)]

        remote_content = b"print('v2 - edited on remote')"
        download_fn = _make_download_fn({
            "root/.hermes/skill.py": remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        mgr._pushed_hashes[remote_path] = _sha256_bytes(original_content)

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        assert host_file.read_bytes() == remote_content


class TestSyncBackNewRemoteFile:
    """File created on remote (not in _pushed_hashes) is applied via _infer_host_path."""

    def test_sync_back_detects_new_remote_file(self, tmp_path):
        # Existing mapping gives _infer_host_path a prefix to work with
        existing_host = tmp_path / "host" / "skills" / "existing.py"
        _write_file(existing_host, b"existing")
        mapping = [(str(existing_host), "/root/.hermes/skills/existing.py")]

        # Remote has a NEW file in the same directory that was never pushed
        new_remote_content = b"# brand new skill created on remote"
        download_fn = _make_download_fn({
            "root/.hermes/skills/new_skill.py": new_remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        # No entry in _pushed_hashes for the new file

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # The new file should have been inferred and written to the host
        expected_host_path = tmp_path / "host" / "skills" / "new_skill.py"
        assert expected_host_path.exists()
        assert expected_host_path.read_bytes() == new_remote_content


class TestSyncBackConflict:
    """Host AND remote both changed since push -- warning logged, remote wins."""

    def test_sync_back_conflict_warns(self, tmp_path, caplog):
        host_file = tmp_path / "host" / "config.json"
        original_content = b'{"v": 1}'
        _write_file(host_file, original_content)

        remote_path = "/root/.hermes/config.json"
        mapping = [(str(host_file), remote_path)]

        # Host was modified after push
        host_file.write_bytes(b'{"v": 2, "host-edit": true}')

        # Remote was also modified
        remote_content = b'{"v": 3, "remote-edit": true}'
        download_fn = _make_download_fn({
            "root/.hermes/config.json": remote_content,
        })

        mgr = _make_manager(tmp_path, file_mapping=mapping, bulk_download_fn=download_fn)
        mgr._pushed_hashes[remote_path] = _sha256_bytes(original_content)

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # Conflict warning was logged
        assert any("conflict" in r.message.lower() for r in caplog.records)

        # Remote version wins (last-write-wins)
        assert host_file.read_bytes() == remote_content


class TestSyncBackRetries:
    """Retry behaviour with exponential backoff."""

    @patch("tools.environments.file_sync._sleep")
    def test_sync_back_retries_on_failure(self, mock_sleep, tmp_path):
        call_count = 0

        def flaky_download(dest: Path):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"network error #{call_count}")
            # Third attempt succeeds -- write a valid (empty) tar
            _make_tar({}, dest)

        mgr = _make_manager(tmp_path, bulk_download_fn=flaky_download)
        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        assert call_count == 3
        # Sleep called twice (between attempt 1->2 and 2->3)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(_SYNC_BACK_BACKOFF[0])
        mock_sleep.assert_any_call(_SYNC_BACK_BACKOFF[1])

    @patch("tools.environments.file_sync._sleep")
    def test_sync_back_all_retries_exhausted(self, mock_sleep, tmp_path, caplog):
        def always_fail(dest: Path):
            raise RuntimeError("persistent failure")

        mgr = _make_manager(tmp_path, bulk_download_fn=always_fail)

        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            # Should NOT raise -- failures are logged, not propagated
            mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # All retries were attempted
        assert mock_sleep.call_count == _SYNC_BACK_MAX_RETRIES - 1

        # Final "all attempts failed" warning was logged
        assert any("all" in r.message.lower() and "failed" in r.message.lower() for r in caplog.records)


class TestPushedHashesPopulated:
    """_pushed_hashes is populated during sync() and cleared on delete."""

    def test_pushed_hashes_populated_on_sync(self, tmp_path):
        host_file = tmp_path / "data.txt"
        host_file.write_bytes(b"hello world")

        remote_path = "/root/.hermes/data.txt"
        mapping = [(str(host_file), remote_path)]

        mgr = FileSyncManager(
            get_files_fn=lambda: mapping,
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )

        mgr.sync(force=True)

        assert remote_path in mgr._pushed_hashes
        assert mgr._pushed_hashes[remote_path] == _sha256_file(str(host_file))

    def test_pushed_hashes_cleared_on_delete(self, tmp_path):
        host_file = tmp_path / "deleteme.txt"
        host_file.write_bytes(b"to be deleted")

        remote_path = "/root/.hermes/deleteme.txt"
        mapping = [(str(host_file), remote_path)]
        current_mapping = list(mapping)

        mgr = FileSyncManager(
            get_files_fn=lambda: current_mapping,
            upload_fn=MagicMock(),
            delete_fn=MagicMock(),
        )

        # Sync to populate hashes
        mgr.sync(force=True)
        assert remote_path in mgr._pushed_hashes

        # Remove the file from the mapping (simulates local deletion)
        os.unlink(str(host_file))
        current_mapping.clear()

        mgr.sync(force=True)

        # Hash should be cleaned up
        assert remote_path not in mgr._pushed_hashes


class TestSyncBackFileLock:
    """Verify that fcntl.flock is used during sync-back."""

    @patch("tools.environments.file_sync.fcntl.flock")
    def test_sync_back_file_lock(self, mock_flock, tmp_path):
        download_fn = _make_download_fn({})
        mgr = _make_manager(tmp_path, bulk_download_fn=download_fn)

        mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # flock should have been called at least twice: LOCK_EX to acquire, LOCK_UN to release
        assert mock_flock.call_count >= 2

        lock_calls = mock_flock.call_args_list
        lock_ops = [c[0][1] for c in lock_calls]
        assert fcntl.LOCK_EX in lock_ops
        assert fcntl.LOCK_UN in lock_ops

    def test_sync_back_skips_flock_when_fcntl_none(self, tmp_path):
        """On Windows (fcntl=None), sync_back should skip file locking."""
        download_fn = _make_download_fn({})
        mgr = _make_manager(tmp_path, bulk_download_fn=download_fn)

        with patch("tools.environments.file_sync.fcntl", None):
            # Should not raise — locking is skipped
            mgr.sync_back(hermes_home=tmp_path / ".hermes")


class TestInferHostPath:
    """Edge cases for _infer_host_path prefix matching."""

    def test_infer_no_matching_prefix(self, tmp_path):
        """Remote path in unmapped directory should return None."""
        host_file = tmp_path / "host" / "skills" / "a.py"
        _write_file(host_file, b"content")
        mapping = [(str(host_file), "/root/.hermes/skills/a.py")]

        mgr = _make_manager(tmp_path, file_mapping=mapping)
        result = mgr._infer_host_path(
            "/root/.hermes/cache/new.json",
            file_mapping=mapping,
        )
        assert result is None

    def test_infer_partial_prefix_no_false_match(self, tmp_path):
        """A partial prefix like /root/.hermes/sk should NOT match /root/.hermes/skills/."""
        host_file = tmp_path / "host" / "skills" / "a.py"
        _write_file(host_file, b"content")
        mapping = [(str(host_file), "/root/.hermes/skills/a.py")]

        mgr = _make_manager(tmp_path, file_mapping=mapping)
        # /root/.hermes/skillsXtra/b.py shares prefix "skills" but the
        # directory is different — should not match /root/.hermes/skills/
        result = mgr._infer_host_path(
            "/root/.hermes/skillsXtra/b.py",
            file_mapping=mapping,
        )
        assert result is None

    def test_infer_matching_prefix(self, tmp_path):
        """A file in a mapped directory should be correctly inferred."""
        host_file = tmp_path / "host" / "skills" / "a.py"
        _write_file(host_file, b"content")
        mapping = [(str(host_file), "/root/.hermes/skills/a.py")]

        mgr = _make_manager(tmp_path, file_mapping=mapping)
        result = mgr._infer_host_path(
            "/root/.hermes/skills/b.py",
            file_mapping=mapping,
        )
        expected = str(tmp_path / "host" / "skills" / "b.py")
        assert result == expected


class TestSyncBackSIGINT:
    """SIGINT deferral during sync-back."""

    def test_sync_back_defers_sigint_on_main_thread(self, tmp_path):
        """On the main thread, SIGINT handler should be swapped during sync."""
        download_fn = _make_download_fn({})
        mgr = _make_manager(tmp_path, bulk_download_fn=download_fn)

        handlers_seen = []
        original_getsignal = signal.getsignal

        with patch("tools.environments.file_sync.signal.getsignal",
                    side_effect=original_getsignal) as mock_get, \
             patch("tools.environments.file_sync.signal.signal") as mock_set:
            mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # signal.getsignal was called to save the original handler
        assert mock_get.called
        # signal.signal was called at least twice: install defer, restore original
        assert mock_set.call_count >= 2

    def test_sync_back_skips_signal_on_worker_thread(self, tmp_path):
        """From a non-main thread, signal.signal should NOT be called."""
        import threading

        download_fn = _make_download_fn({})
        mgr = _make_manager(tmp_path, bulk_download_fn=download_fn)

        signal_called = []

        def tracking_signal(*args):
            signal_called.append(args)

        with patch("tools.environments.file_sync.signal.signal", side_effect=tracking_signal):
            # Run from a worker thread
            exc = []
            def run():
                try:
                    mgr.sync_back(hermes_home=tmp_path / ".hermes")
                except Exception as e:
                    exc.append(e)

            t = threading.Thread(target=run)
            t.start()
            t.join(timeout=10)

        assert not exc, f"sync_back raised: {exc}"
        # signal.signal should NOT have been called from the worker thread
        assert len(signal_called) == 0


class TestSyncBackSizeCap:
    """The size cap refuses to extract tars above the configured limit."""

    def test_sync_back_refuses_oversized_tar(self, tmp_path, caplog):
        """A tar larger than _SYNC_BACK_MAX_BYTES should be skipped with a warning."""
        # Build a download_fn that writes a small tar, but patch the cap
        # so the test doesn't need to produce a 2 GiB file.
        skill_host = _write_file(tmp_path / "host_skill.md", b"original")
        files = {"root/.hermes/skill.md": b"remote_version"}
        download_fn = _make_download_fn(files)

        mgr = _make_manager(
            tmp_path,
            file_mapping=[(skill_host, "/root/.hermes/skill.md")],
            bulk_download_fn=download_fn,
        )

        # Cap at 1 byte so any non-empty tar exceeds it
        with caplog.at_level(logging.WARNING, logger="tools.environments.file_sync"):
            with patch("tools.environments.file_sync._SYNC_BACK_MAX_BYTES", 1):
                mgr.sync_back(hermes_home=tmp_path / ".hermes")

        # Host file should be untouched because extraction was skipped
        assert Path(skill_host).read_bytes() == b"original"
        # Warning should mention the cap
        assert any("cap" in r.message for r in caplog.records)

    def test_sync_back_applies_when_under_cap(self, tmp_path):
        """A tar under the cap should extract normally (sanity check)."""
        host_file = _write_file(tmp_path / "host_skill.md", b"original")
        files = {"root/.hermes/skill.md": b"remote_version"}
        download_fn = _make_download_fn(files)

        mgr = _make_manager(
            tmp_path,
            file_mapping=[(host_file, "/root/.hermes/skill.md")],
            bulk_download_fn=download_fn,
        )

        # Default cap (2 GiB) is far above our tiny tar; extraction should proceed
        mgr.sync_back(hermes_home=tmp_path / ".hermes")
        assert Path(host_file).read_bytes() == b"remote_version"
