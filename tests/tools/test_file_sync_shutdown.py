"""Tests for sync_back during interpreter shutdown.

``cleanup()`` can fire from a late atexit hook or ``__del__``, by which point
Python has torn down module globals -- builtins such as ``open`` included.  The
old code raised a bare ``NameError`` and then slept through the full retry
backoff, since every retry hit the same torn-down interpreter.
"""

from unittest.mock import MagicMock, patch

from tools.environments.file_sync import FileSyncManager


def _manager(**kw):
    """A FileSyncManager with a pushed file, so sync_back does not early-exit."""
    m = FileSyncManager(
        get_files_fn=lambda: [],
        upload_fn=lambda *a: None,
        delete_fn=lambda *a: None,
        bulk_upload_fn=lambda files: None,
        bulk_download_fn=lambda dest, paths=None: None,
        **kw,
    )
    m._pushed_hashes = {"/remote/.hermes/skills/a.md": "deadbeef"}
    return m


class TestSyncBackAtShutdown:
    def test_nameerror_abandons_immediately(self, tmp_path):
        """No retries, no sleeping -- the interpreter is not coming back."""
        m = _manager()
        with patch.object(FileSyncManager, "_sync_back_once",
                          side_effect=NameError("name 'open' is not defined")) as once, \
             patch("tools.environments.file_sync._sleep") as sleep:
            m.sync_back(hermes_home=tmp_path)
            assert once.call_count == 1, "shutdown must not be retried"
            sleep.assert_not_called()

    def test_ordinary_failure_still_retries(self, tmp_path):
        """A transient error keeps the existing retry behaviour."""
        m = _manager()
        with patch.object(FileSyncManager, "_sync_back_once",
                          side_effect=OSError("connection reset")) as once, \
             patch("tools.environments.file_sync._sleep"):
            m.sync_back(hermes_home=tmp_path)
            assert once.call_count > 1, "transient errors should retry"


class TestSyncBackSkipsEmptyPush:
    def test_no_pushed_files_downloads_nothing(self, tmp_path):
        """Pre-existing early-exit, re-pinned because the targeted-download
        change made this path newly load-bearing."""
        download = MagicMock()
        m = FileSyncManager(
            get_files_fn=lambda: [],
            upload_fn=lambda *a: None,
            delete_fn=lambda *a: None,
            bulk_upload_fn=lambda files: None,
            bulk_download_fn=download,
        )
        m._pushed_hashes = {}
        m.sync_back(hermes_home=tmp_path)
        download.assert_not_called()
