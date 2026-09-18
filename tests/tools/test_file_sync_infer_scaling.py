"""Performance regression tests for ``_infer_host_path``.

``_infer_host_path`` runs once per synced file and used to walk the whole
mapping, re-running ``_is_upload_only_host_path`` (a filesystem
``Path.resolve()``) for every candidate -- O(files x mappings) filesystem
calls, which dominated ``sync_back`` against a large remote.

The directory index makes the filter run once per mapping entry per sync
instead of once per (file, mapping) pair. These tests pin the call count and
the call-site placement, which are the properties that actually decay, rather
than wall-clock timing.
"""

import inspect
import textwrap
from unittest.mock import patch

from tools.environments.file_sync import FileSyncManager


def _manager() -> FileSyncManager:
    return FileSyncManager(
        get_files_fn=lambda: [],
        upload_fn=lambda *a, **k: None,
        delete_fn=lambda *a, **k: None,
        bulk_upload_fn=lambda *a, **k: None,
        bulk_download_fn=lambda *a, **k: None,
    )


def _mapping(n: int) -> list[tuple[str, str]]:
    return [(f"/host/dir{i}/f.txt", f"/remote/dir{i}/f.txt") for i in range(n)]


class TestInferHostPathScaling:
    def test_filter_runs_once_per_mapping_entry_not_per_file(self):
        """The upload-only filter must not rerun for every file.

        Drives the real caller (``_build_dir_index`` once, reused across
        files) rather than looping ``_infer_host_path``, so it stays honest
        if the index is ever rebuilt per file again.
        """
        mgr = _manager()
        mapping = _mapping(50)

        with patch.object(
            FileSyncManager,
            "_is_upload_only_host_path",
            return_value=False,
        ) as filt:
            dir_index = mgr._build_dir_index(mapping, set())
            for i in range(20):
                mgr._infer_host_path(
                    f"/remote/dir{i}/new.txt",
                    mapping,
                    dir_index=dir_index,
                )

        # Pre-fix: 20 files x 50 mappings = up to 1000 calls.
        # With the index: 50 (one pass), reused for every subsequent file.
        assert filt.call_count <= len(mapping), (
            f"filter ran {filt.call_count} times for {len(mapping)} mappings "
            "-- the per-file rescan is back"
        )

    def test_sync_back_builds_the_index_once_for_the_whole_run(self):
        """The production call site must hoist the index out of the file loop.

        The previous test proves the index is reusable; this one proves
        ``_sync_back_impl`` actually reuses it. Without this, moving the
        ``_build_dir_index`` call back inside the ``os.walk`` loop restores
        the O(files x mappings) blowup with every other test still green.
        """
        src = textwrap.dedent(inspect.getsource(FileSyncManager._sync_back_impl))
        build_col = src.index("self._build_dir_index(")
        walk_col = src.index("for dirpath, _dirnames, filenames in os.walk(")

        assert build_col < walk_col, (
            "_build_dir_index moved into/after the os.walk loop -- the index "
            "is now rebuilt per file and the O(n*m) rescan is back"
        )

    def test_inference_result_is_still_correct(self):
        """Caching must not change which host path is returned."""
        mgr = _manager()
        mapping = _mapping(5)
        got = mgr._infer_host_path(
            "/remote/dir3/brand_new.txt",
            mapping,
            upload_only_host_paths=set(),
        )
        assert got == "/host/dir3/brand_new.txt"

    def test_first_match_wins_is_preserved(self):
        """Two mappings sharing a remote dir: the first must win."""
        mgr = _manager()
        mapping = [
            ("/host/first/f.txt", "/remote/shared/f.txt"),
            ("/host/second/g.txt", "/remote/shared/g.txt"),
        ]
        got = mgr._infer_host_path(
            "/remote/shared/new.txt",
            mapping,
            upload_only_host_paths=set(),
        )
        assert got == "/host/first/new.txt"

    def test_upload_only_mappings_are_still_skipped(self):
        """An upload-only host dir must never be inferred as a download target."""
        mgr = _manager()
        mapping = [
            ("/host/secret/creds", "/remote/secret/creds"),
            ("/host/ok/f.txt", "/remote/secret/f.txt"),
        ]

        def only_secret(host_path, upload_only):
            return host_path == "/host/secret/creds"

        with patch.object(
            FileSyncManager,
            "_is_upload_only_host_path",
            side_effect=only_secret,
        ):
            got = mgr._infer_host_path(
                "/remote/secret/new.txt",
                mapping,
                upload_only_host_paths={"/host/secret/creds"},
            )

        assert got == "/host/ok/new.txt"

    def test_cache_refreshes_when_filter_set_changes(self):
        """A different upload-only set must not reuse a stale index."""
        mgr = _manager()
        mapping = [("/host/a/f.txt", "/remote/a/f.txt")]

        first = mgr._infer_host_path(
            "/remote/a/new.txt", mapping, upload_only_host_paths=set()
        )
        assert first == "/host/a/new.txt"

        with patch.object(
            FileSyncManager, "_is_upload_only_host_path", return_value=True
        ):
            second = mgr._infer_host_path(
                "/remote/a/new.txt",
                mapping,
                upload_only_host_paths={"/host/a/f.txt"},
            )

        assert second is None, "stale index reused across a different filter set"
