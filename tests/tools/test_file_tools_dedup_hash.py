"""Tests for read_file dedup content-hash verification (#53812).

Verifies that the dedup mechanism catches sub-second edits where
mtime is unchanged but file content differs (mtime granularity issue).
"""

import json
import os
import tempfile
import time

import tools.file_tools as ft


class TestDedupContentHash:
    """Verify mtime+hash dedup catches sub-second edits."""

    def _read(self, path, task_id="test-dedup-hash"):
        return json.loads(ft.read_file_tool(path, task_id=task_id))

    def test_sub_second_edit_detected(self):
        """File edited within same second (same mtime) returns new content."""
        ft._read_tracker.clear()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("original content\n")
            tmp = f.name

        try:
            # First read — stores (mtime, hash)
            r1 = self._read(tmp)
            assert "original" in r1.get("content", "")
            mtime = os.path.getmtime(tmp)

            # Edit file but preserve mtime (sub-second edit)
            with open(tmp, "w") as f:
                f.write("CHANGED content\n")
            os.utime(tmp, (mtime, mtime))

            # Second read — must detect hash mismatch, return new content
            r2 = self._read(tmp)
            assert r2.get("dedup") is None, "Hash mismatch should bypass dedup"
            assert "CHANGED" in r2.get("content", "")

            # Third read (unchanged) — should return dedup stub
            r3 = self._read(tmp)
            assert r3.get("dedup") is True
            assert r3.get("status") == "unchanged"
        finally:
            os.unlink(tmp)

    def test_backward_compat_bare_float_entry(self):
        """Old-format bare-float dedup entries still work."""
        ft._read_tracker.clear()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("compat test\n")
            tmp = f.name

        try:
            # First read — normal
            self._read(tmp)

            # Inject old-format bare-float entry
            task_data = ft._read_tracker["test-dedup-hash"]
            key = (str(os.path.realpath(tmp)), 1, 500)
            task_data["dedup"][key] = os.path.getmtime(tmp)

            # Second read — bare-float mtime match should return stub
            r2 = self._read(tmp)
            assert r2.get("dedup") is True
        finally:
            os.unlink(tmp)

    def test_normal_mtime_change_not_deduped(self):
        """File edited with different mtime falls through to full read."""
        ft._read_tracker.clear()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("first version\n")
            tmp = f.name

        try:
            self._read(tmp)
            time.sleep(1.1)  # Ensure mtime granularity passes

            with open(tmp, "w") as f:
                f.write("second version\n")

            r2 = self._read(tmp)
            assert r2.get("dedup") is None
            assert "second version" in r2.get("content", "")
        finally:
            os.unlink(tmp)
