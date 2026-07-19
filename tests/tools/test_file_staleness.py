#!/usr/bin/env python3
"""
Tests for file staleness detection in write_file and patch.

Full-file write_file overwrites must fail closed when the agent has not
freshly read the existing file or when the file changed since that read.
Targeted patch remains warning-only for stale reads.

Run with:  python -m pytest tests/tools/test_file_staleness.py -v
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from tools import file_state
from tools.file_tools import (
    read_file_tool,
    write_file_tool,
    patch_tool,
    reset_file_dedup,
    _check_file_staleness,
    _read_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReadResult:
    def __init__(self, content="line1\nline2\n", total_lines=2, file_size=100):
        self.content = content
        self._total_lines = total_lines
        self._file_size = file_size

    def to_dict(self):
        return {
            "content": self.content,
            "total_lines": self._total_lines,
            "file_size": self._file_size,
        }


class _FakePatchResult:
    def __init__(self):
        self.success = True

    def to_dict(self):
        return {"success": True, "diff": "--- a\n+++ b\n@@ ...\n"}


def _make_fake_ops(read_content="hello\n", file_size=6):
    fake = MagicMock()
    fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
        content=read_content, total_lines=1, file_size=file_size,
    )
    fake.patch_replace = lambda path, old, new, replace_all=False: _FakePatchResult()
    return fake


def _write_until_mtime_changes(path: str, content: str) -> None:
    before = os.path.getmtime(path)
    deadline = time.time() + 2.0
    while True:
        with open(path, "w") as f:
            f.write(content)
        if os.path.getmtime(path) != before:
            return
        if time.time() > deadline:
            forced = before + 1.0
            os.utime(path, (forced, forced))
            return
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Core write_file staleness/no-baseline behavior
# ---------------------------------------------------------------------------

class TestWriteFileStalenessGuard(unittest.TestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "stale_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_fresh_full_read_allows_write_file(self):
        """Read then write with no external modification succeeds cleanly."""
        read = json.loads(read_file_tool(self._tmpfile, task_id="t1"))
        self.assertNotIn("error", read)

        result = json.loads(write_file_tool(self._tmpfile, "new content\n", task_id="t1"))

        self.assertNotIn("error", result)
        self.assertNotIn("_warning", result)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "new content\n")

    def test_redacted_full_read_does_not_allow_secret_corrupting_overwrite(self):
        """A redacted read is not a round-trippable full-file baseline."""
        secret = "ghp_" + "A" * 40
        original = f"token={secret}\n"
        with open(self._tmpfile, "w") as f:
            f.write(original)

        read = json.loads(read_file_tool(self._tmpfile, task_id="redacted"))
        result = json.loads(write_file_tool(
            self._tmpfile,
            "token=«redacted:ghp_…»\n",
            task_id="redacted",
        ))

        self.assertNotIn("error", read)
        self.assertNotIn(secret, read["content"])
        self.assertIn("«redacted:ghp_…»", read["content"])
        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("has not been read", result["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), original)

    def test_write_file_refuses_when_file_modified_externally(self):
        """Read, external modify, then write_file: refuse before clobbering."""
        read = json.loads(read_file_tool(self._tmpfile, task_id="t1"))
        self.assertNotIn("error", read)

        _write_until_mtime_changes(self._tmpfile, "someone else changed this\n")

        result = json.loads(write_file_tool(self._tmpfile, "new content\n", task_id="t1"))

        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("modified since you last read", result["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "someone else changed this\n")

    def test_existing_file_without_baseline_is_refused(self):
        """Existing files must be read before full-file write_file overwrite."""
        result = json.loads(write_file_tool(self._tmpfile, "new content\n", task_id="t2"))

        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("has not been read", result["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "original content\n")

    def test_net_new_file_can_be_rewritten_by_same_task_without_read(self):
        """A successful write_file creates a baseline for same-task rewrites."""
        new_path = os.path.join(self._tmpdir, "brand_new.txt")

        first = json.loads(write_file_tool(new_path, "one\n", task_id="t3"))
        second = json.loads(write_file_tool(new_path, "two\n", task_id="t3"))

        self.assertNotIn("error", first)
        self.assertNotIn("error", second)
        with open(new_path) as f:
            self.assertEqual(f.read(), "two\n")

    def test_different_task_existing_file_requires_own_baseline(self):
        """Task B cannot overwrite an existing file just because task A read it."""
        read = json.loads(read_file_tool(self._tmpfile, task_id="task_a"))
        self.assertNotIn("error", read)
        _write_until_mtime_changes(self._tmpfile, "changed\n")

        result = json.loads(write_file_tool(self._tmpfile, "new\n", task_id="task_b"))

        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("has not been read", result["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "changed\n")

    @patch("tools.file_tools._get_file_ops")
    def test_relative_path_uses_recorded_session_cwd_for_staleness_tracking(self, mock_ops):
        """Relative-path stale tracking must follow the session's recorded cwd."""
        start_dir = os.path.join(self._tmpdir, "start")
        live_dir = os.path.join(self._tmpdir, "worktree")
        os.makedirs(start_dir, exist_ok=True)
        os.makedirs(live_dir, exist_ok=True)

        start_file = os.path.join(start_dir, "shared.txt")
        live_file = os.path.join(live_dir, "shared.txt")
        with open(start_file, "w") as f:
            f.write("start copy\n")
        with open(live_file, "w") as f:
            f.write("live copy\n")

        fake_ops = _make_fake_ops("live copy\n", 10)
        fake_ops.env = SimpleNamespace(cwd=live_dir)
        fake_ops.cwd = start_dir
        fake_ops.write_file = MagicMock(side_effect=AssertionError("must not write stale content"))
        mock_ops.return_value = fake_ops

        from tools import terminal_tool
        terminal_tool.record_session_cwd("live_task", live_dir)

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": start_dir}, clear=False):
                read = json.loads(read_file_tool("shared.txt", task_id="live_task"))
                self.assertNotIn("error", read)

                _write_until_mtime_changes(live_file, "live copy modified elsewhere\n")

                result = json.loads(
                    write_file_tool("shared.txt", "replacement\n", task_id="live_task")
                )
        finally:
            terminal_tool.clear_session_cwd("live_task")

        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("modified since you last read", result["error"])
        fake_ops.write_file.assert_not_called()
        with open(live_file) as f:
            self.assertEqual(f.read(), "live copy modified elsewhere\n")

    def test_write_file_succeeds_after_reread_refreshes_baseline(self):
        read = json.loads(read_file_tool(self._tmpfile, task_id="refresh"))
        self.assertNotIn("error", read)
        _write_until_mtime_changes(self._tmpfile, "external current\n")

        refused = json.loads(write_file_tool(self._tmpfile, "stale overwrite\n", task_id="refresh"))
        reread = json.loads(read_file_tool(self._tmpfile, task_id="refresh"))
        written = json.loads(write_file_tool(self._tmpfile, "external current\nplus update\n", task_id="refresh"))

        self.assertIn("error", refused)
        self.assertNotIn("error", reread)
        self.assertNotIn("error", written)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "external current\nplus update\n")

    def test_full_reread_clears_partial_read_refusal(self):
        with open(self._tmpfile, "w") as f:
            f.write("one\ntwo\nthree\n")

        partial = json.loads(read_file_tool(self._tmpfile, offset=1, limit=1, task_id="partial"))
        refused = json.loads(write_file_tool(self._tmpfile, "replacement\n", task_id="partial"))
        full = json.loads(read_file_tool(self._tmpfile, offset=1, limit=10, task_id="partial"))
        written = json.loads(write_file_tool(self._tmpfile, "replacement\n", task_id="partial"))

        self.assertNotIn("error", partial)
        self.assertIn("error", refused)
        self.assertIn("partial", refused["error"].lower())
        self.assertNotIn("error", full)
        self.assertNotIn("error", written)
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "replacement\n")

    def test_context_compression_reset_clears_full_write_baseline(self):
        """After compression, prior read content may be gone; require re-read."""
        read = json.loads(read_file_tool(self._tmpfile, task_id="compressed"))
        self.assertNotIn("error", read)

        reset_file_dedup("compressed")
        result = json.loads(write_file_tool(self._tmpfile, "after compression\n", task_id="compressed"))

        self.assertIn("error", result)
        self.assertTrue(result.get("stale_write_blocked"))
        self.assertIn("has not been read", result["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "original content\n")


# ---------------------------------------------------------------------------
# Staleness in patch
# ---------------------------------------------------------------------------

class TestPatchStaleness(unittest.TestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "patch_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original line\n")

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("tools.file_tools._get_file_ops")
    def test_patch_warns_on_stale_file(self, mock_ops):
        """Patch should warn if the target file changed since last read."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        read_file_tool(self._tmpfile, task_id="p1")

        _write_until_mtime_changes(self._tmpfile, "externally modified\n")

        result = json.loads(patch_tool(
            mode="replace", path=self._tmpfile,
            old_string="original", new_string="patched",
            task_id="p1",
        ))
        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])

    @patch("tools.file_tools._get_file_ops")
    def test_patch_no_warning_when_fresh(self, mock_ops):
        """Patch with no external changes — no warning."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        read_file_tool(self._tmpfile, task_id="p2")

        result = json.loads(patch_tool(
            mode="replace", path=self._tmpfile,
            old_string="original", new_string="patched",
            task_id="p2",
        ))
        self.assertNotIn("_warning", result)

    def test_successful_patch_does_not_bless_later_full_overwrite(self):
        """A targeted patch is not a whole-file read baseline for write_file."""
        patched = json.loads(patch_tool(
            mode="replace", path=self._tmpfile,
            old_string="original", new_string="patched",
            task_id="patch_only",
        ))
        overwritten = json.loads(write_file_tool(
            self._tmpfile, "full overwrite\n", task_id="patch_only"
        ))

        self.assertNotIn("error", patched)
        self.assertIn("error", overwritten)
        self.assertIn("has not been read", overwritten["error"])
        with open(self._tmpfile) as f:
            self.assertEqual(f.read(), "patched line\n")


# ---------------------------------------------------------------------------
# Unit test for the helper
# ---------------------------------------------------------------------------

class TestCheckFileStalenessHelper(unittest.TestCase):

    def setUp(self):
        _read_tracker.clear()
        file_state.get_registry().clear()

    def tearDown(self):
        _read_tracker.clear()
        file_state.get_registry().clear()

    def test_returns_none_for_unknown_task(self):
        self.assertIsNone(_check_file_staleness("/tmp/x.py", "nonexistent"))

    def test_returns_none_for_unread_file(self):
        # Populate tracker with a different file
        from tools.file_tools import _read_tracker, _read_tracker_lock
        with _read_tracker_lock:
            _read_tracker["t1"] = {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "read_timestamps": {"/tmp/other.py": 12345.0},
            }
        self.assertIsNone(_check_file_staleness("/tmp/x.py", "t1"))

    def test_returns_none_when_stat_fails(self):
        from tools.file_tools import _read_tracker, _read_tracker_lock
        with _read_tracker_lock:
            _read_tracker["t1"] = {
                "last_key": None, "consecutive": 0,
                "read_history": set(), "dedup": {},
                "read_timestamps": {"/nonexistent/path": 99999.0},
            }
        # File doesn't exist → stat fails → returns None (let write handle it)
        self.assertIsNone(_check_file_staleness("/nonexistent/path", "t1"))


if __name__ == "__main__":
    unittest.main()
