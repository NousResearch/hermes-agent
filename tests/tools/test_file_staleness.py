#!/usr/bin/env python3
"""
Tests for file staleness detection in write_file and patch.

When a file is modified externally between the agent's read and write,
the write should include a warning so the agent can re-read and verify.

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


class _FakeWriteResult:
    def __init__(self):
        self.bytes_written = 10

    def to_dict(self):
        return {"bytes_written": self.bytes_written}


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
    fake.write_file = lambda path, content: _FakeWriteResult()
    fake.patch_replace = lambda path, old, new, replace_all=False: _FakePatchResult()
    return fake


# ---------------------------------------------------------------------------
# Core staleness check
# ---------------------------------------------------------------------------

class TestStalenessCheck(unittest.TestCase):

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
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_no_warning_when_file_unchanged(self, mock_ops):
        """Read then write with no external modification — no warning."""
        mock_ops.return_value = _make_fake_ops("original content\n", 18)
        read_file_tool(self._tmpfile, task_id="t1")

        result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="t1"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops")
    def test_warning_when_file_modified_externally(self, mock_ops):
        """Read, then external modify, then write — should warn."""
        mock_ops.return_value = _make_fake_ops("original content\n", 18)
        read_file_tool(self._tmpfile, task_id="t1")

        # Simulate external modification
        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("someone else changed this\n")

        result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])

    @patch("tools.file_tools._get_file_ops")
    def test_no_warning_when_file_never_read(self, mock_ops):
        """Writing a file that was never read — no warning."""
        mock_ops.return_value = _make_fake_ops()
        result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="t2"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops")
    def test_no_warning_for_new_file(self, mock_ops):
        """Creating a new file — no warning."""
        mock_ops.return_value = _make_fake_ops()
        new_path = os.path.join(self._tmpdir, "brand_new.txt")
        result = json.loads(write_file_tool(new_path, "content", task_id="t3"))
        self.assertNotIn("_warning", result)
        try:
            os.unlink(new_path)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_different_task_isolated(self, mock_ops):
        """Task A reads, file changes, Task B writes — no warning for B."""
        mock_ops.return_value = _make_fake_ops("original content\n", 18)
        read_file_tool(self._tmpfile, task_id="task_a")

        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("changed\n")

        result = json.loads(write_file_tool(self._tmpfile, "new", task_id="task_b"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops")
    def test_relative_path_uses_live_cwd_for_staleness_tracking(self, mock_ops):
        """Relative-path stale tracking must follow the live terminal cwd."""
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
        mock_ops.return_value = fake_ops

        from tools import file_tools

        with file_tools._file_ops_lock:
            previous = file_tools._file_ops_cache.get("live_task")
            file_tools._file_ops_cache["live_task"] = fake_ops

        try:
            with patch.dict(os.environ, {"TERMINAL_CWD": start_dir}, clear=False):
                read_file_tool("shared.txt", task_id="live_task")

                time.sleep(0.05)
                with open(live_file, "w") as f:
                    f.write("live copy modified elsewhere\n")

                result = json.loads(
                    write_file_tool("shared.txt", "replacement", task_id="live_task")
                )
        finally:
            with file_tools._file_ops_lock:
                if previous is None:
                    file_tools._file_ops_cache.pop("live_task", None)
                else:
                    file_tools._file_ops_cache["live_task"] = previous

        self.assertIn("_warning", result)
        self.assertIn("modified since you last read", result["_warning"])


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
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_patch_warns_on_stale_file(self, mock_ops):
        """Patch should warn if the target file changed since last read."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        read_file_tool(self._tmpfile, task_id="p1")

        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("externally modified\n")

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
