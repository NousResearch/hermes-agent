#!/usr/bin/env python3
"""
Tests for file staleness detection in write_file and patch.

When a file is modified externally between the agent's read and write,
the write should include a warning so the agent can re-read and verify.

Run with:  python -m pytest tests/tools/test_file_staleness.py -v
"""

import json
import os
import hashlib
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


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    @patch("tools.file_tools._get_file_ops")
    def test_strict_patch_blocks_after_partial_read(self, mock_ops):
        """Strict mode rejects mutation when the file was only partially read."""
        mock_ops.return_value = _make_fake_ops("line1\n", 12)
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            read_file_tool(self._tmpfile, offset=2, limit=1, task_id="p3")
            result = json.loads(patch_tool(
                mode="replace", path=self._tmpfile,
                old_string="original", new_string="patched",
                task_id="p3",
            ))
        self.assertIn("error", result)
        self.assertIn("partial", result["error"].lower())

    @patch("tools.file_tools._get_file_ops")
    def test_strict_write_blocks_after_partial_read(self, mock_ops):
        """Strict mode rejects overwrite when the file was only partially read."""
        mock_ops.return_value = _make_fake_ops("line1\n", 12)
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            read_file_tool(self._tmpfile, offset=2, limit=1, task_id="p4")
            result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p4"))
        self.assertIn("error", result)
        self.assertIn("partial", result["error"].lower())

    @patch("tools.file_tools._get_file_ops")
    def test_strict_write_blocks_unread_existing_file(self, mock_ops):
        """Strict mode requires a full prior read before overwriting an existing file."""
        mock_ops.return_value = _make_fake_ops()
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p5"))
        self.assertIn("error", result)
        self.assertIn("never read", result["error"].lower())

    @patch("tools.file_tools._get_file_ops")
    def test_strict_write_blocks_when_file_changed_after_full_read(self, mock_ops):
        """Strict mode rejects stale CAS mismatch after external modification."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            read_file_tool(self._tmpfile, task_id="p6")
            time.sleep(0.05)
            with open(self._tmpfile, "w") as f:
                f.write("externally modified\n")
            result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p6"))
        self.assertIn("error", result)
        self.assertIn("modified since you last read", result["error"])

    @patch("tools.file_tools._get_file_ops")
    def test_warn_mode_still_warns_and_applies(self, mock_ops):
        """Warn mode preserves compatibility: warning only, mutation still applies."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "warn"}, clear=False):
            read_file_tool(self._tmpfile, task_id="p7")
            time.sleep(0.05)
            with open(self._tmpfile, "w") as f:
                f.write("externally modified\n")
            result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p7"))
        self.assertIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops")
    def test_strict_new_file_allowed(self, mock_ops):
        """Strict mode still allows creating a brand-new file."""
        mock_ops.return_value = _make_fake_ops()
        new_path = os.path.join(self._tmpdir, "strict_new.txt")
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            result = json.loads(write_file_tool(new_path, "content", task_id="p8"))
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops")
    def test_same_task_consecutive_write_allowed_in_strict_mode(self, mock_ops):
        """A successful own write refreshes the fingerprint for the next strict write."""
        mock_ops.return_value = _make_fake_ops("original line\n", 15)
        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            json.loads(read_file_tool(self._tmpfile, task_id="p9"))
            first = json.loads(write_file_tool(self._tmpfile, "one", task_id="p9"))
            second = json.loads(write_file_tool(self._tmpfile, "two", task_id="p9"))
        self.assertNotIn("error", first)
        self.assertNotIn("error", second)

    @patch("tools.file_tools._get_file_ops")
    def test_strict_write_blocks_if_file_changes_during_read_window(self, mock_ops):
        """Strict CAS must not store a post-read fingerprint for content the model never saw."""
        fake = MagicMock()

        def racing_read(_path, _offset=1, _limit=500):
            with open(self._tmpfile, "w") as f:
                f.write("changed after read\n")
            return _FakeReadResult(content="original line\n", total_lines=1, file_size=14)

        fake.read_file = racing_read
        fake.write_file = lambda path, content: _FakeWriteResult()
        fake.patch_replace = lambda path, old, new, replace_all=False: _FakePatchResult()
        mock_ops.return_value = fake

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False):
            json.loads(read_file_tool(self._tmpfile, task_id="p10"))
            result = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p10"))

        self.assertIn("error", result)
        self.assertIn("changed while it was being read", result["error"])

    def test_strict_write_allows_ast_expanded_read_that_covers_entire_python_file(self):
        pyfile = os.path.join(self._tmpdir, "full_symbol.py")
        with open(pyfile, "w") as f:
            f.write("def demo():\n    return 1\n")

        from tools.file_operations import ShellFileOperations
        from tests.tools.test_file_operations import LocalShellEnv

        fake_ops = ShellFileOperations(LocalShellEnv(self._tmpdir))

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False), \
             patch("tools.file_tools._get_file_ops", return_value=fake_ops):
            read_result = json.loads(read_file_tool(pyfile, offset=2, limit=1, task_id="p10_ast"))
            write_result = json.loads(write_file_tool(pyfile, "def demo():\n    return 2\n", task_id="p10_ast"))

        self.assertIn("content", read_result)
        self.assertNotIn("error", write_result)

    def test_strict_write_blocks_after_ast_expanded_method_read_that_does_not_cover_whole_file(self):
        pyfile = os.path.join(self._tmpdir, "method_only.py")
        with open(pyfile, "w") as f:
            f.write(
                "class Outer:\n"
                "    padding0 = 0\n"
                "    padding1 = 1\n"
                "    padding2 = 2\n"
                "    def method(self):\n"
                "        target = 1\n"
                "        return target\n"
                "    def second(self):\n"
                "        return 2\n"
                "    padding3 = 3\n"
            )

        from tools.file_operations import ShellFileOperations
        from tests.tools.test_file_operations import LocalShellEnv

        fake_ops = ShellFileOperations(LocalShellEnv(self._tmpdir))

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "strict"}, clear=False), \
             patch("tools.file_tools._get_file_ops", return_value=fake_ops):
            read_result = json.loads(read_file_tool(pyfile, offset=6, limit=1, task_id="p10_method"))
            write_result = json.loads(write_file_tool(pyfile, "class Outer:\n    changed = True\n", task_id="p10_method"))

        self.assertEqual(read_result.get("returned_start_line"), 5)
        self.assertEqual(read_result.get("returned_end_line"), 7)
        self.assertIn("error", write_result)
        self.assertIn("partial", write_result["error"].lower())

    def test_dedup_uses_hash_not_mtime_only(self):
        """Same-mtime content drift must force a real reread instead of an unchanged stub."""
        task_id = "p11"
        json.loads(read_file_tool(self._tmpfile, task_id=task_id))
        original_mtime = os.path.getmtime(self._tmpfile)
        with open(self._tmpfile, "w") as f:
            f.write("same mtime different content\n")
        os.utime(self._tmpfile, (original_mtime, original_mtime))

        result = json.loads(read_file_tool(self._tmpfile, task_id=task_id))

        self.assertNotIn("dedup", result)
        self.assertIn("same mtime different content", result.get("content", ""))

    @patch("tools.file_tools._get_file_ops")
    def test_raced_read_does_not_seed_dedup_or_silence_warn_mode(self, mock_ops):
        """If read output raced, the next read must not dedup stale conversation content."""
        fake = MagicMock()
        state = {"calls": 0}

        def racing_read(_path, _offset=1, _limit=500):
            state["calls"] += 1
            if state["calls"] == 1:
                with open(self._tmpfile, "w") as f:
                    f.write("changed during read\n")
                return _FakeReadResult(content="original line\n", total_lines=1, file_size=14)
            return _FakeReadResult(content="changed during read\n", total_lines=1, file_size=20)

        fake.read_file = racing_read
        fake.write_file = lambda path, content: _FakeWriteResult()
        fake.patch_replace = lambda path, old, new, replace_all=False: _FakePatchResult()
        mock_ops.return_value = fake

        with patch.dict(os.environ, {"HERMES_STALE_EDIT_MODE": "warn"}, clear=False):
            first = json.loads(read_file_tool(self._tmpfile, task_id="p12"))
            second = json.loads(read_file_tool(self._tmpfile, task_id="p12"))
            write = json.loads(write_file_tool(self._tmpfile, "new content", task_id="p12"))

        self.assertIn("original line", first.get("content", ""))
        self.assertNotIn("dedup", second)
        self.assertIn("changed during read", second.get("content", ""))
        self.assertNotIn("_warning", write)


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

    def test_helper_detects_hash_drift_when_mtime_is_unchanged(self):
        fd, path = tempfile.mkstemp(prefix="hash_drift_", suffix=".txt")
        try:
            os.close(fd)
            with open(path, "w") as f:
                f.write("before\n")
            mtime = os.path.getmtime(path)
            from tools.file_tools import _read_tracker_lock
            with _read_tracker_lock:
                _read_tracker["t1"] = {
                    "last_key": None, "consecutive": 0,
                    "read_history": set(), "dedup": {},
                    "read_timestamps": {path: mtime},
                    "read_fingerprints": {
                        path: {
                            "mtime": mtime,
                            "hash": _fingerprint("before\n"),
                            "partial": False,
                        }
                    },
                }
            with open(path, "w") as f:
                f.write("after\n")
            os.utime(path, (mtime, mtime))
            warning = _check_file_staleness(path, "t1")
            self.assertIsNotNone(warning)
            self.assertIn("modified since you last read", warning)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
