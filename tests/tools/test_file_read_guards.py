#!/usr/bin/env python3
"""
Tests for read_file_tool safety guards: device-path blocking,
character-count limits, file deduplication, and dedup reset on
context compression.

Run with:  python -m pytest tests/tools/test_file_read_guards.py -v
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from tools.file_tools import (
    read_file_tool,
    clear_read_tracker,
    reset_file_dedup,
    _is_blocked_device,
    _get_max_read_chars,
    _DEFAULT_MAX_READ_CHARS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeReadResult:
    """Minimal stand-in for FileOperations.read_file return value."""
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


def _make_fake_ops(content="hello\n", total_lines=1, file_size=6):
    fake = MagicMock()
    fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
        content=content, total_lines=total_lines, file_size=file_size,
    )
    return fake


# ---------------------------------------------------------------------------
# Device path blocking
# ---------------------------------------------------------------------------

class TestDevicePathBlocking(unittest.TestCase):
    """Paths like /dev/zero should be rejected before any I/O."""

    def test_blocked_device_detection(self):
        for dev in ("/dev/zero", "/dev/random", "/dev/urandom", "/dev/stdin",
                     "/dev/tty", "/dev/console", "/dev/stdout", "/dev/stderr",
                     "/dev/fd/0", "/dev/fd/1", "/dev/fd/2"):
            self.assertTrue(_is_blocked_device(dev), f"{dev} should be blocked")

    def test_safe_device_not_blocked(self):
        self.assertFalse(_is_blocked_device("/dev/null"))
        self.assertFalse(_is_blocked_device("/dev/sda1"))

    def test_proc_fd_blocked(self):
        self.assertTrue(_is_blocked_device("/proc/self/fd/0"))
        self.assertTrue(_is_blocked_device("/proc/12345/fd/2"))

    def test_proc_fd_other_not_blocked(self):
        self.assertFalse(_is_blocked_device("/proc/self/fd/3"))
        self.assertFalse(_is_blocked_device("/proc/self/maps"))

    def test_normal_files_not_blocked(self):
        self.assertFalse(_is_blocked_device("/tmp/test.py"))
        self.assertFalse(_is_blocked_device("/home/user/.bashrc"))

    def test_read_file_tool_rejects_device(self):
        """read_file_tool returns an error without any file I/O."""
        result = json.loads(read_file_tool("/dev/zero", task_id="dev_test"))
        self.assertIn("error", result)
        self.assertIn("device file", result["error"])


# ---------------------------------------------------------------------------
# Character-count limits
# ---------------------------------------------------------------------------

class TestCharacterCountGuard(unittest.TestCase):
    """Large reads should be rejected with guidance to use offset/limit."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops")
    @patch("tools.file_tools._get_max_read_chars", return_value=_DEFAULT_MAX_READ_CHARS)
    def test_oversized_read_rejected(self, _mock_limit, mock_ops):
        """A read that returns >max chars is rejected."""
        big_content = "x" * (_DEFAULT_MAX_READ_CHARS + 1)
        mock_ops.return_value = _make_fake_ops(
            content=big_content,
            total_lines=5000,
            file_size=len(big_content) + 100,  # bigger than content
        )
        result = json.loads(read_file_tool("/tmp/huge.txt", task_id="big"))
        self.assertIn("error", result)
        self.assertIn("safety limit", result["error"])
        self.assertIn("offset and limit", result["error"])
        self.assertIn("total_lines", result)

    @patch("tools.file_tools._get_file_ops")
    def test_small_read_not_rejected(self, mock_ops):
        """Normal-sized reads pass through fine."""
        mock_ops.return_value = _make_fake_ops(content="short\n", file_size=6)
        result = json.loads(read_file_tool("/tmp/small.txt", task_id="small"))
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops")
    @patch("tools.file_tools._get_max_read_chars", return_value=_DEFAULT_MAX_READ_CHARS)
    def test_content_under_limit_passes(self, _mock_limit, mock_ops):
        """Content just under the limit should pass through fine."""
        mock_ops.return_value = _make_fake_ops(
            content="y" * (_DEFAULT_MAX_READ_CHARS - 1),
            file_size=_DEFAULT_MAX_READ_CHARS - 1,
        )
        result = json.loads(read_file_tool("/tmp/justunder.txt", task_id="under"))
        self.assertNotIn("error", result)
        self.assertIn("content", result)


# ---------------------------------------------------------------------------
# File deduplication
# ---------------------------------------------------------------------------

class TestFileDedup(unittest.TestCase):
    """Re-reading an unchanged file should return a lightweight stub."""

    def setUp(self):
        clear_read_tracker()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "dedup_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("line one\nline two\n")

    def tearDown(self):
        clear_read_tracker()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_second_read_returns_dedup_stub(self, mock_ops):
        """Second read of same file+range returns dedup stub."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        # First read — full content
        r1 = json.loads(read_file_tool(self._tmpfile, task_id="dup"))
        self.assertNotIn("dedup", r1)

        # Second read — should get dedup stub
        r2 = json.loads(read_file_tool(self._tmpfile, task_id="dup"))
        self.assertTrue(r2.get("dedup"), "Second read should return dedup stub")
        self.assertIn("unchanged", r2.get("content", ""))

    @patch("tools.file_tools._get_file_ops")
    def test_modified_file_not_deduped(self, mock_ops):
        """After the file is modified, dedup returns full content."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="mod")

        # Modify the file — ensure mtime changes
        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("changed content\n")

        r2 = json.loads(read_file_tool(self._tmpfile, task_id="mod"))
        self.assertNotEqual(r2.get("dedup"), True, "Modified file should not dedup")

    @patch("tools.file_tools._get_file_ops")
    def test_different_range_not_deduped(self, mock_ops):
        """Same file but different offset/limit should not dedup."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, offset=1, limit=500, task_id="rng")

        r2 = json.loads(read_file_tool(
            self._tmpfile, offset=10, limit=500, task_id="rng",
        ))
        self.assertNotEqual(r2.get("dedup"), True)

    @patch("tools.file_tools._get_file_ops")
    def test_different_task_not_deduped(self, mock_ops):
        """Different task_ids have separate dedup caches."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="task_a")

        r2 = json.loads(read_file_tool(self._tmpfile, task_id="task_b"))
        self.assertNotEqual(r2.get("dedup"), True)


# ---------------------------------------------------------------------------
# Dedup reset on compression
# ---------------------------------------------------------------------------

class TestDedupResetOnCompression(unittest.TestCase):
    """reset_file_dedup should clear the dedup cache so post-compression
    reads return full content."""

    def setUp(self):
        clear_read_tracker()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "compress_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        clear_read_tracker()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_reset_clears_dedup(self, mock_ops):
        """After reset_file_dedup, the same read returns full content."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        # First read — populates dedup cache
        read_file_tool(self._tmpfile, task_id="comp")

        # Verify dedup works before reset
        r_dedup = json.loads(read_file_tool(self._tmpfile, task_id="comp"))
        self.assertTrue(r_dedup.get("dedup"), "Should dedup before reset")

        # Simulate compression
        reset_file_dedup("comp")

        # Read again — should get full content
        r_post = json.loads(read_file_tool(self._tmpfile, task_id="comp"))
        self.assertNotEqual(r_post.get("dedup"), True,
                            "Post-compression read should return full content")

    @patch("tools.file_tools._get_file_ops")
    def test_reset_all_tasks(self, mock_ops):
        """reset_file_dedup(None) clears all tasks."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        read_file_tool(self._tmpfile, task_id="t1")
        read_file_tool(self._tmpfile, task_id="t2")

        reset_file_dedup()  # no task_id — clear all

        r1 = json.loads(read_file_tool(self._tmpfile, task_id="t1"))
        r2 = json.loads(read_file_tool(self._tmpfile, task_id="t2"))
        self.assertNotEqual(r1.get("dedup"), True)
        self.assertNotEqual(r2.get("dedup"), True)

    @patch("tools.file_tools._get_file_ops")
    def test_reset_preserves_loop_detection(self, mock_ops):
        """reset_file_dedup does NOT affect the consecutive-read counter."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        # Build up consecutive count (read 1 and 2)
        read_file_tool(self._tmpfile, task_id="loop")
        # 2nd read is deduped — doesn't increment consecutive counter
        read_file_tool(self._tmpfile, task_id="loop")

        reset_file_dedup("loop")

        # 3rd read — counter should still be at 2 from before reset
        # (dedup was hit for read 2, but consecutive counter was 1 for that)
        # After reset, this read goes through full path, incrementing to 2
        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        # Should NOT be blocked or warned — counter restarted since dedup
        # intercepted reads before they reached the counter
        self.assertNotIn("error", r3)


# ---------------------------------------------------------------------------
# Large-file hint
# ---------------------------------------------------------------------------

class TestLargeFileHint(unittest.TestCase):
    """Large truncated files should include a hint about targeted reads."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops")
    def test_large_truncated_file_gets_hint(self, mock_ops):
        content = "line\n" * 400  # 2000 chars, small enough to pass char guard
        fake = _make_fake_ops(content=content, total_lines=10000, file_size=600_000)
        # Make to_dict return truncated=True
        orig_read = fake.read_file
        def patched_read(path, offset=1, limit=500):
            r = orig_read(path, offset, limit)
            orig_to_dict = r.to_dict
            def new_to_dict():
                d = orig_to_dict()
                d["truncated"] = True
                return d
            r.to_dict = new_to_dict
            return r
        fake.read_file = patched_read
        mock_ops.return_value = fake

        result = json.loads(read_file_tool("/tmp/bigfile.log", task_id="hint"))
        self.assertIn("_hint", result)
        self.assertIn("section you need", result["_hint"])


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------

class TestConfigOverride(unittest.TestCase):
    """file_read_max_chars in config.yaml should control the char guard."""

    def setUp(self):
        clear_read_tracker()
        # Reset the cached value so each test gets a fresh lookup
        import tools.file_tools as _ft
        _ft._max_read_chars_cached = None

    def tearDown(self):
        clear_read_tracker()
        import tools.file_tools as _ft
        _ft._max_read_chars_cached = None

    @patch("tools.file_tools._get_file_ops")
    @patch("hermes_cli.config.load_config", return_value={"file_read_max_chars": 50})
    def test_custom_config_lowers_limit(self, _mock_cfg, mock_ops):
        """A config value of 50 should reject reads over 50 chars."""
        mock_ops.return_value = _make_fake_ops(content="x" * 60, file_size=60)
        result = json.loads(read_file_tool("/tmp/cfgtest.txt", task_id="cfg1"))
        self.assertIn("error", result)
        self.assertIn("safety limit", result["error"])
        self.assertIn("50", result["error"])  # should show the configured limit

    @patch("tools.file_tools._get_file_ops")
    @patch("hermes_cli.config.load_config", return_value={"file_read_max_chars": 500_000})
    def test_custom_config_raises_limit(self, _mock_cfg, mock_ops):
        """A config value of 500K should allow reads up to 500K chars."""
        # 200K chars would be rejected at the default 100K but passes at 500K
        mock_ops.return_value = _make_fake_ops(
            content="y" * 200_000, file_size=200_000,
        )
        result = json.loads(read_file_tool("/tmp/cfgtest2.txt", task_id="cfg2"))
        self.assertNotIn("error", result)
        self.assertIn("content", result)


if __name__ == "__main__":
    unittest.main()
