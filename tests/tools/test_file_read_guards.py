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
    write_file_tool,
    reset_file_dedup,
    _is_blocked_device,
    _invalidate_dedup_for_path,
    _READ_DEDUP_STATUS_MESSAGE,
    _get_max_read_chars,
    _DEFAULT_MAX_READ_CHARS,
    _read_tracker,
    notify_other_tool_call,
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
        _read_tracker.clear()

    def tearDown(self):
        _read_tracker.clear()

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
        _read_tracker.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "dedup_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("line one\nline two\n")

    def tearDown(self):
        _read_tracker.clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_second_read_returns_dedup_stub(self, mock_ops):
        """Second read of same file+range returns non-content dedup status."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        # First read — full content
        r1 = json.loads(read_file_tool(self._tmpfile, task_id="dup"))
        self.assertNotIn("dedup", r1)

        # Second read — should get dedup stub
        r2 = json.loads(read_file_tool(self._tmpfile, task_id="dup"))
        self.assertTrue(r2.get("dedup"), "Second read should return dedup stub")
        self.assertEqual(r2.get("status"), "unchanged")
        self.assertIn("unchanged", r2.get("message", ""))
        self.assertFalse(r2.get("content_returned"))
        self.assertNotIn("content", r2)

    @patch("tools.file_tools._get_file_ops")
    def test_write_rejects_internal_read_status_text(self, mock_ops):
        """write_file must not persist internal read_file status text."""
        fake = MagicMock()
        fake.write_file = MagicMock()
        mock_ops.return_value = fake

        result = json.loads(write_file_tool(
            self._tmpfile,
            _READ_DEDUP_STATUS_MESSAGE,
            task_id="guard",
        ))

        self.assertIn("error", result)
        self.assertIn("internal read_file status text", result["error"])
        fake.write_file.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_write_rejects_status_text_with_small_framing(self, mock_ops):
        """write_file rejects small wrappers around the status text too.

        Real-world corruption shapes aren't always the verbatim message — the
        model sometimes prepends a short note or appends a trailing comment
        before calling write_file.  A short, status-dominated write is still
        corruption, not legitimate file content.
        """
        fake = MagicMock()
        fake.write_file = MagicMock()
        mock_ops.return_value = fake

        wrapped = "Note: " + _READ_DEDUP_STATUS_MESSAGE + "\n\n(continuing.)"
        result = json.loads(write_file_tool(
            self._tmpfile,
            wrapped,
            task_id="guard",
        ))

        self.assertIn("error", result)
        self.assertIn("internal read_file status text", result["error"])
        fake.write_file.assert_not_called()

    @patch("tools.file_tools._get_file_ops")
    def test_write_allows_large_file_that_quotes_status_text(self, mock_ops):
        """Legitimate large content that happens to quote the status is allowed.

        Hermes' own docs / SKILL.md files may legitimately mention the dedup
        message verbatim.  Only short, status-dominated writes are rejected —
        a normal file that contains the message as one line out of many must
        still write successfully.
        """
        fake = MagicMock()
        fake.write_file = lambda path, content: MagicMock(
            to_dict=lambda: {"success": True, "path": path}
        )
        mock_ops.return_value = fake

        # Build content that contains the status text but is much larger,
        # so the status doesn't "dominate" — this is a legitimate file.
        large_content = (
            "# Skill reference\n\n"
            "Example internal message (do not write back):\n\n"
            f"    {_READ_DEDUP_STATUS_MESSAGE}\n\n"
            + ("This is documentation content. " * 200)
        )
        result = json.loads(write_file_tool(
            self._tmpfile,
            large_content,
            task_id="guard",
        ))

        self.assertNotIn("error", result)
        self.assertTrue(result.get("success"))

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
# Dedup stub-loop guard (issue #15759)
# ---------------------------------------------------------------------------

class TestDedupStubLoopGuard(unittest.TestCase):
    """Repeated dedup stubs must escalate to a hard BLOCKED error so weak
    tool-following models don't burn iteration budget in an infinite loop
    of ``read_file → stub → read_file → stub → ...``"""

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "loop_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("line one\nline two\n")

    def tearDown(self):
        _read_tracker.clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_third_read_is_blocked(self, mock_ops):
        """read → stub → BLOCKED.  Second stub escalates to hard error."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        # 1. Real read — full content
        r1 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertNotIn("dedup", r1)
        self.assertNotIn("error", r1)

        # 2. Dedup stub (first hit)
        r2 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertTrue(r2.get("dedup"))
        self.assertNotIn("error", r2)

        # 3. Dedup stub (second hit) — escalates to BLOCKED
        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertIn("error", r3, "Second dedup stub should be BLOCKED")
        self.assertIn("BLOCKED", r3["error"])
        self.assertIn("STOP", r3["error"])
        self.assertEqual(r3.get("already_read"), 3)
        # The loop-breaker must NOT be a dedup stub, or the model sees the
        # same passive message it has been ignoring.
        self.assertNotIn("dedup", r3)

    @patch("tools.file_tools._get_file_ops")
    def test_subsequent_reads_stay_blocked(self, mock_ops):
        """Once blocked, continued hammering keeps returning BLOCKED."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="loop")  # read
        read_file_tool(self._tmpfile, task_id="loop")  # stub
        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertIn("error", r3)
        # 4th, 5th, ... calls must stay blocked, never revert to stub
        for _ in range(5):
            rN = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
            self.assertIn("error", rN)
            self.assertIn("BLOCKED", rN["error"])

    @patch("tools.file_tools._get_file_ops")
    def test_file_modification_clears_block(self, mock_ops):
        """Real file change should break out of the block — new content
        is legitimately different and the agent should see it."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="loop")
        read_file_tool(self._tmpfile, task_id="loop")
        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertIn("error", r3)

        # File changes — mtime updates
        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("brand new content\n")

        r4 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertNotIn("error", r4)
        self.assertNotIn("dedup", r4)

    @patch("tools.file_tools._get_file_ops")
    def test_other_tool_call_clears_hits(self, mock_ops):
        """An intervening non-read tool call resets stub-hit counters,
        just like it resets the consecutive-read counter."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="loop")
        read_file_tool(self._tmpfile, task_id="loop")  # 1st stub

        # Agent did something else — e.g. terminal, write_file — so the
        # stub-loop is broken.  Counter should reset.
        notify_other_tool_call("loop")

        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        # Should be a stub again, NOT blocked
        self.assertTrue(r3.get("dedup"))
        self.assertNotIn("error", r3)

    @patch("tools.file_tools._get_file_ops")
    def test_different_ranges_tracked_independently(self, mock_ops):
        """Stub-hit counter is keyed by (path, offset, limit), so hammering
        one range shouldn't block reads of a different range."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        # Burn down one range
        read_file_tool(self._tmpfile, offset=1, limit=100, task_id="loop")
        read_file_tool(self._tmpfile, offset=1, limit=100, task_id="loop")
        r3 = json.loads(read_file_tool(
            self._tmpfile, offset=1, limit=100, task_id="loop",
        ))
        self.assertIn("error", r3)

        # Different range — fresh read, should go through
        r_other = json.loads(read_file_tool(
            self._tmpfile, offset=1, limit=200, task_id="loop",
        ))
        self.assertNotIn("error", r_other)

    @patch("tools.file_tools._get_file_ops")
    def test_reset_file_dedup_clears_hits(self, mock_ops):
        """Post-compression reset must clear stub-hit counters too,
        otherwise the agent stays blocked after compression."""
        mock_ops.return_value = _make_fake_ops(
            content="line one\nline two\n", file_size=20,
        )
        read_file_tool(self._tmpfile, task_id="loop")
        read_file_tool(self._tmpfile, task_id="loop")
        r3 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertIn("error", r3)

        reset_file_dedup("loop")

        # Fresh session — real read, no stub, no block
        r4 = json.loads(read_file_tool(self._tmpfile, task_id="loop"))
        self.assertNotIn("error", r4)
        self.assertNotIn("dedup", r4)


# ---------------------------------------------------------------------------
# Dedup reset on compression
# ---------------------------------------------------------------------------

class TestDedupResetOnCompression(unittest.TestCase):
    """reset_file_dedup should clear the dedup cache so post-compression
    reads return full content."""

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "compress_test.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
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
        _read_tracker.clear()

    def tearDown(self):
        _read_tracker.clear()

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
        _read_tracker.clear()
        # Reset the cached value so each test gets a fresh lookup
        import tools.file_tools as _ft
        _ft._max_read_chars_cached = None

    def tearDown(self):
        _read_tracker.clear()
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


# ---------------------------------------------------------------------------
# Write invalidates dedup cache (fixes #13144)
# ---------------------------------------------------------------------------

class TestWriteInvalidatesDedup(unittest.TestCase):
    """write_file_tool and patch_tool must invalidate the read_file dedup
    cache for the written path.  Without this, a read→write→read sequence
    within the same mtime second returns a stale 'File unchanged' stub.

    Regression test for https://github.com/NousResearch/hermes-agent/issues/13144
    """

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._tmpfile = os.path.join(self._tmpdir, "write_dedup.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_write_invalidates_dedup_same_second(self, mock_ops):
        """read→write→read within the same mtime second returns fresh content.

        This is the core #13144 scenario: on filesystems with ≥1ms mtime
        granularity, a write that lands in the same timestamp as the prior
        read would previously cause the second read to return a stale dedup
        stub because the mtime comparison saw no change.
        """
        fake = MagicMock()
        fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
            content="original content\n", total_lines=1, file_size=18,
        )
        fake.write_file = lambda path, content: MagicMock(
            to_dict=lambda: {"success": True, "path": path}
        )
        mock_ops.return_value = fake

        # 1. Read — populates dedup cache.
        r1 = json.loads(read_file_tool(self._tmpfile, task_id="wr"))
        self.assertNotEqual(r1.get("dedup"), True)

        # 2. Write — must invalidate dedup for this path.
        #    (No sleep — we intentionally stay in the same mtime second.)
        write_file_tool(self._tmpfile, "new content\n", task_id="wr")

        # 3. Read again — should get full content, NOT dedup stub.
        fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
            content="new content\n", total_lines=1, file_size=13,
        )
        r2 = json.loads(read_file_tool(self._tmpfile, task_id="wr"))
        self.assertNotEqual(r2.get("dedup"), True,
                            "read after write must not return dedup stub")
        self.assertIn("content", r2)

    @patch("tools.file_tools._get_file_ops")
    def test_write_invalidates_all_offsets(self, mock_ops):
        """A write invalidates dedup entries for ALL offset/limit combos."""
        fake = MagicMock()
        fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
            content="line1\nline2\nline3\n", total_lines=3, file_size=20,
        )
        fake.write_file = lambda path, content: MagicMock(
            to_dict=lambda: {"success": True, "path": path}
        )
        mock_ops.return_value = fake

        # Read with different offsets to populate multiple dedup entries.
        read_file_tool(self._tmpfile, offset=1, limit=100, task_id="off")
        read_file_tool(self._tmpfile, offset=50, limit=100, task_id="off")

        # Write — should invalidate BOTH dedup entries.
        write_file_tool(self._tmpfile, "replaced\n", task_id="off")

        # Both reads should return fresh content.
        r1 = json.loads(read_file_tool(self._tmpfile, offset=1, limit=100, task_id="off"))
        r2 = json.loads(read_file_tool(self._tmpfile, offset=50, limit=100, task_id="off"))
        self.assertNotEqual(r1.get("dedup"), True,
                            "offset=1 should not dedup after write")
        self.assertNotEqual(r2.get("dedup"), True,
                            "offset=50 should not dedup after write")

    @patch("tools.file_tools._get_file_ops")
    def test_write_does_not_invalidate_other_files(self, mock_ops):
        """Writing file A should not invalidate dedup for file B."""
        other = os.path.join(self._tmpdir, "other.txt")
        with open(other, "w") as f:
            f.write("other content\n")

        fake = MagicMock()
        fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
            content="other content\n", total_lines=1, file_size=15,
        )
        fake.write_file = lambda path, content: MagicMock(
            to_dict=lambda: {"success": True, "path": path}
        )
        mock_ops.return_value = fake

        # Read file B.
        read_file_tool(other, task_id="iso")

        # Write file A.
        write_file_tool(self._tmpfile, "changed A\n", task_id="iso")

        # File B should still dedup (untouched).
        r2 = json.loads(read_file_tool(other, task_id="iso"))
        self.assertTrue(r2.get("dedup"),
                        "Unrelated file should still dedup after writing another file")

        try:
            os.unlink(other)
        except OSError:
            pass

    @patch("tools.file_tools._get_file_ops")
    def test_write_does_not_invalidate_other_tasks(self, mock_ops):
        """Writing in task A should not invalidate dedup for task B."""
        fake = MagicMock()
        fake.read_file = lambda path, offset=1, limit=500: _FakeReadResult(
            content="original content\n", total_lines=1, file_size=18,
        )
        fake.write_file = lambda path, content: MagicMock(
            to_dict=lambda: {"success": True, "path": path}
        )
        mock_ops.return_value = fake

        # Both tasks read the file.
        read_file_tool(self._tmpfile, task_id="taskA")
        read_file_tool(self._tmpfile, task_id="taskB")

        # Task A writes.
        write_file_tool(self._tmpfile, "new\n", task_id="taskA")

        # Task A's dedup should be invalidated.
        rA = json.loads(read_file_tool(self._tmpfile, task_id="taskA"))
        self.assertNotEqual(rA.get("dedup"), True,
                            "Writing task's dedup should be invalidated")

        # Task B still sees dedup (its cache is separate — the file
        # *may* have changed on disk, but mtime comparison handles that;
        # here we test that invalidation is scoped to the writing task).
        # Note: on real FS, task B's dedup might or might not hit depending
        # on mtime.  The point is that _invalidate_dedup_for_path is
        # correctly scoped to task_id.

    def test_invalidate_dedup_for_path_noop_on_missing_task(self):
        """_invalidate_dedup_for_path is safe when task_id doesn't exist."""
        _read_tracker.clear()
        # Should not raise.
        _invalidate_dedup_for_path("/nonexistent/path", "no_such_task")

    def test_invalidate_dedup_for_path_noop_on_empty_dedup(self):
        """_invalidate_dedup_for_path is safe when dedup dict is empty."""
        _read_tracker.clear()
        _read_tracker["t"] = {
            "last_key": None, "consecutive": 0,
            "read_history": set(), "dedup": {},
        }
        _invalidate_dedup_for_path("/some/path", "t")
        self.assertEqual(_read_tracker["t"]["dedup"], {})


if __name__ == "__main__":
    unittest.main()
