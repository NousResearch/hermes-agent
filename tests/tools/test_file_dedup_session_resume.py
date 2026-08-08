#!/usr/bin/env python3
"""
Tests for the per-session dedup cache carry-over added for upstream
issue #81725: when the user runs ``/resume <other_session>`` mid-turn,
``self.session_id`` rotates, but the agent's read_file dedup cache is
keyed on that same id.  Without a handoff, the resumed session starts
with an empty cache and the agent re-reads files it already has in its
rehydrated transcript — pulling every attachment through the gateway
again.

Run with:
    python -m pytest tests/tools/test_file_dedup_session_resume.py -v
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from tools.file_tools import (
    read_file_tool,
    reset_file_dedup,
    transfer_file_dedup,
    _read_tracker,
)


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


def _make_safe_tempdir(prefix: str) -> str:
    """Create a temp dir outside macOS system-sensitive /private/var paths."""
    return tempfile.mkdtemp(prefix=prefix, dir=os.getcwd())


class TestTransferFileDedup(unittest.TestCase):
    """``transfer_file_dedup`` is the handoff helper added for #81725."""

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = _make_safe_tempdir("hermes-resume-")
        self._tmpfile = os.path.join(self._tmpdir, "session_resume.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    # ── guard cases ──────────────────────────────────────────────────

    def test_transfer_noop_when_old_id_missing(self):
        """No source tracker → no transfer.  ``False`` returned, no
        surprise entries left behind on the target side."""
        result = transfer_file_dedup("never-existed", "new-id")
        self.assertFalse(result)
        self.assertNotIn("new-id", _read_tracker)

    def test_transfer_noop_on_empty_ids(self):
        result = transfer_file_dedup("", "new-id")
        self.assertFalse(result)
        result = transfer_file_dedup("old-id", "")
        self.assertFalse(result)
        result = transfer_file_dedup("same", "same")
        self.assertFalse(result)

    def test_transfer_does_not_clobber_existing_target(self):
        """If a parallel session already owns the target id, the transfer
        refuses to overwrite it.  Prevents stealing cache state from a
        concurrent session that happens to share the id."""
        _read_tracker["old"] = {
            "last_key": None, "consecutive": 0,
            "read_history": set(), "dedup": {("x", 1, 500): 1.0},
        }
        _read_tracker["new"] = {
            "last_key": None, "consecutive": 0,
            "read_history": set(), "dedup": {("y", 1, 500): 2.0},
        }
        result = transfer_file_dedup("old", "new")
        self.assertFalse(result)
        # Source survives (caller may retry later); target untouched.
        self.assertIn("old", _read_tracker)
        self.assertIn("new", _read_tracker)
        self.assertEqual(
            _read_tracker["new"]["dedup"], {("y", 1, 500): 2.0},
        )


class TestResumeDedupHandoff(unittest.TestCase):
    """End-to-end: dedup must survive a session_id rotation."""

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = _make_safe_tempdir("hermes-resume-")
        self._tmpfile = os.path.join(self._tmpdir, "session_resume.txt")
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
    def test_resume_skips_reread_via_handoff(self, mock_ops):
        """The core #81725 scenario: read in session A, ``/resume`` to
        session B, read the same file in B → must hit the dedup cache and
        not re-fetch the bytes."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )

        # Session A reads the file.
        first = json.loads(read_file_tool(self._tmpfile, task_id="A"))
        self.assertNotIn("error", first)
        self.assertNotEqual(first.get("dedup"), True)

        # User runs ``/resume B`` mid-turn — session_id rotates.
        transferred = transfer_file_dedup("A", "B")
        self.assertTrue(transferred)

        # Session B reads the same file: must dedup (no re-fetch).
        second = json.loads(read_file_tool(self._tmpfile, task_id="B"))
        self.assertTrue(
            second.get("dedup"),
            "After resume handoff, second read in session B should "
            "return the cached 'unchanged' stub instead of re-reading",
        )
        self.assertNotIn("error", second)

    @patch("tools.file_tools._get_file_ops")
    def test_resume_mtime_change_invalidates_after_handoff(self, mock_ops):
        """Carrying the cache forward MUST NOT serve stale content when
        the file changed on disk between sessions.  mtime is the
        fallback correctness guard."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        # Session A reads → populates dedup.
        read_file_tool(self._tmpfile, task_id="A")
        # Verify dedup works in session A.
        self.assertTrue(
            json.loads(read_file_tool(self._tmpfile, task_id="A")).get("dedup"),
        )

        # /resume B (carry cache forward).
        transfer_file_dedup("A", "B")

        # File on disk changes.
        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("brand new content\n")

        # Session B read must NOT dedup — the file changed.
        result = json.loads(read_file_tool(self._tmpfile, task_id="B"))
        self.assertNotEqual(
            result.get("dedup"), True,
            "File mtime changed between sessions — dedup must "
            "fall through and return fresh content",
        )
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops")
    def test_unrelated_session_isolated_after_handoff(self, mock_ops):
        """The handoff only moves state from the source session; other
        session ids must keep their independent caches."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        # Session A and session C both read.
        read_file_tool(self._tmpfile, task_id="A")
        read_file_tool(self._tmpfile, task_id="C")

        # /resume moves A's cache to B.
        transfer_file_dedup("A", "B")

        # C is unaffected.
        self.assertIn("C", _read_tracker)
        self.assertNotIn("A", _read_tracker)
        self.assertIn("B", _read_tracker)
        # C still sees its own dedup.
        self.assertTrue(
            json.loads(read_file_tool(self._tmpfile, task_id="C")).get("dedup"),
        )

    @patch("tools.file_tools._get_file_ops")
    def test_transfer_then_reset_file_dedup_clears_new_session(self, mock_ops):
        """The handoff is reversible via reset_file_dedup — a subsequent
        context compression on session B clears the carried-over entries
        like any native cache."""
        mock_ops.return_value = _make_fake_ops(
            content="original content\n", file_size=18,
        )
        read_file_tool(self._tmpfile, task_id="A")
        transfer_file_dedup("A", "B")

        # Session B dedups before reset.
        self.assertTrue(
            json.loads(read_file_tool(self._tmpfile, task_id="B")).get("dedup"),
        )

        # Compression-style reset.
        reset_file_dedup("B")

        # After reset, session B re-reads (the same way as a native session).
        result = json.loads(read_file_tool(self._tmpfile, task_id="B"))
        self.assertNotEqual(result.get("dedup"), True)


if __name__ == "__main__":
    unittest.main()