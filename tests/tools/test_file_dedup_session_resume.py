#!/usr/bin/env python3
"""
Tests for the read_file dedup cache carry-over for upstream issue #81725:
when the user runs ``/resume <other_session>`` mid-conversation, the agent
should not re-read files it already has in its rehydrated transcript.

The key this test pins down is that ``read_file_tool``'s dedup cache
(``tools.file_tools._read_tracker``) is keyed on the **per-turn
``effective_task_id``** — a fresh uuid generated in
``agent.turn_context.run_conversation`` and held on
``agent._current_task_id`` — NOT on the session id.  ``/resume`` rotates
``self.session_id`` but leaves the agent object in place between turns, so
a cache keyed on the old/new session ids would be a guaranteed no-op.

The fix restores the carry-over by giving the resumed session's first turn
the previous turn's cache: ``_handle_resume_command`` stashes the outgoing
turn's id on ``agent._pending_dedup_transfer_from``, and the next
``run_conversation`` hands that id (and its dedup entries) to the fresh
per-turn key via ``transfer_file_dedup``.

The end-to-end tests here drive REAL ``AIAgent`` turns through the real
tool-dispatch path (turn → ``handle_function_call`` → ``read_file_tool``),
so the dedup keys are genuinely the per-turn ``effective_task_id`` from the
agent, never hand-assigned strings.

Run with:
    python -m pytest tests/tools/test_file_dedup_session_resume.py -v
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

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


class _CountingOps:
    """FileOperations stand-in that counts real content re-fetches.

    A dedup hit inside ``read_file_tool`` returns the cheap "File unchanged"
    stub WITHOUT calling ``file_ops.read_file`` again, so counting those
    calls tells us whether the resumed turn truly skipped a re-read.
    """

    def __init__(self, content="hello\n"):
        self.content = content
        self.refetch_count = 0

    def read_file(self, path, offset=1, limit=500):
        self.refetch_count += 1
        return _FakeReadResult(
            content=self.content,
            total_lines=2,
            file_size=len(self.content.encode()),
        )

    def _add_line_numbers(self, text, offset):
        return text


def _make_safe_tempdir(prefix: str) -> str:
    """Create a temp dir outside macOS system-sensitive /private/var paths."""
    return tempfile.mkdtemp(prefix=prefix, dir=os.getcwd())


def _mock_tool_call(name="read_file", arguments="{}", call_id=None):
    """Mimic the OpenAI tool-call object the model returns for one call."""
    import uuid

    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="", finish_reason="tool_calls", tool_calls=None):
    """Mimic an OpenAI ChatCompletion response carrying tool_calls."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


class TestTransferFileDedup(unittest.TestCase):
    """``transfer_file_dedup`` is the cache-move helper used by the fix."""

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
        """No source tracker -> no transfer.  ``False`` returned, no
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


def _make_agent():
    """Build a real AIAgent whose model calls return canned read_file turns.

    The client is mocked so no network/model is touched, but the tool
    dispatch goes through the REAL ``read_file_tool`` via the REAL
    ``handle_function_call`` — so ``_read_tracker`` is populated under the
    per-turn ``effective_task_id`` that the agent itself generates.
    """
    from run_agent import AIAgent

    tool_def = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_def),
        patch("run_agent.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_turn(agent, read_path, ops):
    """Drive one real ``run_conversation`` turn that reads ``read_path``.

    The fake model emits a single ``read_file`` tool call, the tool loop
    dispatches it through the real path (so ``read_file_tool`` runs under
    this turn's ``effective_task_id``), then the model returns a plain
    "stop" so the turn completes.  Returns the turn's ``effective_task_id``
    (``agent._current_task_id``), which is the dedup-key source.
    """
    tc = _mock_tool_call(name="read_file", arguments=json.dumps({"path": read_path}))
    responses = [
        _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc]),
        _mock_response(content="done", finish_reason="stop", tool_calls=None),
    ]
    agent.client.chat.completions.create.side_effect = iter(responses)
    with (
        patch("tools.file_tools._get_file_ops", return_value=ops),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("read the file")
    return agent._current_task_id, result


class TestResumeDedupHandoffRealTurns(unittest.TestCase):
    """Dedup must survive a ``/resume`` across REAL agent turns.

    Both turns are genuine ``run_conversation`` turns: the dedup keys are
    the per-turn ``effective_task_id`` the agent generates, never hand-set.
    The resume is simulated exactly the way ``_handle_resume_command``
    drives it — by stashing the outgoing turn's id on
    ``agent._pending_dedup_transfer_from``.
    """

    def setUp(self):
        _read_tracker.clear()
        self._tmpdir = _make_safe_tempdir("hermes-resume-")
        self._tmpfile = os.path.join(self._tmpdir, "session_resume.txt")
        with open(self._tmpfile, "w") as f:
            f.write("original content\n")
        self._agent = _make_agent()
        self._ops = _CountingOps("original content\n")

    def tearDown(self):
        _read_tracker.clear()
        try:
            os.unlink(self._tmpfile)
            os.rmdir(self._tmpdir)
        except OSError:
            pass

    def test_resume_skips_reread_via_real_turns(self):
        """The core #81725 scenario, through real turns: read in session A,
        ``/resume`` to B, read the same file in B -> must hit the carried
        dedup cache and NOT re-fetch the file content."""
        # Turn 1 — old session reads the file (real dispatch).
        tid1, _ = _run_turn(self._agent, self._tmpfile, self._ops)
        self.assertIn(tid1, _read_tracker)
        # First read: no dedup hits yet (the file was fetched, not served from
        # cache). Assert the hit count is zero rather than a vacuous key check.
        self.assertEqual(len(_read_tracker[tid1].get("dedup_hits", {})), 0)
        # One real content fetch so far.
        self.assertEqual(self._ops.refetch_count, 1)

        # /resume A->B mid-conversation: exactly what _handle_resume_command
        # now does — stash the outgoing turn's effective_task_id.
        self._agent._pending_dedup_transfer_from = tid1

        # Turn 2 — resumed session reads the SAME file.  Its fresh
        # effective_task_id inherits the carried cache via transfer_file_dedup.
        tid2, _ = _run_turn(self._agent, self._tmpfile, self._ops)
        self.assertIsInstance(tid2, str)
        self.assertNotEqual(tid2, tid1, "resumed turns get distinct task ids")
        # The transfer MOVED the tracker to the new key (old key removed).
        self.assertNotIn(tid1, _read_tracker)
        self.assertIn(tid2, _read_tracker)
        self.assertIn(
            (self._tmpfile, 1, 500), _read_tracker[tid2]["dedup"],
            "carried dedup mapping must live under the resumed turn's key",
        )
        # The second read hit the dedup cache: no re-fetch of file content.
        self.assertEqual(
            self._ops.refetch_count, 1,
            "Resumed turn must hit the carried dedup cache and not "
            "re-read the file from disk",
        )

    def test_resume_mtime_change_invalidates_after_handoff(self):
        """Carrying the cache forward MUST NOT serve stale content when the
        file changed on disk between turns.  mtime is the correctness guard."""
        tid1, _ = _run_turn(self._agent, self._tmpfile, self._ops)
        self.assertEqual(self._ops.refetch_count, 1)

        self._agent._pending_dedup_transfer_from = tid1

        # File changes on disk between turns.
        time.sleep(0.05)
        with open(self._tmpfile, "w") as f:
            f.write("brand new content\n")
        self._ops.content = "brand new content\n"
        self._ops.refetch_count = 1  # reset the counter for a clean assertion

        # Resumed turn reads: mtime differs -> must re-fetch, no stale dedup.
        _run_turn(self._agent, self._tmpfile, self._ops)
        self.assertEqual(
            self._ops.refetch_count, 2,
            "A changed mtime between sessions must invalidate the carried "
            "dedup entry and force a fresh read",
        )

    def test_without_resume_flag_resumed_turn_rereads(self):
        """Control: a fresh turn gets a brand-new per-turn key with no
        carried state, so it re-reads.  Proves the cache is keyed per-turn
        (not session-stable) AND that the resume flag is what carries it."""
        tid1, _ = _run_turn(self._agent, self._tmpfile, self._ops)
        self.assertEqual(self._ops.refetch_count, 1)

        # NOTE: no _pending_dedup_transfer_from set — as if the turn was a
        # normal follow-up with no resume handoff.
        _run_turn(self._agent, self._tmpfile, self._ops)

        self.assertEqual(
            self._ops.refetch_count, 2,
            "Without the resume carry-over, a new turn must re-read the file",
        )


if __name__ == "__main__":
    unittest.main()
