#!/usr/bin/env python3
"""Tests for subagent checkpoint / orphan adoption (live upgrade).

Verifies that:
  - checkpoint_active_subagents() writes a valid JSON file with full state
  - adopt_orphaned_subagents() consumes, removes, and stores orphans
  - get_pending_orphans() returns stored orphan data
  - recreate_pending_subagents() re-delegates orphans (with mock)
  - No checkpoint file means adopt_orphaned_subagents() is a no-op
  - Corrupted checkpoint is handled gracefully

Run with:  python -m pytest tests/tools/test_delegate_checkpoint.py -v
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.delegate_tool import (
    _CHECKPOINT_FILE_NAME,
    _checkpoint_dir,
    _checkpoint_path,
    _pending_orphans,
    _remove_checkpoint_safe,
    _safe_serialize_messages,
    adopt_orphaned_subagents,
    checkpoint_active_subagents,
    list_active_subagents,
    get_pending_orphans,
    recreate_pending_subagents,
    _register_subagent,
    _active_subagents,
    _active_subagents_lock,
)


def _clean_active_subagents():
    """Remove all entries from the module-level active subagents dict."""
    with _active_subagents_lock:
        _active_subagents.clear()
    global _pending_orphans
    try:
        from tools.delegate_tool import _pending_orphans as _po
        _po.clear()
    except (ImportError, AttributeError):
        pass


class TestSafeSerializeMessages(unittest.TestCase):
    """_safe_serialize_messages edge cases."""

    def test_none_messages(self):
        self.assertEqual(_safe_serialize_messages(None), [])

    def test_empty_list(self):
        self.assertEqual(_safe_serialize_messages([]), [])

    def test_skips_non_dict(self):
        result = _safe_serialize_messages(["bad", 42, {"role": "user", "content": "hello"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "user")

    def test_truncates_long_content(self):
        long_content = "x" * 20000
        result = _safe_serialize_messages([{"role": "user", "content": long_content}])
        self.assertIn("... [truncated]", result[0]["content"])
        self.assertLess(len(result[0]["content"]), 20000)

    def test_strips_callbacks(self):
        result = _safe_serialize_messages([{
            "role": "assistant",
            "content": "OK",
            "callback": lambda: None,
            "stream": object(),
        }])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "assistant")
        self.assertNotIn("callback", result[0])
        self.assertNotIn("stream", result[0])

    def test_handles_binary_bytes(self):
        result = _safe_serialize_messages([{"role": "tool", "content": b"\x00\x01\x02"}])
        result_content = result[0]["content"]
        # bytes should be decoded, not crash
        self.assertIsInstance(result_content, str)

    def test_handles_tool_calls(self):
        result = _safe_serialize_messages([{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "function": {"name": "web_search", "arguments": "{}"},
            }],
        }])
        self.assertEqual(len(result), 1)
        self.assertIn("tool_calls", result[0])


class TestCheckpointPath(unittest.TestCase):
    """Path resolution for the checkpoint file."""

    @patch("hermes_constants.get_hermes_home")
    def test_checkpoint_path_under_hermes_home(self, mock_home):
        mock_home.return_value = Path("/tmp/hermes_test")
        path = _checkpoint_path()
        self.assertIn("state", path)
        self.assertIn(_CHECKPOINT_FILE_NAME, path)
        self.assertTrue(path.endswith(_CHECKPOINT_FILE_NAME))


class TestCheckpointWrite(unittest.TestCase):
    """checkpoint_active_subagents serialization."""

    def setUp(self):
        _clean_active_subagents()

    def tearDown(self):
        _clean_active_subagents()
        p = _checkpoint_path()
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    @patch("hermes_constants.get_hermes_home")
    def test_no_subagents_returns_none(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            result = checkpoint_active_subagents()
            self.assertIsNone(result)

    @patch("hermes_constants.get_hermes_home")
    def test_writes_checkpoint_file_v2(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            # Register a fake subagent WITH a mock agent that has messages
            mock_agent = MagicMock()
            mock_agent._session_messages = [
                {"role": "user", "content": "Do research"},
                {"role": "assistant", "content": "Researching...", "tool_calls": []},
                {"role": "tool", "content": "Result: data"},
            ]
            mock_agent._subagent_goal = "Research task context"
            _register_subagent({
                "subagent_id": "test-sa-1",
                "parent_id": None,
                "depth": 0,
                "goal": "Test task with full state",
                "model": "test-model",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                "agent": mock_agent,
            })
            result = checkpoint_active_subagents()
            self.assertIsNotNone(result)
            self.assertTrue(os.path.isfile(result))

            # Validate JSON content includes full state
            with open(result) as f:
                payload = json.load(f)
            self.assertIn("pid", payload)
            self.assertIn("parent_pid", payload)
            self.assertIn("created_at", payload)
            self.assertIn("version", payload)
            self.assertEqual(payload["version"], 2)
            self.assertIn("subagents", payload)
            self.assertEqual(len(payload["subagents"]), 1)

            sa = payload["subagents"][0]
            self.assertEqual(sa["subagent_id"], "test-sa-1")
            self.assertEqual(sa["goal"], "Test task with full state")
            # Should have saved messages
            self.assertIn("saved_messages", sa)
            self.assertGreater(sa["saved_message_count"], 0)
            # Should have resolved_context
            self.assertIn("resolved_context", sa)
            self.assertEqual(sa["resolved_context"], "Research task context")

    @patch("hermes_constants.get_hermes_home")
    def test_writes_v2_even_without_agent_object(self, mock_home):
        """Version 2 checkpoint is written even when agent field is missing."""
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            _register_subagent({
                "subagent_id": "test-sa-1",
                "parent_id": None,
                "depth": 0,
                "goal": "Test task",
                "model": "test-model",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 0,
                # no agent field
            })
            result = checkpoint_active_subagents()
            self.assertIsNotNone(result)
            with open(result) as f:
                payload = json.load(f)
            self.assertEqual(payload["version"], 2)
            sa = payload["subagents"][0]
            self.assertEqual(sa["subagent_id"], "test-sa-1")
            # When no agent, saved_messages is absent
            self.assertNotIn("saved_messages", sa)

    @patch("hermes_constants.get_hermes_home")
    def test_multiple_subagents(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            _register_subagent({
                "subagent_id": "sa-a",
                "parent_id": None,
                "depth": 0,
                "goal": "Research A",
                "model": "gpt-4",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 2,
            })
            _register_subagent({
                "subagent_id": "sa-b",
                "parent_id": "sa-a",
                "depth": 1,
                "goal": "Research B",
                "model": "claude-3",
                "started_at": time.time(),
                "status": "running",
                "tool_count": 5,
            })
            path = checkpoint_active_subagents()
            self.assertIsNotNone(path)
            with open(path) as f:
                payload = json.load(f)
            self.assertEqual(len(payload["subagents"]), 2)
            ids = {sa["subagent_id"] for sa in payload["subagents"]}
            self.assertEqual(ids, {"sa-a", "sa-b"})


class TestOrphanAdoption(unittest.TestCase):
    """adopt_orphaned_subagents consuming checkpoints."""

    def setUp(self):
        """Clear pending orphans before each test."""
        try:
            from tools.delegate_tool import _pending_orphans as _po
            _po.clear()
        except (ImportError, AttributeError):
            pass

    def _create_checkpoint(self, directory, subagents, pid=12345, version=2):
        """Helper: write a valid checkpoint under *directory*."""
        state_dir = os.path.join(directory, "state")
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, _CHECKPOINT_FILE_NAME)
        payload = {
            "pid": pid,
            "parent_pid": pid - 1,
            "created_at": time.time(),
            "version": version,
            "subagents": subagents,
        }
        with open(path, "w") as f:
            json.dump(payload, f)
        return path

    @patch("hermes_constants.get_hermes_home")
    def test_no_checkpoint_returns_zero(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 0)
            # No pending orphans
            self.assertEqual(get_pending_orphans(), [])

    @patch("hermes_constants.get_hermes_home")
    def test_adopts_and_removes_checkpoint(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            path = self._create_checkpoint(td, [
                {
                    "subagent_id": "orphan-1",
                    "parent_id": None,
                    "depth": 0,
                    "goal": "Lost research task",
                    "model": "gpt-4",
                    "started_at": time.time(),
                    "status": "running",
                    "tool_count": 3,
                    "saved_messages": [
                        {"role": "user", "content": "Research topic X"},
                        {"role": "assistant", "content": "Found results"},
                    ],
                    "saved_message_count": 2,
                }
            ])
            self.assertTrue(os.path.isfile(path))
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 1)
            # Checkpoint must be removed after adoption
            self.assertFalse(os.path.isfile(path))
            # Pending orphans should be populated
            orphans = get_pending_orphans()
            self.assertEqual(len(orphans), 1)
            self.assertEqual(orphans[0]["subagent_id"], "orphan-1")
            self.assertIn("saved_messages", orphans[0])

    @patch("hermes_constants.get_hermes_home")
    def test_adopts_multiple_orphans(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            self._create_checkpoint(td, [
                {
                    "subagent_id": "orphan-1",
                    "parent_id": None,
                    "depth": 0,
                    "goal": "Task 1",
                    "model": "gpt-4",
                    "started_at": time.time() - 3600,
                    "status": "running",
                    "tool_count": 5,
                },
                {
                    "subagent_id": "orphan-2",
                    "parent_id": "orphan-1",
                    "depth": 1,
                    "goal": "Task 2",
                    "model": "claude-3",
                    "started_at": time.time() - 1800,
                    "status": "running",
                    "tool_count": 2,
                },
                {
                    "subagent_id": "orphan-3",
                    "parent_id": None,
                    "depth": 0,
                    "goal": "Task 3",
                    "model": "gemini-pro",
                    "started_at": time.time() - 600,
                    "status": "running",
                    "tool_count": 0,
                },
            ])
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 3)
            orphans = get_pending_orphans()
            self.assertEqual(len(orphans), 3)

    @patch("hermes_constants.get_hermes_home")
    def test_corrupt_checkpoint_does_not_crash(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            # Write invalid JSON
            state_dir = os.path.join(td, "state")
            os.makedirs(state_dir, exist_ok=True)
            bad_path = os.path.join(state_dir, _CHECKPOINT_FILE_NAME)
            with open(bad_path, "w") as f:
                f.write("{invalid json!!!}")
            self.assertTrue(os.path.isfile(bad_path))
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 0)
            # Should have cleaned up the corrupt file
            self.assertFalse(os.path.isfile(bad_path))
            self.assertEqual(get_pending_orphans(), [])

    @patch("hermes_constants.get_hermes_home")
    def test_empty_subagents_list_cleans_up(self, mock_home):
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            self._create_checkpoint(td, [])
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 0)
            self.assertEqual(get_pending_orphans(), [])

    @patch("hermes_constants.get_hermes_home")
    def test_minimal_goal_does_not_crash(self, mock_home):
        """Edge case: subagent with minimal fields (just goal)."""
        with tempfile.TemporaryDirectory() as td:
            mock_home.return_value = Path(td)
            path = self._create_checkpoint(td, [
                {"goal": "Short task"},
            ])
            count = adopt_orphaned_subagents()
            self.assertEqual(count, 1)
            self.assertFalse(os.path.isfile(path))
            orphans = get_pending_orphans()
            self.assertEqual(len(orphans), 1)
            self.assertEqual(orphans[0]["goal"], "Short task")


class TestOrphanRecreation(unittest.TestCase):
    """recreate_pending_subagents auto-re-delegation."""

    def setUp(self):
        try:
            from tools.delegate_tool import _pending_orphans as _po
            _po.clear()
        except (ImportError, AttributeError):
            pass

    @patch("tools.delegate_tool._pending_orphans", [])
    def test_no_orphans_returns_zero(self):
        count = recreate_pending_subagents(MagicMock())
        self.assertEqual(count, 0)

    @patch("tools.delegate_tool.delegate_task")
    def test_recreates_with_goal(self, mock_delegate):
        # Manually set pending orphans
        import tools.delegate_tool as dt
        dt._pending_orphans = [
            {
                "subagent_id": "orphan-1",
                "goal": "Research topic X",
                "model": "gpt-4",
                "depth": 0,
                "started_at": time.time(),
            }
        ]
        parent = MagicMock()
        count = recreate_pending_subagents(parent)
        self.assertEqual(count, 1)
        mock_delegate.assert_called_once()
        args, kwargs = mock_delegate.call_args
        self.assertEqual(kwargs.get("goal"), "Research topic X")

    @patch("tools.delegate_tool.delegate_task")
    def test_recreates_with_saved_messages_as_context(self, mock_delegate):
        import tools.delegate_tool as dt
        dt._pending_orphans = [
            {
                "subagent_id": "orphan-2",
                "goal": "Continue research",
                "model": "claude-3",
                "depth": 0,
                "started_at": time.time(),
                "saved_messages": [
                    {"role": "user", "content": "Find data on X"},
                    {"role": "assistant", "content": "Found partial data"},
                    {"role": "tool", "content": "Search result: some info"},
                ],
                "saved_message_count": 3,
            }
        ]
        parent = MagicMock()
        count = recreate_pending_subagents(parent)
        self.assertEqual(count, 1)
        mock_delegate.assert_called_once()
        args, kwargs = mock_delegate.call_args
        self.assertEqual(kwargs.get("goal"), "Continue research")
        # Context should include saved messages
        ctx = kwargs.get("context", "")
        self.assertIn("CONTINUATION", ctx)
        self.assertIn("Find data on X", ctx)
        self.assertIn("Found partial data", ctx)
        self.assertIn("Do NOT re-do work", ctx)

    @patch("tools.delegate_tool.delegate_task")
    def test_skips_orphan_without_goal(self, mock_delegate):
        import tools.delegate_tool as dt
        dt._pending_orphans = [
            {
                "subagent_id": "orphan-empty",
                "goal": "",
                "depth": 0,
            }
        ]
        parent = MagicMock()
        count = recreate_pending_subagents(parent)
        self.assertEqual(count, 0)
        mock_delegate.assert_not_called()


class TestRemoveCheckpointSafe(unittest.TestCase):
    """_remove_checkpoint_safe should never raise."""

    def test_nonexistent_file(self):
        _remove_checkpoint_safe("/nonexistent/path.json")

    def test_valid_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp = f.name
        self.assertTrue(os.path.isfile(tmp))
        _remove_checkpoint_safe(tmp)
        self.assertFalse(os.path.isfile(tmp))


if __name__ == "__main__":
    unittest.main()
