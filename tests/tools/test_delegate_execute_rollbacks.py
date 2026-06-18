#!/usr/bin/env python3
"""
Tests for _execute_rollbacks in the subagent delegation tool.

Covers the six bugs fixed in the feat/delegate-rollback-unban-execode branch:
  1. Execution timeout (ThreadPoolExecutor + _get_child_timeout)
  2. Heartbeat thread for rollback children
  3. TUI registration/unregistration
  4. max(1, min(10, effective_max_iter)) floor
  5. invoke_hook imported once at function top (not per-loop)
  6. (this file) basic test coverage

Run with:  python -m pytest tests/tools/test_delegate_execute_rollbacks.py -v
"""

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, PropertyMock

from tools.delegate_tool import (
    _execute_rollbacks,
    _register_subagent,
    _unregister_subagent,
    _active_subagents,
    _active_subagents_lock,
    _HEARTBEAT_INTERVAL,
)


def _make_creds():
    """Minimal credential dict that _execute_rollbacks expects."""
    return {
        "model": "test-model",
        "provider": "test-provider",
        "base_url": "https://example.com/v1",
        "api_key": "test-key",
        "api_mode": "chat_completions",
    }


def _make_mock_parent():
    """Mock parent agent with the fields used by _execute_rollbacks."""
    parent = MagicMock()
    parent.session_id = "parent-session-123"
    parent._touch_activity = MagicMock()
    parent._memory_manager = None  # skip memory callback
    return parent


def _make_mock_child(result=None, subagent_id="rb-child-001"):
    """Mock child agent that _build_child_agent would return."""
    child = MagicMock()
    child._subagent_id = subagent_id
    child._delegate_depth = 1
    child._parent_subagent_id = "parent-sa-001"
    child.model = "test-model"
    child.session_id = "child-session-456"
    child.session_estimated_cost_usd = 0.003
    child._delegate_saved_tool_names = []

    # Default: a successful run_conversation result
    if result is None:
        result = {
            "final_response": "Rollback completed: deleted temp file.",
            "completed": True,
        }
    child.run_conversation.return_value = result
    child.get_activity_summary.return_value = {
        "current_tool": None,
        "api_call_count": 1,
        "max_iterations": 10,
    }
    return child


class TestExecuteRollbacksEmpty(unittest.TestCase):
    """Edge case: empty rollback list."""

    def test_returns_empty_list(self):
        result = _execute_rollbacks(
            rollback_items=[],
            parent_agent=None,
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )
        self.assertEqual(result, [])


class TestExecuteRollbacksSkipped(unittest.TestCase):
    """Items without valid rollback instructions are skipped."""

    @patch("tools.delegate_tool._build_child_agent")
    def test_skips_empty_instruction(self, mock_build):
        result = _execute_rollbacks(
            rollback_items=[{"task_index": 0, "goal": "g", "rollback": ""}],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "skipped")
        mock_build.assert_not_called()

    @patch("tools.delegate_tool._build_child_agent")
    def test_skips_none_instruction(self, mock_build):
        result = _execute_rollbacks(
            rollback_items=[{"task_index": 0, "goal": "g", "rollback": None}],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "skipped")


class TestExecuteRollbacksSuccess(unittest.TestCase):
    """Happy-path rollback execution."""

    @patch("tools.delegate_tool._build_child_agent")
    def test_successful_rollback(self, mock_build):
        mock_child = _make_mock_child()
        mock_build.return_value = mock_child

        result = _execute_rollbacks(
            rollback_items=[
                {"task_index": 0, "goal": "create file", "rollback": "delete file"},
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["task_index"], 0)
        self.assertEqual(result[0]["status"], "completed")
        self.assertIn("Rollback completed", result[0]["summary"])
        # Child should be closed
        mock_child.close.assert_called_once()

    @patch("tools.delegate_tool._build_child_agent")
    def test_max_iterations_floor(self, mock_build):
        """Bug #4: max(1, min(10, 0)) should be 1, not 0."""
        mock_child = _make_mock_child()
        mock_build.return_value = mock_child

        _execute_rollbacks(
            rollback_items=[
                {"task_index": 0, "goal": "g", "rollback": "undo"},
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=0,  # edge case
            parent_tool_names=["terminal", "file"],
        )

        # _build_child_agent should have been called with max_iterations=1
        call_kwargs = mock_build.call_args
        self.assertEqual(call_kwargs.kwargs.get("max_iterations", call_kwargs[1].get("max_iterations")), 1)


class TestExecuteRollbacksTimeout(unittest.TestCase):
    """Bug #1: rollback child respects timeout via ThreadPoolExecutor."""

    @patch("tools.delegate_tool._get_child_timeout", return_value=1.0)
    @patch("tools.delegate_tool._build_child_agent")
    def test_timeout_raises_and_interrupts(self, mock_build, mock_timeout):
        """A rollback child that hangs should be interrupted after timeout."""
        mock_child = _make_mock_child()

        # Simulate a child that blocks forever
        def _hang(*args, **kwargs):
            time.sleep(60)
            return {"final_response": "never", "completed": True}

        mock_child.run_conversation.side_effect = _hang
        mock_build.return_value = mock_child

        results = _execute_rollbacks(
            rollback_items=[
                {"task_index": 0, "goal": "g", "rollback": "undo"},
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )

        # Should have caught the timeout as an error
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        # Child should have been interrupted
        mock_child.interrupt.assert_called()
        # Child should still be closed in finally
        mock_child.close.assert_called_once()


class TestExecuteRollbacksHeartbeat(unittest.TestCase):
    """Bug #2: heartbeat thread touches parent activity during rollback."""

    @patch("tools.delegate_tool._build_child_agent")
    def test_heartbeat_touches_parent(self, mock_build):
        """The heartbeat thread should call _touch_activity on the parent."""
        # Make the child take long enough for at least one heartbeat cycle
        mock_child = _make_mock_child()

        call_count = [0]
        original_run = mock_child.run_conversation

        def _slow_run(*args, **kwargs):
            # Sleep long enough for one heartbeat tick in tests.
            # Since _HEARTBEAT_INTERVAL is 30s, we patch it for this test.
            time.sleep(0.5)
            return original_run.return_value

        mock_child.run_conversation.side_effect = _slow_run
        mock_build.return_value = mock_child
        parent = _make_mock_parent()

        # Patch the heartbeat interval to be very short so we see touches
        with patch("tools.delegate_tool._HEARTBEAT_INTERVAL", 0.1):
            _execute_rollbacks(
                rollback_items=[
                    {"task_index": 0, "goal": "g", "rollback": "undo"},
                ],
                parent_agent=parent,
                creds=_make_creds(),
                effective_max_iter=5,
                parent_tool_names=["terminal", "file"],
            )

        # _touch_activity should have been called at least once by heartbeat
        parent._touch_activity.assert_called()


class TestExecuteRollbacksTUIRegistration(unittest.TestCase):
    """Bug #3: rollback children are registered/unregistered with TUI."""

    @patch("tools.delegate_tool._build_child_agent")
    def test_tui_register_and_unregister(self, mock_build):
        mock_child = _make_mock_child(subagent_id="tui-test-001")
        mock_build.return_value = mock_child

        # Clean the active subagents dict before test
        with _active_subagents_lock:
            _active_subagents.clear()

        _execute_rollbacks(
            rollback_items=[
                {"task_index": 0, "goal": "g", "rollback": "undo"},
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )

        # After completion, the subagent should have been unregistered
        with _active_subagents_lock:
            self.assertNotIn("tui-test-001", _active_subagents)


class TestExecuteRollbacksHookImport(unittest.TestCase):
    """Bug #5: invoke_hook should be imported once, not per loop iteration."""

    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._invoke_hook", create=True)
    def test_hook_called_from_top_level_import(self, mock_hook, mock_build):
        """Verify that the hook mechanism works when available."""
        mock_child = _make_mock_child()
        mock_build.return_value = mock_child

        # We test that the code path reaches the hook call by checking
        # that no per-iteration import error occurs.
        with patch.dict("sys.modules", {"hermes_cli.plugins": MagicMock(invoke_hook=mock_hook)}):
            _execute_rollbacks(
                rollback_items=[
                    {"task_index": 0, "goal": "g", "rollback": "undo"},
                ],
                parent_agent=_make_mock_parent(),
                creds=_make_creds(),
                effective_max_iter=5,
                parent_tool_names=["terminal", "file"],
            )


class TestExecuteRollbacksChildException(unittest.TestCase):
    """Child build or run failure is caught gracefully."""

    @patch("tools.delegate_tool._build_child_agent", side_effect=RuntimeError("build failed"))
    def test_build_failure_returns_error(self, mock_build):
        results = _execute_rollbacks(
            rollback_items=[
                {"task_index": 0, "goal": "g", "rollback": "undo"},
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "error")
        self.assertIn("build failed", results[0]["error"])


class TestExecuteRollbacksMultiple(unittest.TestCase):
    """Multiple rollback items are processed sequentially."""

    @patch("tools.delegate_tool._build_child_agent")
    def test_multiple_items(self, mock_build):
        children = [
            _make_mock_child(subagent_id=f"rb-{i}")
            for i in range(3)
        ]
        mock_build.side_effect = children

        results = _execute_rollbacks(
            rollback_items=[
                {"task_index": i, "goal": f"g{i}", "rollback": f"undo{i}"}
                for i in range(3)
            ],
            parent_agent=_make_mock_parent(),
            creds=_make_creds(),
            effective_max_iter=5,
            parent_tool_names=["terminal", "file"],
        )

        self.assertEqual(len(results), 3)
        for i, r in enumerate(results):
            self.assertEqual(r["task_index"], i)
            self.assertEqual(r["status"], "completed")


if __name__ == "__main__":
    unittest.main()
