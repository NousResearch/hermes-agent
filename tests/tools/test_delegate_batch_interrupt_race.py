#!/usr/bin/env python3
"""
Tests for the batch delegation interrupt race condition fix.

Reproduces the scenario where a parent agent is interrupted while children
are completing — verifies that completed children's results are preserved
rather than being discarded as "interrupted".

Race condition: in the old code, when the parent interrupt fires, the loop
iterates over `pending` futures checking `f.done()` point-in-time. A future
that completes between the last `_cf_wait` return and the interrupt check
(or during the fabrication loop itself) could have its result discarded.

The fix uses `_cf_wait(pending, timeout=2.0, return_when=ALL_COMPLETED)` to
give children a grace window to deliver already-computed results.

Run with:  python -m pytest tests/tools/test_delegate_batch_interrupt_race.py -v
"""

import json
import threading
import time
import unittest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

from tools.delegate_tool import delegate_task


def _make_mock_parent(depth=0):
    """Create a mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent._interrupt_requested = False
    parent._interrupt_message = None
    parent._current_task_id = "test-parent-task"
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_status = "unknown"
    parent.session_cost_source = "none"
    return parent


def _make_mock_child(response="Task done", api_calls=3, delay=0.0):
    """Create a mock child agent that returns after optional delay."""
    child = MagicMock()
    child.model = "anthropic/claude-sonnet-4"
    child._delegate_depth = 1
    child._delegate_role = "leaf"
    child._subagent_id = f"subagent-{id(child)}"
    child._parent_subagent_id = None
    child.session_prompt_tokens = 100
    child.session_completion_tokens = 50
    child.session_reasoning_tokens = 0
    child.session_estimated_cost_usd = 0.01
    child.quiet_mode = True

    def _run_conversation(user_message, task_id=None, **kwargs):
        if delay > 0:
            time.sleep(delay)
        return {
            "final_response": response,
            "completed": True,
            "interrupted": False,
            "api_calls": api_calls,
            "messages": [],
        }

    child.run_conversation = _run_conversation
    child._delegate_saved_tool_names = ["read_file", "terminal"]
    child.get_activity_summary = lambda: {
        "api_call_count": api_calls,
        "max_iterations": 50,
        "current_tool": None,
    }
    return child


class TestBatchInterruptRaceCondition(unittest.TestCase):
    """Verify that completed children's results survive parent interrupt."""

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._get_child_timeout")
    def test_completed_child_result_preserved_on_interrupt(
        self, mock_timeout, mock_build, mock_creds, mock_config
    ):
        """A child that finishes before the grace window expires should have
        its result preserved, not fabricated as 'interrupted'.

        Scenario: 2 batch tasks. Child 0 completes quickly (0.1s).
        Child 1 is slow (10s). Parent interrupt fires after 0.5s.
        Expected: child 0 = completed, child 1 = interrupted.
        """
        mock_config.return_value = {
            "max_concurrent_children": 5,
            "child_timeout_seconds": 600,
            "max_iterations": 50,
            "max_spawn_depth": 2,
        }
        mock_creds.return_value = {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
        }
        mock_timeout.return_value = 600.0

        # Child 0: fast (completes in 0.1s)
        fast_child = _make_mock_child(response="Fast task done", delay=0.1)
        # Child 1: slow (takes 10s — will be interrupted)
        slow_child = _make_mock_child(response="Slow task done", delay=10.0)

        mock_build.side_effect = [fast_child, slow_child]

        parent = _make_mock_parent(depth=0)

        # Fire interrupt after 0.5s — fast child should be done by then
        def _fire_interrupt():
            time.sleep(0.5)
            parent._interrupt_requested = True

        interrupt_thread = threading.Thread(target=_fire_interrupt, daemon=True)
        interrupt_thread.start()

        result_json = delegate_task(
            tasks=[
                {"goal": "Fast task"},
                {"goal": "Slow task"},
            ],
            parent_agent=parent,
        )

        result = json.loads(result_json)
        results = result["results"]

        # Sort by task_index for deterministic assertions
        results.sort(key=lambda r: r["task_index"])

        # Child 0 (fast) should have its actual result preserved
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[0]["summary"], "Fast task done")

        # Child 1 (slow) should be marked interrupted
        self.assertEqual(results[1]["status"], "interrupted")

        interrupt_thread.join(timeout=2)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._get_child_timeout")
    def test_all_completed_before_interrupt_all_preserved(
        self, mock_timeout, mock_build, mock_creds, mock_config
    ):
        """If all children complete before the interrupt's grace window,
        all results should be preserved (none fabricated as interrupted).

        Scenario: 3 batch tasks, all complete in <0.2s.
        Parent interrupt fires at 0.3s. Grace window is 2s.
        Expected: all 3 = completed.
        """
        mock_config.return_value = {
            "max_concurrent_children": 5,
            "child_timeout_seconds": 600,
            "max_iterations": 50,
            "max_spawn_depth": 2,
        }
        mock_creds.return_value = {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
        }
        mock_timeout.return_value = 600.0

        children = [
            _make_mock_child(response=f"Task {i} done", delay=0.1)
            for i in range(3)
        ]
        mock_build.side_effect = children

        parent = _make_mock_parent(depth=0)

        def _fire_interrupt():
            time.sleep(0.3)
            parent._interrupt_requested = True

        interrupt_thread = threading.Thread(target=_fire_interrupt, daemon=True)
        interrupt_thread.start()

        result_json = delegate_task(
            tasks=[
                {"goal": "Task 0"},
                {"goal": "Task 1"},
                {"goal": "Task 2"},
            ],
            parent_agent=parent,
        )

        result = json.loads(result_json)
        results = result["results"]
        results.sort(key=lambda r: r["task_index"])

        # All should be completed — none should be fabricated as interrupted
        for i, entry in enumerate(results):
            self.assertEqual(
                entry["status"], "completed",
                f"Task {i} should be 'completed' but got '{entry['status']}'"
            )
            self.assertEqual(entry["summary"], f"Task {i} done")

        interrupt_thread.join(timeout=2)

    @patch("tools.delegate_tool._load_config")
    @patch("tools.delegate_tool._resolve_delegation_credentials")
    @patch("tools.delegate_tool._build_child_agent")
    @patch("tools.delegate_tool._get_child_timeout")
    def test_child_completing_during_grace_window_is_collected(
        self, mock_timeout, mock_build, mock_creds, mock_config
    ):
        """A child that finishes DURING the 2s grace window (after interrupt
        fires but before grace expires) should be collected, not interrupted.

        Scenario: Child 0 completes at 0.1s. Child 1 completes at 1.5s.
        Parent interrupt fires at 0.3s. Grace window is 2s.
        Expected: both = completed (child 1 finishes within grace).
        """
        mock_config.return_value = {
            "max_concurrent_children": 5,
            "child_timeout_seconds": 600,
            "max_iterations": 50,
            "max_spawn_depth": 2,
        }
        mock_creds.return_value = {
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "***",
            "api_mode": "chat_completions",
            "command": None,
            "args": None,
        }
        mock_timeout.return_value = 600.0

        # Child 0: completes at 0.1s
        child_0 = _make_mock_child(response="Child 0 done", delay=0.1)
        # Child 1: completes at 1.5s — within the 2s grace window
        child_1 = _make_mock_child(response="Child 1 done", delay=1.5)

        mock_build.side_effect = [child_0, child_1]

        parent = _make_mock_parent(depth=0)

        def _fire_interrupt():
            time.sleep(0.3)
            parent._interrupt_requested = True

        interrupt_thread = threading.Thread(target=_fire_interrupt, daemon=True)
        interrupt_thread.start()

        result_json = delegate_task(
            tasks=[
                {"goal": "Child 0 task"},
                {"goal": "Child 1 task"},
            ],
            parent_agent=parent,
        )

        result = json.loads(result_json)
        results = result["results"]
        results.sort(key=lambda r: r["task_index"])

        # Both should be completed — child 1 finishes within grace window
        self.assertEqual(results[0]["status"], "completed")
        self.assertEqual(results[0]["summary"], "Child 0 done")
        self.assertEqual(results[1]["status"], "completed")
        self.assertEqual(results[1]["summary"], "Child 1 done")

        interrupt_thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
