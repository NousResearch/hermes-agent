#!/usr/bin/env python3
"""
Tests for the async delegation tools.

Uses mock parent/child agents to test the delegation logic without
requiring API keys or real LLM calls.

Run with:
    .venv-test/bin/python -m pytest tests/tools/test_async_delegate.py -v --override-ini='addopts='
"""

import json
import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from tools.async_delegate_tool import (
    AsyncTask,
    _get_task_registry,
    delegate_task_async,
    check_task,
    collect_task,
    steer_task,
    cancel_task,
    list_tasks,
)


# =============================================================================
# Test helpers
# =============================================================================

class MockParentAgent:
    """A minimal parent agent with no attributes — used to test attribute creation."""
    pass


class MockParentForDelegation:
    """Plain parent agent for delegate_task_async tests.

    Uses a plain class (not MagicMock) so that hasattr checks work correctly
    and _async_tasks is not auto-created by MagicMock magic.
    """
    def __init__(self, depth=0):
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "parent-key"
        self.provider = "openrouter"
        self.api_mode = "chat_completions"
        self.model = "anthropic/claude-sonnet-4"
        self.platform = "cli"
        self.providers_allowed = None
        self.providers_ignored = None
        self.providers_order = None
        self.provider_sort = None
        self._session_db = None
        self._delegate_depth = depth
        self._active_children = []
        self._active_children_lock = threading.Lock()


def make_mock_parent(depth=0):
    """Create a mock parent agent for delegate_task_async tests."""
    return MockParentForDelegation(depth=depth)


def make_mock_child():
    """Create a mock child agent with required async attributes."""
    child = MagicMock()
    child._steering_injection = queue.Queue()
    child.quiet_mode = True
    child._delegate_saved_tool_names = []
    return child


def _make_creds():
    """Return a minimal credentials dict."""
    return {
        "model": "anthropic/claude-sonnet-4",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "api_mode": "chat_completions",
    }


# Context manager that patches all delegate_task_async internals
class _AsyncDelegatePatches:
    """Patches the tools.delegate_tool imports used inside delegate_task_async."""

    def __init__(self, mock_child, creds=None):
        self._mock_child = mock_child
        self._creds = creds or _make_creds()
        self._patches = []

    def __enter__(self):
        import model_tools
        if not hasattr(model_tools, '_last_resolved_tool_names'):
            model_tools._last_resolved_tool_names = []

        self._patches = [
            patch('tools.delegate_tool._load_config', return_value={}),
            patch('tools.delegate_tool._resolve_delegation_credentials',
                  return_value=self._creds),
            patch('tools.delegate_tool._build_child_agent',
                  return_value=self._mock_child),
            patch('tools.delegate_tool._load_skill_for_subagent', return_value=None),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.stop()


# =============================================================================
# AsyncTask dataclass tests
# =============================================================================

class TestAsyncTaskDataclass(unittest.TestCase):
    """Tests for the AsyncTask dataclass elapsed property."""

    def test_elapsed_running_is_positive(self):
        """elapsed returns positive time since started_at when task is running."""
        task = AsyncTask(task_id="t1", goal="test goal")
        time.sleep(0.05)
        elapsed = task.elapsed
        self.assertGreater(elapsed, 0.0)
        self.assertIsNone(task.completed_at)

    def test_elapsed_done_is_fixed(self):
        """elapsed returns completed_at - started_at when completed_at is set."""
        task = AsyncTask(task_id="t2", goal="test goal")
        task.started_at = 1000.0
        task.completed_at = 1005.0
        self.assertAlmostEqual(task.elapsed, 5.0, places=5)

    def test_elapsed_increases_while_running(self):
        """elapsed should increase over time while the task is running."""
        task = AsyncTask(task_id="t3", goal="test goal")
        t1 = task.elapsed
        time.sleep(0.05)
        t2 = task.elapsed
        self.assertGreater(t2, t1)

    def test_elapsed_does_not_change_after_completion(self):
        """elapsed should be fixed (not drift) once completed_at is set."""
        task = AsyncTask(task_id="t4", goal="test goal")
        task.started_at = 1000.0
        task.completed_at = 1003.5
        e1 = task.elapsed
        time.sleep(0.05)
        e2 = task.elapsed
        self.assertAlmostEqual(e1, e2, places=5)
        self.assertAlmostEqual(e1, 3.5, places=5)

    def test_default_status_is_running(self):
        """Default status should be 'running'."""
        task = AsyncTask(task_id="t5", goal="test")
        self.assertEqual(task.status, "running")

    def test_done_event_initially_not_set(self):
        """done_event should not be set initially."""
        task = AsyncTask(task_id="t6", goal="test")
        self.assertFalse(task.done_event.is_set())

    def test_completed_at_is_none_by_default(self):
        """completed_at should be None initially (task is running)."""
        task = AsyncTask(task_id="t7", goal="test")
        self.assertIsNone(task.completed_at)


# =============================================================================
# _get_task_registry tests
# =============================================================================

class TestGetTaskRegistry(unittest.TestCase):
    """Tests for _get_task_registry."""

    def test_creates_attrs_if_absent(self):
        """Creates _async_tasks and _async_tasks_lock on parent if absent."""
        parent = MockParentAgent()
        self.assertFalse(hasattr(parent, '_async_tasks'))
        self.assertFalse(hasattr(parent, '_async_tasks_lock'))

        tasks_dict, lock = _get_task_registry(parent)

        self.assertTrue(hasattr(parent, '_async_tasks'))
        self.assertTrue(hasattr(parent, '_async_tasks_lock'))
        self.assertIsInstance(tasks_dict, dict)
        self.assertIsInstance(lock, type(threading.Lock()))

    def test_returns_same_registry_on_second_call(self):
        """Returns existing registry on second call (same dict/lock identity)."""
        parent = MockParentAgent()
        tasks1, lock1 = _get_task_registry(parent)
        tasks2, lock2 = _get_task_registry(parent)
        self.assertIs(tasks1, tasks2)
        self.assertIs(lock1, lock2)

    def test_does_not_reset_existing_tasks(self):
        """Existing tasks in registry are preserved on second call."""
        parent = MockParentAgent()
        tasks_dict, _ = _get_task_registry(parent)
        fake_task = AsyncTask(task_id="existing", goal="x")
        tasks_dict["existing"] = fake_task

        tasks2, _ = _get_task_registry(parent)
        self.assertIn("existing", tasks2)
        self.assertIs(tasks2["existing"], fake_task)

    def test_initial_tasks_dict_is_empty(self):
        """Newly created tasks dict should be empty."""
        parent = MockParentAgent()
        tasks_dict, _ = _get_task_registry(parent)
        self.assertEqual(len(tasks_dict), 0)


# =============================================================================
# delegate_task_async tests
# =============================================================================

class TestDelegateTaskAsync(unittest.TestCase):
    """Tests for delegate_task_async."""

    def test_no_parent_agent_returns_error(self):
        """Returns error JSON when no parent_agent provided."""
        result = json.loads(delegate_task_async(goal="test"))
        self.assertIn("error", result)
        self.assertIn("parent agent", result["error"])

    def test_empty_goal_returns_error(self):
        """Returns error JSON when goal is empty or whitespace."""
        parent = make_mock_parent()
        result = json.loads(delegate_task_async(goal="   ", parent_agent=parent))
        self.assertIn("error", result)

    def test_blank_goal_returns_error(self):
        """Returns error JSON for blank goal."""
        parent = make_mock_parent()
        result = json.loads(delegate_task_async(goal="", parent_agent=parent))
        self.assertIn("error", result)

    def test_depth_limit_returns_error(self):
        """Returns error JSON when delegation depth limit is reached."""
        parent = make_mock_parent(depth=2)  # MAX_DEPTH is 2
        result = json.loads(delegate_task_async(goal="test", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("depth", result["error"].lower())

    def test_returns_task_id_and_running_status(self):
        """Returns dict with task_id and status=running immediately."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="Do a thing", parent_agent=parent))

        self.assertIn("task_id", result)
        self.assertEqual(result["status"], "running")
        self.assertTrue(result["task_id"].startswith("async_"))

    def test_task_appears_in_parent_registry(self):
        """Task is stored in parent._async_tasks immediately after call."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="Do a thing", parent_agent=parent))
            task_id = result["task_id"]

        self.assertIn(task_id, parent._async_tasks)

    def test_thread_is_alive_after_spawn(self):
        """A background thread is started for the task."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        hold_event = threading.Event()

        def slow_run(user_message):
            hold_event.wait(timeout=3.0)
            return {"completed": True, "final_response": "done"}

        mock_child.run_conversation.side_effect = slow_run

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="background task", parent_agent=parent))
            task_id = result["task_id"]

        task = parent._async_tasks[task_id]
        self.assertTrue(task.thread.is_alive())

        # Clean up — unblock the thread
        hold_event.set()
        task.done_event.wait(timeout=3.0)

    def test_output_capture_via_print_fn(self):
        """Child's _print_fn (set by delegate_task_async) writes to task.output_lines."""
        parent = make_mock_parent()
        mock_child = make_mock_child()

        def run_and_print(user_message):
            # Simulate child calling its print function
            mock_child._print_fn("line one")
            mock_child._print_fn("line two")
            return {"completed": True, "final_response": "done"}

        mock_child.run_conversation.side_effect = run_and_print

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="capture output", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=3.0)

        self.assertIn("line one", task.output_lines)
        self.assertIn("line two", task.output_lines)

    def test_completion_sets_status_and_event(self):
        """On successful run: task.status == 'completed' and done_event is set."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="complete task", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=3.0)

        self.assertEqual(task.status, "completed")
        self.assertTrue(task.done_event.is_set())
        self.assertIsNotNone(task.completed_at)

    def test_run_returns_false_sets_failed_status(self):
        """When run_conversation returns completed=False: task.status == 'failed'."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": False, "final_response": ""
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="incomplete task", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=3.0)

        self.assertEqual(task.status, "failed")

    def test_exception_in_thread_sets_failed_with_error(self):
        """On exception in thread: task.status == 'failed' and task.error is set."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.side_effect = RuntimeError("something broke")

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="failing task", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=3.0)

        self.assertEqual(task.status, "failed")
        self.assertIsNotNone(task.error)
        self.assertIn("something broke", task.error)

    def test_agent_construction_fails_returns_error_json(self):
        """If _build_child_agent raises, returns error JSON not an exception."""
        parent = make_mock_parent()

        with patch('tools.delegate_tool._load_config', return_value={}), \
             patch('tools.delegate_tool._resolve_delegation_credentials',
                   return_value=_make_creds()), \
             patch('tools.delegate_tool._build_child_agent',
                   side_effect=RuntimeError("build failed")), \
             patch('tools.delegate_tool._load_skill_for_subagent', return_value=None):
            result = json.loads(delegate_task_async(goal="test failure", parent_agent=parent))

        self.assertIn("error", result)
        self.assertIn("build failed", result["error"])

    def test_creds_value_error_returns_error_json(self):
        """If credential resolution raises ValueError, returns error JSON."""
        parent = make_mock_parent()

        with patch('tools.delegate_tool._load_config', return_value={}), \
             patch('tools.delegate_tool._resolve_delegation_credentials',
                   side_effect=ValueError("no api key")), \
             patch('tools.delegate_tool._load_skill_for_subagent', return_value=None):
            result = json.loads(delegate_task_async(goal="creds fail", parent_agent=parent))

        self.assertIn("error", result)
        self.assertIn("no api key", result["error"])

    def test_done_event_set_even_after_exception(self):
        """done_event is always set in finally block, even on exception."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.side_effect = RuntimeError("crash")

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="crash task", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=3.0)

        self.assertTrue(task.done_event.is_set())
        self.assertIsNotNone(task.completed_at)


# =============================================================================
# check_task tests
# =============================================================================

class TestCheckTask(unittest.TestCase):
    """Tests for check_task."""

    def _make_parent_with_task(self, task_id="t-001", **kwargs):
        """Helper: create parent with a pre-registered task."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        defaults = {"task_id": task_id, "goal": "test goal", "status": "running"}
        defaults.update(kwargs)
        task = AsyncTask(**defaults)
        tasks[task_id] = task
        return parent, task

    def test_no_parent_agent_returns_error(self):
        """Returns error JSON when no parent_agent provided."""
        result = json.loads(check_task(task_id="t-001"))
        self.assertIn("error", result)

    def test_unknown_task_id_returns_error(self):
        """Returns error JSON for unknown task_id."""
        parent = MockParentAgent()
        _get_task_registry(parent)
        result = json.loads(check_task(task_id="nonexistent", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("nonexistent", result["error"])

    def test_returns_required_fields(self):
        """Returns task_id, status, elapsed, output_preview, output_lines."""
        parent, task = self._make_parent_with_task()
        task.output_lines = ["line 1", "line 2", "line 3"]

        result = json.loads(check_task(task_id="t-001", parent_agent=parent))

        self.assertEqual(result["task_id"], "t-001")
        self.assertEqual(result["status"], "running")
        self.assertIn("elapsed", result)
        self.assertIn("output_preview", result)
        self.assertEqual(result["output_lines"], 3)

    def test_output_preview_contains_last_10_lines(self):
        """output_preview shows last 10 lines of output_lines."""
        parent, task = self._make_parent_with_task()
        task.output_lines = [f"line {i}" for i in range(20)]

        result = json.loads(check_task(task_id="t-001", parent_agent=parent))

        preview = result["output_preview"]
        # Last 10 lines are line 10..19
        self.assertIn("line 19", preview)
        self.assertIn("line 10", preview)
        # line 9 should not appear (it's outside the last 10)
        lines_in_preview = set(preview.split("\n"))
        self.assertNotIn("line 9", lines_in_preview)

    def test_output_lines_count_reflects_actual_list_length(self):
        """output_lines field reflects the length of the output_lines list."""
        parent, task = self._make_parent_with_task()
        task.output_lines = ["a", "b", "c", "d", "e"]

        result = json.loads(check_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["output_lines"], 5)

    def test_completed_task_shows_correct_status(self):
        """check_task works on completed tasks and shows status."""
        parent, task = self._make_parent_with_task(status="completed")
        task.completed_at = task.started_at + 3.0

        result = json.loads(check_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["status"], "completed")

    def test_elapsed_is_numeric(self):
        """elapsed field is a number (float or int)."""
        parent, task = self._make_parent_with_task()
        result = json.loads(check_task(task_id="t-001", parent_agent=parent))
        self.assertIsInstance(result["elapsed"], (int, float))

    def test_empty_output_preview(self):
        """output_preview is empty string when no output_lines."""
        parent, task = self._make_parent_with_task()
        # output_lines is empty by default
        result = json.loads(check_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["output_preview"], "")
        self.assertEqual(result["output_lines"], 0)


# =============================================================================
# collect_task tests
# =============================================================================

class TestCollectTask(unittest.TestCase):
    """Tests for collect_task."""

    def _make_parent_with_task(self, task_id="t-001", **kwargs):
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        defaults = {"task_id": task_id, "goal": "test goal", "status": "running"}
        defaults.update(kwargs)
        task = AsyncTask(**defaults)
        tasks[task_id] = task
        return parent, task

    def test_no_parent_agent_returns_error(self):
        """Returns error JSON when no parent_agent provided."""
        result = json.loads(collect_task(task_id="t-001"))
        self.assertIn("error", result)

    def test_unknown_task_id_returns_error(self):
        """Returns error JSON for unknown task_id."""
        parent = MockParentAgent()
        _get_task_registry(parent)
        result = json.loads(collect_task(task_id="unknown", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("unknown", result["error"])

    def test_blocks_until_done_event_fires(self):
        """Blocks until done_event fires then returns full result."""
        parent, task = self._make_parent_with_task()
        task.output_lines = ["output line 1", "output line 2"]
        task.result = {"completed": True, "final_response": "task done"}

        def fire_event():
            time.sleep(0.1)
            task.status = "completed"
            task.completed_at = time.monotonic()
            task.done_event.set()

        t = threading.Thread(target=fire_event, daemon=True)
        t.start()

        result = json.loads(collect_task(task_id="t-001", parent_agent=parent))
        t.join(timeout=2.0)

        self.assertEqual(result["task_id"], "t-001")
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["timed_out"])
        self.assertIn("output line 1", result["output"])
        self.assertEqual(result["summary"], "task done")

    def test_timeout_returns_timed_out_true(self):
        """Returns timed_out=True when done_event is not set within timeout."""
        parent, task = self._make_parent_with_task()

        result = json.loads(collect_task(task_id="t-001", timeout=0, parent_agent=parent))

        self.assertTrue(result["timed_out"])
        self.assertEqual(result["status"], "running")

    def test_timeout_task_status_still_running(self):
        """Task status remains 'running' when collect_task times out."""
        parent, task = self._make_parent_with_task()
        result = json.loads(collect_task(task_id="t-001", timeout=0, parent_agent=parent))
        self.assertEqual(result["status"], "running")

    def test_returns_error_field_for_failed_task(self):
        """Includes error field in response when task has an error."""
        parent, task = self._make_parent_with_task(status="failed")
        task.error = "child crashed"
        task.completed_at = time.monotonic()
        task.done_event.set()

        result = json.loads(collect_task(task_id="t-001", parent_agent=parent))
        self.assertIn("error", result)
        self.assertEqual(result["error"], "child crashed")

    def test_output_joined_as_newline_text(self):
        """output field is newline-joined output_lines."""
        parent, task = self._make_parent_with_task()
        task.output_lines = ["alpha", "beta", "gamma"]
        task.done_event.set()
        task.status = "completed"
        task.completed_at = time.monotonic()
        task.result = {"completed": True, "final_response": ""}

        result = json.loads(collect_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["output"], "alpha\nbeta\ngamma")

    def test_output_lines_count_in_response(self):
        """output_lines count is returned in the response."""
        parent, task = self._make_parent_with_task()
        task.output_lines = ["x", "y"]
        task.done_event.set()
        task.status = "completed"
        task.completed_at = time.monotonic()
        task.result = {"completed": True, "final_response": ""}

        result = json.loads(collect_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["output_lines"], 2)

    def test_already_completed_returns_immediately(self):
        """If done_event is already set, collect_task returns without blocking."""
        parent, task = self._make_parent_with_task(status="completed")
        task.done_event.set()
        task.completed_at = task.started_at + 1.0
        task.result = {"completed": True, "final_response": "fast"}

        start = time.monotonic()
        result = json.loads(collect_task(task_id="t-001", parent_agent=parent))
        elapsed = time.monotonic() - start

        self.assertFalse(result["timed_out"])
        self.assertLess(elapsed, 1.0)  # Should not block


# =============================================================================
# steer_task tests
# =============================================================================

class TestSteerTask(unittest.TestCase):
    """Tests for steer_task."""

    def _make_parent_with_task(self, status="running"):
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        task = AsyncTask(task_id="t-001", goal="test", status=status)
        mock_child = MagicMock()
        mock_child._steering_injection = queue.Queue()
        task.child_agent = mock_child
        tasks["t-001"] = task
        return parent, task

    def test_no_parent_agent_returns_error(self):
        """Returns error JSON when no parent_agent provided."""
        result = json.loads(steer_task(task_id="t-001", message="steer"))
        self.assertIn("error", result)

    def test_unknown_task_id_returns_error(self):
        """Returns error JSON for unknown task_id."""
        parent = MockParentAgent()
        _get_task_registry(parent)
        result = json.loads(steer_task(task_id="unknown", message="steer", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("unknown", result["error"])

    def test_task_not_running_returns_error(self):
        """Returns error (ok=False) when task is not running."""
        parent, task = self._make_parent_with_task(status="completed")
        result = json.loads(steer_task(task_id="t-001", message="steer", parent_agent=parent))
        self.assertFalse(result["ok"])
        self.assertIn("not running", result["error"])

    def test_failed_task_returns_error(self):
        """Returns error (ok=False) when task has failed."""
        parent, task = self._make_parent_with_task(status="failed")
        result = json.loads(steer_task(task_id="t-001", message="steer", parent_agent=parent))
        self.assertFalse(result["ok"])

    def test_cancelled_task_returns_error(self):
        """Returns error (ok=False) when task is cancelled."""
        parent, task = self._make_parent_with_task(status="cancelled")
        result = json.loads(steer_task(task_id="t-001", message="steer", parent_agent=parent))
        self.assertFalse(result["ok"])

    def test_puts_message_in_steering_queue(self):
        """Puts message into child_agent._steering_injection when running."""
        parent, task = self._make_parent_with_task(status="running")

        result = json.loads(steer_task(task_id="t-001", message="please stop", parent_agent=parent))

        self.assertTrue(result["ok"])
        queued = task.child_agent._steering_injection.get_nowait()
        self.assertEqual(queued, "please stop")

    def test_steering_ok_response_contains_task_id(self):
        """Successful steer response mentions the task_id."""
        parent, task = self._make_parent_with_task(status="running")
        result = json.loads(steer_task(task_id="t-001", message="redirect", parent_agent=parent))
        self.assertTrue(result["ok"])
        self.assertIn("t-001", result["message"])

    def test_no_steering_injection_attr_returns_error(self):
        """Returns error (ok=False) if child has no _steering_injection."""
        parent, task = self._make_parent_with_task(status="running")
        # Remove the _steering_injection attribute
        task.child_agent._steering_injection = None
        result = json.loads(steer_task(task_id="t-001", message="steer", parent_agent=parent))
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_no_child_agent_returns_error(self):
        """Returns error (ok=False) if task has no child_agent."""
        parent, task = self._make_parent_with_task(status="running")
        task.child_agent = None
        result = json.loads(steer_task(task_id="t-001", message="steer", parent_agent=parent))
        self.assertFalse(result["ok"])


# =============================================================================
# cancel_task tests
# =============================================================================

class TestCancelTask(unittest.TestCase):
    """Tests for cancel_task."""

    def _make_parent_with_task(self, status="running"):
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        task = AsyncTask(task_id="t-001", goal="test", status=status)
        mock_child = MagicMock()
        mock_child.interrupt = MagicMock()
        task.child_agent = mock_child
        tasks["t-001"] = task
        return parent, task

    def test_no_parent_agent_returns_error(self):
        """Returns error JSON when no parent_agent provided."""
        result = json.loads(cancel_task(task_id="t-001"))
        self.assertIn("error", result)

    def test_unknown_task_id_returns_error(self):
        """Returns error JSON for unknown task_id."""
        parent = MockParentAgent()
        _get_task_registry(parent)
        result = json.loads(cancel_task(task_id="unknown", parent_agent=parent))
        self.assertIn("error", result)
        self.assertIn("unknown", result["error"])

    def test_already_completed_returns_error(self):
        """Returns error (ok=False) when task is not running."""
        parent, task = self._make_parent_with_task(status="completed")
        result = json.loads(cancel_task(task_id="t-001", parent_agent=parent))
        self.assertFalse(result["ok"])

    def test_calls_child_interrupt(self):
        """Calls child_agent.interrupt() when cancelling a running task."""
        parent, task = self._make_parent_with_task(status="running")
        result = json.loads(cancel_task(task_id="t-001", parent_agent=parent))
        self.assertTrue(result["ok"])
        task.child_agent.interrupt.assert_called_once()

    def test_sets_status_to_cancelled(self):
        """Sets task.status to 'cancelled' after cancellation."""
        parent, task = self._make_parent_with_task(status="running")
        cancel_task(task_id="t-001", parent_agent=parent)
        self.assertEqual(task.status, "cancelled")

    def test_sets_done_event(self):
        """Sets done_event after cancellation."""
        parent, task = self._make_parent_with_task(status="running")
        cancel_task(task_id="t-001", parent_agent=parent)
        self.assertTrue(task.done_event.is_set())

    def test_sets_completed_at_timestamp(self):
        """Sets completed_at timestamp after cancellation."""
        parent, task = self._make_parent_with_task(status="running")
        cancel_task(task_id="t-001", parent_agent=parent)
        self.assertIsNotNone(task.completed_at)

    def test_response_contains_cancelled_task_id(self):
        """Response includes the cancelled task_id in 'cancelled' field."""
        parent, task = self._make_parent_with_task(status="running")
        result = json.loads(cancel_task(task_id="t-001", parent_agent=parent))
        self.assertEqual(result["cancelled"], "t-001")

    def test_response_ok_true_on_success(self):
        """Response has ok=True when cancellation succeeds."""
        parent, task = self._make_parent_with_task(status="running")
        result = json.loads(cancel_task(task_id="t-001", parent_agent=parent))
        self.assertTrue(result["ok"])

    def test_no_child_agent_still_cancels(self):
        """Task is still cancelled even if child_agent is None."""
        parent, task = self._make_parent_with_task(status="running")
        task.child_agent = None
        result = json.loads(cancel_task(task_id="t-001", parent_agent=parent))
        self.assertTrue(result["ok"])
        self.assertEqual(task.status, "cancelled")


# =============================================================================
# list_tasks tests
# =============================================================================

class TestListTasks(unittest.TestCase):
    """Tests for list_tasks."""

    def test_no_parent_agent_returns_empty_tasks(self):
        """Returns empty tasks list when no parent_agent provided."""
        result = json.loads(list_tasks())
        self.assertIn("tasks", result)
        self.assertEqual(result["tasks"], [])

    def test_empty_registry_returns_empty_list(self):
        """Returns empty tasks list when registry is empty."""
        parent = MockParentAgent()
        _get_task_registry(parent)
        result = json.loads(list_tasks(parent_agent=parent))
        self.assertIn("tasks", result)
        self.assertEqual(len(result["tasks"]), 0)

    def test_single_task_has_required_fields(self):
        """Single task entry has task_id/goal/status/elapsed/output_lines fields."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        task = AsyncTask(task_id="t-abc", goal="do something important", status="running")
        task.output_lines = ["a", "b", "c"]
        tasks["t-abc"] = task

        result = json.loads(list_tasks(parent_agent=parent))
        self.assertEqual(len(result["tasks"]), 1)

        entry = result["tasks"][0]
        self.assertEqual(entry["task_id"], "t-abc")
        self.assertIn("do something", entry["goal"])
        self.assertEqual(entry["status"], "running")
        self.assertIn("elapsed", entry)
        self.assertEqual(entry["output_lines"], 3)

    def test_multiple_tasks_all_shown(self):
        """Returns all tasks when multiple are registered."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)

        for tid, status in [("t-1", "running"), ("t-2", "completed"), ("t-3", "failed")]:
            t = AsyncTask(task_id=tid, goal=f"goal for {tid}", status=status)
            tasks[tid] = t

        result = json.loads(list_tasks(parent_agent=parent))
        self.assertEqual(len(result["tasks"]), 3)

        task_ids = {t["task_id"] for t in result["tasks"]}
        self.assertEqual(task_ids, {"t-1", "t-2", "t-3"})

    def test_goal_truncated_to_60_chars(self):
        """Goal is truncated to 60 chars in list output."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        long_goal = "x" * 100
        t = AsyncTask(task_id="t-long", goal=long_goal, status="running")
        tasks["t-long"] = t

        result = json.loads(list_tasks(parent_agent=parent))
        entry = result["tasks"][0]
        self.assertLessEqual(len(entry["goal"]), 60)

    def test_multiple_tasks_statuses_correct(self):
        """Each task's status is correctly reported."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)

        statuses = {"t-run": "running", "t-done": "completed", "t-fail": "failed",
                    "t-cancel": "cancelled"}
        for tid, status in statuses.items():
            t = AsyncTask(task_id=tid, goal="test", status=status)
            tasks[tid] = t

        result = json.loads(list_tasks(parent_agent=parent))
        reported = {t["task_id"]: t["status"] for t in result["tasks"]}
        for tid, expected_status in statuses.items():
            self.assertEqual(reported[tid], expected_status)

    def test_elapsed_is_numeric_for_all_tasks(self):
        """All elapsed values in list are numeric."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)

        for i in range(3):
            t = AsyncTask(task_id=f"t-{i}", goal="test", status="running")
            tasks[f"t-{i}"] = t

        result = json.loads(list_tasks(parent_agent=parent))
        for entry in result["tasks"]:
            self.assertIsInstance(entry["elapsed"], (int, float))


# =============================================================================
# TestOutputTruncation — _MAX_OUTPUT_LINES and _MAX_LINE_CHARS
# =============================================================================

class TestOutputTruncation(unittest.TestCase):
    """Tests for output truncation limits (_MAX_OUTPUT_LINES=200, _MAX_LINE_CHARS=500)."""

    def test_import_constants(self):
        """Verify _MAX_OUTPUT_LINES and _MAX_LINE_CHARS are importable."""
        from tools.async_delegate_tool import _MAX_OUTPUT_LINES, _MAX_LINE_CHARS
        self.assertEqual(_MAX_OUTPUT_LINES, 200)
        self.assertEqual(_MAX_LINE_CHARS, 500)

    def test_oldest_lines_dropped_over_200(self):
        """When >200 output lines are produced, oldest are dropped."""
        from tools.async_delegate_tool import _MAX_OUTPUT_LINES, _MAX_LINE_CHARS, AsyncTask

        task = AsyncTask(task_id="trunc-test", goal="test")

        # Replicate _capture_print closure logic
        def _capture_print(msg):
            line = str(msg)[:_MAX_LINE_CHARS]
            task.output_lines.append(line)
            if len(task.output_lines) > _MAX_OUTPUT_LINES:
                del task.output_lines[0]

        # Produce 250 lines
        for i in range(250):
            _capture_print(f"line {i}")

        self.assertEqual(len(task.output_lines), 200)
        # Oldest lines should be dropped; first remaining is line 50
        self.assertEqual(task.output_lines[0], "line 50")
        # Most recent should be line 249
        self.assertEqual(task.output_lines[-1], "line 249")

    def test_individual_lines_truncated_at_500_chars(self):
        """Individual lines longer than _MAX_LINE_CHARS are truncated."""
        from tools.async_delegate_tool import _MAX_LINE_CHARS, AsyncTask

        task = AsyncTask(task_id="trunc-chars", goal="test")

        def _capture_print(msg):
            line = str(msg)[:_MAX_LINE_CHARS]
            task.output_lines.append(line)
            if len(task.output_lines) > 200:
                del task.output_lines[0]

        long_msg = "A" * 1000
        _capture_print(long_msg)

        self.assertEqual(len(task.output_lines), 1)
        self.assertEqual(len(task.output_lines[0]), 500)
        self.assertTrue(task.output_lines[0].startswith("A" * 500))

    def test_output_truncation_via_delegate_task_async(self):
        """Integration: delegate_task_async child producing >200 lines drops oldest."""
        from tools.async_delegate_tool import _MAX_OUTPUT_LINES, _MAX_LINE_CHARS

        parent = make_mock_parent()
        mock_child = make_mock_child()

        def run_and_spam(user_message):
            # Simulate child printing 250 lines via its _print_fn
            for i in range(250):
                mock_child._print_fn(f"spam line {i}")
            return {"completed": True, "final_response": "done"}

        mock_child.run_conversation.side_effect = run_and_spam

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="spam output", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        self.assertEqual(len(task.output_lines), 200)
        self.assertEqual(task.output_lines[0], "spam line 50")
        self.assertEqual(task.output_lines[-1], "spam line 249")


# =============================================================================
# TestConcurrentRegistry — Thread safety of registry
# =============================================================================

class TestConcurrentRegistry(unittest.TestCase):
    """Tests for thread safety of _get_task_registry and task operations."""

    def test_same_registry_from_two_threads(self):
        """Two threads calling _get_task_registry get the same registry objects."""
        parent = MockParentAgent()
        results = []
        barrier = threading.Barrier(2, timeout=5)

        def get_reg():
            barrier.wait()
            tasks, lock = _get_task_registry(parent)
            results.append((tasks, lock))

        t1 = threading.Thread(target=get_reg)
        t2 = threading.Thread(target=get_reg)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        self.assertEqual(len(results), 2)
        self.assertIs(results[0][0], results[1][0])  # same dict
        self.assertIs(results[0][1], results[1][1])  # same lock

    def test_concurrent_steer_and_cancel_one_fails(self):
        """Concurrent steer_task + cancel_task on same task: one should fail."""
        parent = MockParentAgent()
        tasks, _ = _get_task_registry(parent)
        task = AsyncTask(task_id="t-conflict", goal="test", status="running")
        mock_child = MagicMock()
        mock_child._steering_injection = queue.Queue()
        mock_child.interrupt = MagicMock()
        task.child_agent = mock_child
        tasks["t-conflict"] = task

        results = [None, None]
        barrier = threading.Barrier(2, timeout=5)

        def do_steer():
            barrier.wait()
            results[0] = json.loads(steer_task(task_id="t-conflict", message="msg", parent_agent=parent))

        def do_cancel():
            barrier.wait()
            results[1] = json.loads(cancel_task(task_id="t-conflict", parent_agent=parent))

        t1 = threading.Thread(target=do_steer)
        t2 = threading.Thread(target=do_cancel)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # At least one should report not-ok (not running anymore)
        oks = []
        for r in results:
            if r is not None:
                oks.append(r.get("ok", "error" not in r))
        # At least one operation should fail
        self.assertFalse(all(oks), f"Expected at least one failure, got: {results}")

    def test_multiple_simultaneous_tasks_in_registry(self):
        """Multiple tasks spawned simultaneously all appear in registry."""
        parent = make_mock_parent()
        task_ids = []
        hold_events = {}

        def slow_run_factory(task_id):
            evt = threading.Event()
            hold_events[task_id] = evt

            def slow_run(user_message):
                evt.wait(timeout=5.0)
                return {"completed": True, "final_response": f"done-{task_id}"}
            return slow_run

        for i in range(3):
            mock_child = make_mock_child()
            mock_child.run_conversation.side_effect = slow_run_factory(f"task-{i}")
            with _AsyncDelegatePatches(mock_child):
                result = json.loads(delegate_task_async(goal=f"goal {i}", parent_agent=parent))
                task_ids.append(result["task_id"])

        # All three should be in the registry
        tasks_dict, _ = _get_task_registry(parent)
        for tid in task_ids:
            self.assertIn(tid, tasks_dict)

        self.assertEqual(len(tasks_dict), 3)

        # Cleanup: release all hold events
        for evt in hold_events.values():
            evt.set()
        for tid in task_ids:
            tasks_dict[tid].done_event.wait(timeout=5.0)


# =============================================================================
# TestSkillLoading — _load_skill_for_subagent
# =============================================================================

class TestSkillLoading(unittest.TestCase):
    """Tests for _load_skill_for_subagent from tools.delegate_tool."""

    def test_empty_skill_returns_none(self):
        """Empty string skill identifier returns None."""
        from tools.delegate_tool import _load_skill_for_subagent
        self.assertIsNone(_load_skill_for_subagent(""))

    def test_none_skill_returns_none(self):
        """None skill identifier returns None."""
        from tools.delegate_tool import _load_skill_for_subagent
        self.assertIsNone(_load_skill_for_subagent(None))

    def test_whitespace_skill_returns_none(self):
        """Whitespace-only skill identifier returns None."""
        from tools.delegate_tool import _load_skill_for_subagent
        self.assertIsNone(_load_skill_for_subagent("   "))

    def test_nonexistent_skill_returns_none(self):
        """Nonexistent skill returns None (skill_view returns failure)."""
        from tools.delegate_tool import _load_skill_for_subagent
        with patch('tools.delegate_tool.skill_view' if hasattr(__import__('tools.delegate_tool', fromlist=['skill_view']), 'skill_view')
                   else 'tools.skills_tool.skill_view',
                   return_value=json.dumps({"success": False, "error": "not found"})):
            # Patch via the import path used inside the function
            pass

        # Use the actual import path as the function imports it dynamically
        with patch('tools.skills_tool.skill_view',
                   return_value=json.dumps({"success": False, "error": "not found"})):
            result = _load_skill_for_subagent("nonexistent-skill")
        self.assertIsNone(result)

    def test_valid_skill_returns_correct_data(self):
        """Valid skill returns dict with content, model, provider, name."""
        from tools.delegate_tool import _load_skill_for_subagent
        skill_data = {
            "success": True,
            "content": "Do the thing step by step.",
            "model": "anthropic/claude-sonnet-4",
            "provider": "openrouter",
            "name": "my-skill",
        }
        with patch('tools.skills_tool.skill_view',
                   return_value=json.dumps(skill_data)):
            result = _load_skill_for_subagent("my-skill")

        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "Do the thing step by step.")
        self.assertEqual(result["model"], "anthropic/claude-sonnet-4")
        self.assertEqual(result["provider"], "openrouter")
        self.assertEqual(result["name"], "my-skill")

    def test_skill_exception_returns_none(self):
        """If skill_view raises, returns None (doesn't propagate exception)."""
        from tools.delegate_tool import _load_skill_for_subagent
        with patch('tools.skills_tool.skill_view', side_effect=RuntimeError("db error")):
            result = _load_skill_for_subagent("broken-skill")
        self.assertIsNone(result)


# =============================================================================
# TestThreadLifecycle — Thread cleanup after task completion
# =============================================================================

class TestThreadLifecycle(unittest.TestCase):
    """Tests for thread cleanup and lifecycle management."""

    def test_thread_not_alive_after_completion(self):
        """After task completes, the thread is no longer alive."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="lifecycle test", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        # Give thread a moment to fully exit
        task.thread.join(timeout=5.0)
        self.assertFalse(task.thread.is_alive())

    def test_completed_task_has_completed_at_and_done_event(self):
        """Completed task has completed_at set and done_event is set."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="completion check", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        self.assertIsNotNone(task.completed_at)
        self.assertTrue(task.done_event.is_set())
        self.assertEqual(task.status, "completed")

    def test_child_removed_from_active_children(self):
        """child_agent is removed from parent's _active_children after completion."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        # Manually add child to active_children (simulating what _build_child_agent would do)
        parent._active_children.append(mock_child)

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="cleanup test", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        # Give cleanup code a moment
        time.sleep(0.1)
        self.assertNotIn(mock_child, parent._active_children)

    def test_failed_task_also_cleans_up(self):
        """Failed task still sets completed_at and done_event."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.side_effect = RuntimeError("boom")

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="fail cleanup", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        self.assertTrue(task.done_event.is_set())
        self.assertIsNotNone(task.completed_at)
        self.assertEqual(task.status, "failed")
        self.assertIn("boom", task.error)


# =============================================================================
# TestMultipleTasks — Multiple concurrent tasks
# =============================================================================

class TestMultipleTasks(unittest.TestCase):
    """Tests for managing multiple concurrent async tasks."""

    def test_spawn_3_tasks_all_appear_in_list(self):
        """Spawn 3 tasks, verify all appear in list_tasks output."""
        parent = make_mock_parent()
        hold_events = {}

        def slow_run_factory(tid):
            evt = threading.Event()
            hold_events[tid] = evt

            def slow_run(user_message):
                evt.wait(timeout=10.0)
                return {"completed": True, "final_response": f"done-{tid}"}
            return slow_run

        task_ids = []
        for i in range(3):
            mock_child = make_mock_child()
            mock_child.run_conversation.side_effect = slow_run_factory(f"t{i}")
            with _AsyncDelegatePatches(mock_child):
                result = json.loads(
                    delegate_task_async(goal=f"concurrent goal {i}", parent_agent=parent)
                )
                task_ids.append(result["task_id"])

        # list_tasks should show all 3
        listed = json.loads(list_tasks(parent_agent=parent))
        listed_ids = {t["task_id"] for t in listed["tasks"]}
        for tid in task_ids:
            self.assertIn(tid, listed_ids)
        self.assertEqual(len(listed["tasks"]), 3)

        # Cleanup
        for evt in hold_events.values():
            evt.set()
        tasks_dict = parent._async_tasks
        for tid in task_ids:
            tasks_dict[tid].done_event.wait(timeout=5.0)

    def test_collect_each_individually(self):
        """Spawn 3 tasks, collect each individually."""
        parent = make_mock_parent()

        task_ids = []
        for i in range(3):
            mock_child = make_mock_child()
            mock_child.run_conversation.return_value = {
                "completed": True, "final_response": f"result-{i}"
            }
            with _AsyncDelegatePatches(mock_child):
                result = json.loads(
                    delegate_task_async(goal=f"collectable {i}", parent_agent=parent)
                )
                task_ids.append(result["task_id"])

        # Wait for all to finish then collect
        tasks_dict = parent._async_tasks
        for tid in task_ids:
            tasks_dict[tid].done_event.wait(timeout=5.0)

        for i, tid in enumerate(task_ids):
            collected = json.loads(collect_task(task_id=tid, parent_agent=parent))
            self.assertEqual(collected["status"], "completed")
            self.assertEqual(collected["summary"], f"result-{i}")

    def test_cancel_one_others_still_running(self):
        """Cancel one task, verify others are still running."""
        parent = make_mock_parent()
        hold_events = {}

        def slow_run_factory(tid):
            evt = threading.Event()
            hold_events[tid] = evt

            def slow_run(user_message):
                evt.wait(timeout=10.0)
                return {"completed": True, "final_response": f"done-{tid}"}
            return slow_run

        task_ids = []
        for i in range(3):
            mock_child = make_mock_child()
            mock_child.run_conversation.side_effect = slow_run_factory(f"t{i}")
            with _AsyncDelegatePatches(mock_child):
                result = json.loads(
                    delegate_task_async(goal=f"cancel test {i}", parent_agent=parent)
                )
                task_ids.append(result["task_id"])

        # Cancel the first task
        cancel_result = json.loads(cancel_task(task_id=task_ids[0], parent_agent=parent))
        self.assertTrue(cancel_result["ok"])

        # Verify the other two are still running
        listed = json.loads(list_tasks(parent_agent=parent))
        status_map = {t["task_id"]: t["status"] for t in listed["tasks"]}

        self.assertEqual(status_map[task_ids[0]], "cancelled")
        self.assertEqual(status_map[task_ids[1]], "running")
        self.assertEqual(status_map[task_ids[2]], "running")

        # Cleanup
        for evt in hold_events.values():
            evt.set()
        tasks_dict = parent._async_tasks
        for tid in task_ids[1:]:
            tasks_dict[tid].done_event.wait(timeout=5.0)


# =============================================================================
# TestRegistryPersistence — Memory leak / retention behavior
# =============================================================================

class TestRegistryPersistence(unittest.TestCase):
    """Tests documenting that completed/failed/cancelled tasks remain in registry."""

    def test_completed_task_remains_in_registry(self):
        """Completed tasks are NOT removed from registry (documented behavior)."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.return_value = {
            "completed": True, "final_response": "done"
        }

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="persist test", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        tasks_dict, _ = _get_task_registry(parent)
        self.assertIn(task_id, tasks_dict)
        self.assertEqual(tasks_dict[task_id].status, "completed")

    def test_failed_task_remains_in_registry(self):
        """Failed tasks are NOT removed from registry."""
        parent = make_mock_parent()
        mock_child = make_mock_child()
        mock_child.run_conversation.side_effect = RuntimeError("fail")

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="fail persist", parent_agent=parent))
            task_id = result["task_id"]
            task = parent._async_tasks[task_id]
            task.done_event.wait(timeout=5.0)

        tasks_dict, _ = _get_task_registry(parent)
        self.assertIn(task_id, tasks_dict)
        self.assertEqual(tasks_dict[task_id].status, "failed")

    def test_cancelled_task_remains_in_registry(self):
        """Cancelled tasks are NOT removed from registry."""
        parent = make_mock_parent()
        mock_child = make_mock_child()

        hold_event = threading.Event()
        def slow_run(user_message):
            hold_event.wait(timeout=5.0)
            return {"completed": True, "final_response": "done"}
        mock_child.run_conversation.side_effect = slow_run

        with _AsyncDelegatePatches(mock_child):
            result = json.loads(delegate_task_async(goal="cancel persist", parent_agent=parent))
            task_id = result["task_id"]

        cancel_result = json.loads(cancel_task(task_id=task_id, parent_agent=parent))
        self.assertTrue(cancel_result["ok"])

        tasks_dict, _ = _get_task_registry(parent)
        self.assertIn(task_id, tasks_dict)
        self.assertEqual(tasks_dict[task_id].status, "cancelled")

        # Cleanup
        hold_event.set()
        tasks_dict[task_id].done_event.wait(timeout=5.0)

    def test_list_tasks_shows_all_including_completed(self):
        """list_tasks shows all tasks including completed, failed, cancelled ones."""
        parent = make_mock_parent()
        tasks, _ = _get_task_registry(parent)

        # Manually add tasks of each status
        for status in ["running", "completed", "failed", "cancelled"]:
            t = AsyncTask(task_id=f"t-{status}", goal=f"goal {status}", status=status)
            if status != "running":
                t.completed_at = t.started_at + 1.0
                t.done_event.set()
            tasks[f"t-{status}"] = t

        result = json.loads(list_tasks(parent_agent=parent))
        reported = {t["task_id"]: t["status"] for t in result["tasks"]}

        self.assertEqual(len(reported), 4)
        self.assertEqual(reported["t-running"], "running")
        self.assertEqual(reported["t-completed"], "completed")
        self.assertEqual(reported["t-failed"], "failed")
        self.assertEqual(reported["t-cancelled"], "cancelled")


if __name__ == "__main__":
    unittest.main()
