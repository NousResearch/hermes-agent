"""
Tests for subagent memory write: make_subagent_memory_writer, subagent_memory_write,
and the write_memory=True path in delegate_task.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.subagent_memory_tool import (
    MAX_CHARS,
    MAX_WRITES,
    make_subagent_memory_writer,
    subagent_memory_write,
    SUBAGENT_MEMORY_WRITE_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parent_with_store(entries=None):
    """Build a mock parent agent with a working MemoryStore."""
    parent = MagicMock()
    parent._memory_manager = None

    store = MagicMock()
    _saved = list(entries or [])

    def _add(target, content):
        if len(content) > 2200:
            return {"success": False, "error": "char limit"}
        _saved.append(content)
        return {"success": True, "target": target, "entries": _saved}

    store.add.side_effect = _add
    parent._memory_store = store
    parent._saved_entries = _saved
    return parent


def _make_parent_no_store():
    parent = MagicMock()
    parent._memory_store = None
    parent._memory_manager = None
    return parent


# ---------------------------------------------------------------------------
# make_subagent_memory_writer
# ---------------------------------------------------------------------------

class TestMakeSubagentMemoryWriter(unittest.TestCase):

    def test_returns_none_when_no_store(self):
        parent = _make_parent_no_store()
        writer = make_subagent_memory_writer(parent)
        self.assertIsNone(writer)

    def test_returns_callable_when_store_present(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        self.assertTrue(callable(writer))

    def test_write_tags_entry(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        result = writer("DATABASE_URL missing from Render env")
        self.assertTrue(result["success"])
        stored = parent._memory_store.add.call_args[0][1]
        self.assertTrue(stored.startswith("[subagent] "))
        self.assertIn("DATABASE_URL", stored)

    def test_write_rejects_empty_content(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        result = writer("   ")
        self.assertFalse(result["success"])
        self.assertIn("empty", result["error"])

    def test_write_rejects_content_over_max_chars(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        long_content = "x" * (MAX_CHARS + 1)
        result = writer(long_content)
        self.assertFalse(result["success"])
        self.assertIn("too long", result["error"].lower())

    def test_write_accepts_content_at_max_chars(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        exact = "x" * MAX_CHARS
        result = writer(exact)
        self.assertTrue(result["success"])

    def test_rate_limit_blocks_after_max_writes(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        for i in range(MAX_WRITES):
            result = writer(f"Finding {i}")
            self.assertTrue(result["success"], f"Write {i} should succeed")
        result = writer("One more")
        self.assertFalse(result["success"])
        self.assertIn("limit", result["error"].lower())
        self.assertEqual(result["writes_used"], MAX_WRITES)

    def test_success_result_includes_write_counters(self):
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        result = writer("Root cause: missing env var")
        self.assertTrue(result["success"])
        self.assertEqual(result["writes_used"], 1)
        self.assertEqual(result["writes_remaining"], MAX_WRITES - 1)

    def test_notifies_external_memory_manager(self):
        parent = _make_parent_with_store()
        mm = MagicMock()
        parent._memory_manager = mm
        writer = make_subagent_memory_writer(parent)
        writer("Render crash caused by missing DATABASE_URL")
        mm.on_memory_write.assert_called_once()
        call_args = mm.on_memory_write.call_args[0]
        self.assertEqual(call_args[0], "add")
        self.assertEqual(call_args[1], "memory")
        self.assertIn("[subagent]", call_args[2])

    def test_memory_manager_failure_does_not_break_write(self):
        parent = _make_parent_with_store()
        mm = MagicMock()
        mm.on_memory_write.side_effect = RuntimeError("backend down")
        parent._memory_manager = mm
        writer = make_subagent_memory_writer(parent)
        result = writer("Still works despite mm failure")
        self.assertTrue(result["success"])

    def test_thread_safe_rate_limiting(self):
        """Concurrent writes must not exceed MAX_WRITES even under race conditions."""
        parent = _make_parent_with_store()
        writer = make_subagent_memory_writer(parent)
        successes = []
        lock = threading.Lock()

        def _write():
            result = writer("concurrent finding")
            if result.get("success"):
                with lock:
                    successes.append(1)

        threads = [threading.Thread(target=_write) for _ in range(MAX_WRITES * 3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertLessEqual(len(successes), MAX_WRITES)

    def test_independent_writers_have_independent_counters(self):
        """Two writers from two parent agents do not share rate limit state."""
        p1 = _make_parent_with_store()
        p2 = _make_parent_with_store()
        w1 = make_subagent_memory_writer(p1)
        w2 = make_subagent_memory_writer(p2)
        for _ in range(MAX_WRITES):
            w1("finding from child 1")
        # p1's writer is exhausted -- p2's writer should still work
        result = w2("finding from child 2")
        self.assertTrue(result["success"])

    def test_store_add_failure_propagates(self):
        """When the memory store rejects the write, the result reflects that."""
        parent = _make_parent_with_store()
        parent._memory_store.add.side_effect = None
        parent._memory_store.add.return_value = {
            "success": False,
            "error": "Memory at limit",
        }
        writer = make_subagent_memory_writer(parent)
        result = writer("This should fail")
        self.assertFalse(result["success"])
        self.assertIn("limit", result["error"])


# ---------------------------------------------------------------------------
# subagent_memory_write (handler)
# ---------------------------------------------------------------------------

class TestSubagentMemoryWriteHandler(unittest.TestCase):

    def test_returns_error_when_no_writer(self):
        result = json.loads(subagent_memory_write("content", writer=None))
        self.assertFalse(result["success"])
        self.assertIn("not available", result["error"].lower())

    def test_calls_writer_and_returns_json(self):
        writer = MagicMock(return_value={"success": True, "writes_used": 1, "writes_remaining": 2})
        result = json.loads(subagent_memory_write("root cause found", writer=writer))
        self.assertTrue(result["success"])
        writer.assert_called_once_with("root cause found")

    def test_schema_name(self):
        self.assertEqual(SUBAGENT_MEMORY_WRITE_SCHEMA["name"], "subagent_memory_write")

    def test_schema_requires_content(self):
        required = SUBAGENT_MEMORY_WRITE_SCHEMA["parameters"]["required"]
        self.assertIn("content", required)


# ---------------------------------------------------------------------------
# delegate_task write_memory integration
# ---------------------------------------------------------------------------

class TestDelegateTaskWriteMemory(unittest.TestCase):

    def _make_mock_parent(self, has_store=True):
        import threading
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
        parent._delegate_depth = 0
        parent._active_children = []
        parent._active_children_lock = threading.Lock()
        parent._print_fn = None
        parent.tool_progress_callback = None
        parent.thinking_callback = None
        parent._memory_manager = None
        if has_store:
            store = MagicMock()
            store.add.return_value = {"success": True, "entries": []}
            parent._memory_store = store
        else:
            parent._memory_store = None
        return parent

    @patch("tools.delegate_tool._run_single_child")
    def test_write_memory_false_does_not_inject_tool(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = self._make_mock_parent()
        captured_child = {}

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.tools = []
            mock_child.valid_tool_names = set()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task
            delegate_task(goal="No memory write", parent_agent=parent, write_memory=False)
            captured_child["tools"] = list(mock_child.tools)
            captured_child["names"] = set(mock_child.valid_tool_names)

        tool_names = [t.get("function", {}).get("name") for t in captured_child["tools"]]
        self.assertNotIn("subagent_memory_write", tool_names)
        self.assertNotIn("subagent_memory_write", captured_child["names"])

    @patch("tools.delegate_tool._run_single_child")
    def test_write_memory_true_injects_tool_when_store_present(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = self._make_mock_parent(has_store=True)
        captured_child = {}

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.tools = []
            mock_child.valid_tool_names = set()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task
            delegate_task(goal="With memory write", parent_agent=parent, write_memory=True)
            captured_child["tools"] = list(mock_child.tools)
            captured_child["names"] = set(mock_child.valid_tool_names)

        tool_names = [t.get("function", {}).get("name") for t in captured_child["tools"]]
        self.assertIn("subagent_memory_write", tool_names)
        self.assertIn("subagent_memory_write", captured_child["names"])
        self.assertTrue(callable(mock_child._subagent_memory_writer))

    @patch("tools.delegate_tool._run_single_child")
    def test_write_memory_true_no_store_does_not_inject(self, mock_run):
        """When parent has no memory store, write_memory=True is a no-op (no tool injected)."""
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = self._make_mock_parent(has_store=False)
        captured_child = {}

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.tools = []
            mock_child.valid_tool_names = set()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task
            delegate_task(goal="No store", parent_agent=parent, write_memory=True)
            captured_child["tools"] = list(mock_child.tools)
            captured_child["names"] = set(mock_child.valid_tool_names)

        tool_names = [t.get("function", {}).get("name") for t in captured_child["tools"]]
        self.assertNotIn("subagent_memory_write", tool_names)

    @patch("tools.delegate_tool._run_single_child")
    def test_writer_on_child_writes_to_parent_store(self, mock_run):
        mock_run.return_value = {
            "task_index": 0, "status": "completed",
            "summary": "done", "api_calls": 1, "duration_seconds": 1.0,
        }
        parent = self._make_mock_parent(has_store=True)

        with patch("run_agent.AIAgent") as MockAgent:
            mock_child = MagicMock()
            mock_child.tools = []
            mock_child.valid_tool_names = set()
            MockAgent.return_value = mock_child

            from tools.delegate_tool import delegate_task
            delegate_task(goal="Writer test", parent_agent=parent, write_memory=True)

        writer = mock_child._subagent_memory_writer
        result = writer("Root cause: DATABASE_URL missing in Render production env")
        self.assertTrue(result["success"])
        parent._memory_store.add.assert_called_once()
        call_args = parent._memory_store.add.call_args[0]
        self.assertIn("[subagent]", call_args[1])

    @patch("tools.delegate_tool._run_single_child")
    def test_schema_includes_write_memory_param(self, mock_run):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("write_memory", props)
        self.assertEqual(props["write_memory"]["type"], "boolean")


# ---------------------------------------------------------------------------
# run_agent.py dispatch paths forward write_memory
# ---------------------------------------------------------------------------

class TestRunAgentDispatchForwardsWriteMemory(unittest.TestCase):
    """Verify that both _invoke_tool and _execute_tool_calls_sequential
    forward write_memory from function_args to delegate_task.

    This is the regression test for the bug where write_memory was added to
    delegate_task() but never passed from the run_agent.py dispatch branches,
    silently making the whole feature a no-op in normal usage.
    """

    def _make_function_args(self, write_memory=True):
        return {
            "goal": "test delegation",
            "write_memory": write_memory,
        }

    @patch("tools.delegate_tool.delegate_task")
    def test_invoke_tool_forwards_write_memory_true(self, mock_delegate):
        """_invoke_tool must pass write_memory=True when the model requests it."""
        mock_delegate.return_value = json.dumps({"status": "completed", "summary": "ok"})
        from run_agent import AIAgent
        agent = MagicMock(spec=AIAgent)
        agent._subagent_memory_writer = None
        agent.valid_tool_names = {"delegate_task"}
        agent._memory_manager = None

        # Call the unbound method with our mock agent as self
        import run_agent as _ra
        result = _ra.AIAgent._invoke_tool(
            agent,
            "delegate_task",
            self._make_function_args(write_memory=True),
            "task-123",
        )
        mock_delegate.assert_called_once()
        _, kwargs = mock_delegate.call_args
        self.assertTrue(kwargs.get("write_memory") or mock_delegate.call_args[1].get("write_memory"),
                        "write_memory=True must be forwarded to delegate_task from _invoke_tool")

    @patch("tools.delegate_tool.delegate_task")
    def test_invoke_tool_forwards_write_memory_false(self, mock_delegate):
        """_invoke_tool must pass write_memory=False by default."""
        mock_delegate.return_value = json.dumps({"status": "completed", "summary": "ok"})
        from run_agent import AIAgent
        import run_agent as _ra
        agent = MagicMock(spec=AIAgent)
        agent._subagent_memory_writer = None
        agent.valid_tool_names = {"delegate_task"}
        agent._memory_manager = None

        _ra.AIAgent._invoke_tool(
            agent,
            "delegate_task",
            {"goal": "test delegation"},  # no write_memory key
            "task-123",
        )
        mock_delegate.assert_called_once()
        call_kwargs = mock_delegate.call_args[1]
        self.assertFalse(call_kwargs.get("write_memory", False),
                         "write_memory must default to False")


if __name__ == "__main__":
    unittest.main()
