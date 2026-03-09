#!/usr/bin/env python3
"""
Tests for the read-loop detection mechanism in file_tools.

Verifies that:
1. Re-reading the same file region produces a warning
2. Different regions/files don't trigger false warnings
3. Task isolation works (different tasks have separate trackers)
4. get_read_files_summary returns accurate history
5. clear_read_tracker resets state
6. Context compression injects file-read history

Run with:  python -m pytest tests/tools/test_read_loop_detection.py -v
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from tools.file_tools import (
    read_file_tool,
    search_tool,
    get_read_files_summary,
    clear_read_tracker,
    _read_tracker,
)


class _FakeReadResult:
    """Minimal stand-in for FileOperations.read_file return value."""
    def __init__(self, content="line1\nline2\n", total_lines=2):
        self.content = content
        self._total_lines = total_lines

    def to_dict(self):
        return {"content": self.content, "total_lines": self._total_lines}


def _fake_read_file(path, offset=1, limit=500):
    return _FakeReadResult(content=f"content of {path}", total_lines=10)


class _FakeSearchResult:
    """Minimal stand-in for FileOperations.search return value."""
    def __init__(self):
        self.matches = []

    def to_dict(self):
        return {"matches": [{"file": "test.py", "line": 1, "text": "match"}]}


def _make_fake_file_ops():
    fake = MagicMock()
    fake.read_file = _fake_read_file
    fake.search = lambda **kw: _FakeSearchResult()
    return fake


class TestReadLoopDetection(unittest.TestCase):
    """Verify that read_file_tool detects and warns on re-reads."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_first_read_has_no_warning(self, _mock_ops):
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_second_read_same_region_has_warning(self, _mock_ops):
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        result = json.loads(
            read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        )
        self.assertIn("_warning", result)
        self.assertIn("already read", result["_warning"])
        self.assertIn("2 times", result["_warning"])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_third_read_is_blocked(self, _mock_ops):
        """3rd read of the same region returns error, no content."""
        for _ in range(2):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("error", result)
        self.assertIn("BLOCKED", result["error"])
        self.assertNotIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_fourth_read_still_blocked(self, _mock_ops):
        """Subsequent reads remain blocked with incrementing count."""
        for _ in range(3):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("BLOCKED", result["error"])
        self.assertIn("4 times", result["error"])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_region_no_warning(self, _mock_ops):
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        result = json.loads(
            read_file_tool("/tmp/test.py", offset=501, limit=500, task_id="t1")
        )
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_file_no_warning(self, _mock_ops):
        read_file_tool("/tmp/a.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/b.py", task_id="t1"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_tasks_isolated(self, _mock_ops):
        read_file_tool("/tmp/test.py", task_id="task_a")
        result = json.loads(
            read_file_tool("/tmp/test.py", task_id="task_b")
        )
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_warning_still_returns_content(self, _mock_ops):
        """Even with a warning, the file content is still returned."""
        read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("content", result)
        self.assertIn("content of /tmp/test.py", result["content"])


class TestReadFilesSummary(unittest.TestCase):
    """Verify get_read_files_summary returns accurate file-read history."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_empty_when_no_reads(self, _mock_ops):
        summary = get_read_files_summary("t1")
        self.assertEqual(summary, [])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_single_file_single_region(self, _mock_ops):
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        summary = get_read_files_summary("t1")
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["path"], "/tmp/test.py")
        self.assertIn("lines 1-500", summary[0]["regions"])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_single_file_multiple_regions(self, _mock_ops):
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        read_file_tool("/tmp/test.py", offset=501, limit=500, task_id="t1")
        summary = get_read_files_summary("t1")
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(summary[0]["regions"]), 2)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_multiple_files(self, _mock_ops):
        read_file_tool("/tmp/a.py", task_id="t1")
        read_file_tool("/tmp/b.py", task_id="t1")
        summary = get_read_files_summary("t1")
        self.assertEqual(len(summary), 2)
        paths = [s["path"] for s in summary]
        self.assertIn("/tmp/a.py", paths)
        self.assertIn("/tmp/b.py", paths)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_task_has_separate_summary(self, _mock_ops):
        read_file_tool("/tmp/a.py", task_id="task_a")
        read_file_tool("/tmp/b.py", task_id="task_b")
        summary_a = get_read_files_summary("task_a")
        summary_b = get_read_files_summary("task_b")
        self.assertEqual(len(summary_a), 1)
        self.assertEqual(summary_a[0]["path"], "/tmp/a.py")
        self.assertEqual(len(summary_b), 1)
        self.assertEqual(summary_b[0]["path"], "/tmp/b.py")


class TestClearReadTracker(unittest.TestCase):
    """Verify clear_read_tracker resets state properly."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_clear_specific_task(self, _mock_ops):
        read_file_tool("/tmp/test.py", task_id="t1")
        read_file_tool("/tmp/test.py", task_id="t2")
        clear_read_tracker("t1")
        self.assertEqual(get_read_files_summary("t1"), [])
        self.assertEqual(len(get_read_files_summary("t2")), 1)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_clear_all(self, _mock_ops):
        read_file_tool("/tmp/test.py", task_id="t1")
        read_file_tool("/tmp/test.py", task_id="t2")
        clear_read_tracker()
        self.assertEqual(get_read_files_summary("t1"), [])
        self.assertEqual(get_read_files_summary("t2"), [])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_clear_then_reread_no_warning(self, _mock_ops):
        read_file_tool("/tmp/test.py", task_id="t1")
        clear_read_tracker("t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertNotIn("_warning", result)


class TestCompressionFileHistory(unittest.TestCase):
    """Verify that _compress_context injects file-read history."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_compress_context_includes_read_files(self, _mock_ops):
        """After reading files, _compress_context should inject a message
        listing which files were already read."""
        # Simulate reads
        read_file_tool("/tmp/foo.py", offset=1, limit=100, task_id="compress_test")
        read_file_tool("/tmp/bar.py", offset=1, limit=200, task_id="compress_test")

        # Build minimal messages for compression (need enough messages)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Analyze the codebase."},
            {"role": "assistant", "content": "I'll read the files."},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": "Reading more files."},
            {"role": "user", "content": "What did you find?"},
            {"role": "assistant", "content": "Here are my findings."},
            {"role": "user", "content": "Great, write the fix."},
            {"role": "assistant", "content": "Working on it."},
            {"role": "user", "content": "Status?"},
        ]

        # Mock the compressor to return a simple compression
        mock_compressor = MagicMock()
        mock_compressor.compress.return_value = [
            messages[0],  # system
            messages[1],  # first user
            {"role": "user", "content": "[CONTEXT SUMMARY]: Files were analyzed."},
            messages[-1],  # last user
        ]
        mock_compressor.last_prompt_tokens = 5000

        # Mock the agent's _compress_context dependencies
        mock_agent = MagicMock()
        mock_agent.context_compressor = mock_compressor
        mock_agent._todo_store.format_for_injection.return_value = None
        mock_agent._session_db = None
        mock_agent.quiet_mode = True
        mock_agent._invalidate_system_prompt = MagicMock()
        mock_agent._build_system_prompt = MagicMock(return_value="system prompt")
        mock_agent._cached_system_prompt = None

        # Call the real _compress_context
        from run_agent import AIAgent
        result, _ = AIAgent._compress_context(
            mock_agent, messages, "system prompt",
            approx_tokens=5000, task_id="compress_test",
        )

        # Find the injected file-read history message
        file_history_msgs = [
            m for m in result
            if isinstance(m.get("content"), str)
            and "already read" in m.get("content", "").lower()
        ]
        self.assertEqual(len(file_history_msgs), 1,
                         "Should inject exactly one file-read history message")

        history_content = file_history_msgs[0]["content"]
        self.assertIn("/tmp/foo.py", history_content)
        self.assertIn("/tmp/bar.py", history_content)
        self.assertIn("do NOT re-read", history_content)


class TestSearchLoopDetection(unittest.TestCase):
    """Verify that search_tool detects and blocks repeated searches."""

    def setUp(self):
        clear_read_tracker()

    def tearDown(self):
        clear_read_tracker()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_first_search_no_warning(self, _mock_ops):
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_second_search_has_warning(self, _mock_ops):
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("2 times", result["_warning"])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_third_search_is_blocked(self, _mock_ops):
        for _ in range(2):
            search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertIn("error", result)
        self.assertIn("BLOCKED", result["error"])
        self.assertNotIn("matches", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_pattern_no_warning(self, _mock_ops):
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("class Foo", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_task_isolated(self, _mock_ops):
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t2"))
        self.assertNotIn("_warning", result)


class TestTodoInjectionFiltering(unittest.TestCase):
    """Verify that format_for_injection filters completed/cancelled todos."""

    def test_filters_completed_and_cancelled(self):
        from tools.todo_tool import TodoStore
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Read codebase", "status": "completed"},
            {"id": "2", "content": "Write fix", "status": "in_progress"},
            {"id": "3", "content": "Run tests", "status": "pending"},
            {"id": "4", "content": "Abandoned", "status": "cancelled"},
        ])
        injection = store.format_for_injection()
        self.assertNotIn("Read codebase", injection)
        self.assertNotIn("Abandoned", injection)
        self.assertIn("Write fix", injection)
        self.assertIn("Run tests", injection)

    def test_all_completed_returns_none(self):
        from tools.todo_tool import TodoStore
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Done", "status": "completed"},
            {"id": "2", "content": "Also done", "status": "cancelled"},
        ])
        self.assertIsNone(store.format_for_injection())

    def test_empty_store_returns_none(self):
        from tools.todo_tool import TodoStore
        store = TodoStore()
        self.assertIsNone(store.format_for_injection())

    def test_all_active_included(self):
        from tools.todo_tool import TodoStore
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Task A", "status": "pending"},
            {"id": "2", "content": "Task B", "status": "in_progress"},
        ])
        injection = store.format_for_injection()
        self.assertIn("Task A", injection)
        self.assertIn("Task B", injection)


if __name__ == "__main__":
    unittest.main()
