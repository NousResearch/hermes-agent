#!/usr/bin/env python3
"""
Tests for the read-loop detection mechanism in file_tools.

Verifies that:
1. Only *consecutive* identical reads trigger warnings/blocks
2. Any other tool call in between resets the consecutive counter
3. Warn on 3rd consecutive, block on 4th+
4. Different regions/files/tasks don't trigger false warnings
5. get_read_files_summary returns accurate history (unaffected by search keys)
6. clear_read_tracker resets state
7. notify_other_tool_call resets consecutive counters
8. Context compression injects file-read history

Run with:  python -m pytest tests/tools/test_read_loop_detection.py -v
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from tools.file_tools import (
    read_file_tool,
    search_tool,
    notify_other_tool_call,
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
    """Verify that read_file_tool detects and warns on consecutive re-reads."""

    def setUp(self):
        _read_tracker.clear()

    def tearDown(self):
        _read_tracker.clear()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_first_read_has_no_warning(self, _mock_ops):
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_second_consecutive_read_no_warning(self, _mock_ops):
        """2nd consecutive read should NOT warn (threshold is 3)."""
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        result = json.loads(
            read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        )
        self.assertNotIn("_warning", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_third_consecutive_read_has_warning(self, _mock_ops):
        """3rd consecutive read of the same region triggers a warning."""
        for _ in range(2):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("3 times", result["_warning"])
        # Warning still returns content
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_fourth_consecutive_read_is_blocked(self, _mock_ops):
        """4th consecutive read of the same region is BLOCKED — no content."""
        for _ in range(3):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("error", result)
        self.assertIn("BLOCKED", result["error"])
        self.assertIn("4 times", result["error"])
        self.assertNotIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_fifth_consecutive_read_still_blocked(self, _mock_ops):
        """Subsequent reads remain blocked with incrementing count."""
        for _ in range(4):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("BLOCKED", result["error"])
        self.assertIn("5 times", result["error"])

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_region_resets_consecutive(self, _mock_ops):
        """Reading a different region of the same file resets consecutive count."""
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        read_file_tool("/tmp/test.py", offset=1, limit=500, task_id="t1")
        # Now read a different region — this resets the consecutive counter
        result = json.loads(
            read_file_tool("/tmp/test.py", offset=501, limit=500, task_id="t1")
        )
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_file_resets_consecutive(self, _mock_ops):
        """Reading a different file resets the consecutive counter."""
        read_file_tool("/tmp/a.py", task_id="t1")
        read_file_tool("/tmp/a.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/b.py", task_id="t1"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_tasks_isolated(self, _mock_ops):
        """Different task_ids have separate consecutive counters."""
        read_file_tool("/tmp/test.py", task_id="task_a")
        result = json.loads(
            read_file_tool("/tmp/test.py", task_id="task_b")
        )
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_warning_still_returns_content(self, _mock_ops):
        """Even with a warning (3rd read), the file content is still returned."""
        for _ in range(2):
            read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("content", result)
        self.assertIn("content of /tmp/test.py", result["content"])


class TestNotifyOtherToolCall(unittest.TestCase):
    """Verify that notify_other_tool_call resets the consecutive counter."""

    def setUp(self):
        _read_tracker.clear()

    def tearDown(self):
        _read_tracker.clear()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_other_tool_resets_consecutive(self, _mock_ops):
        """After another tool runs, re-reading the same file is NOT consecutive."""
        read_file_tool("/tmp/test.py", task_id="t1")
        read_file_tool("/tmp/test.py", task_id="t1")
        # Simulate a different tool being called
        notify_other_tool_call("t1")
        # This should be treated as a fresh read (consecutive reset)
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_other_tool_prevents_block(self, _mock_ops):
        """Agent can keep reading if other tools are used in between."""
        for i in range(10):
            read_file_tool("/tmp/test.py", task_id="t1")
            notify_other_tool_call("t1")
        # After 10 reads interleaved with other tools, still no warning
        result = json.loads(read_file_tool("/tmp/test.py", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)
        self.assertIn("content", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_notify_on_unknown_task_is_safe(self, _mock_ops):
        """notify_other_tool_call on a task that hasn't read anything is a no-op."""
        notify_other_tool_call("nonexistent_task")  # Should not raise





class TestSearchLoopDetection(unittest.TestCase):
    """Verify that search_tool detects and blocks consecutive repeated searches."""

    def setUp(self):
        _read_tracker.clear()

    def tearDown(self):
        _read_tracker.clear()

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_first_search_no_warning(self, _mock_ops):
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_second_consecutive_search_no_warning(self, _mock_ops):
        """2nd consecutive search should NOT warn (threshold is 3)."""
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_third_consecutive_search_has_warning(self, _mock_ops):
        """3rd consecutive identical search triggers a warning."""
        for _ in range(2):
            search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertIn("_warning", result)
        self.assertIn("3 times", result["_warning"])
        # Warning still returns results
        self.assertIn("matches", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_fourth_consecutive_search_is_blocked(self, _mock_ops):
        """4th consecutive identical search is BLOCKED."""
        for _ in range(3):
            search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertIn("error", result)
        self.assertIn("BLOCKED", result["error"])
        self.assertNotIn("matches", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_pattern_resets_consecutive(self, _mock_ops):
        """A different search pattern resets the consecutive counter."""
        search_tool("def main", task_id="t1")
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("class Foo", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_different_task_isolated(self, _mock_ops):
        """Different tasks have separate consecutive counters."""
        search_tool("def main", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t2"))
        self.assertNotIn("_warning", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_other_tool_resets_search_consecutive(self, _mock_ops):
        """notify_other_tool_call resets search consecutive counter too."""
        search_tool("def main", task_id="t1")
        search_tool("def main", task_id="t1")
        notify_other_tool_call("t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_pagination_offset_does_not_count_as_repeat(self, _mock_ops):
        """Paginating truncated results should not be blocked as a repeat search."""
        for offset in (0, 50, 100, 150):
            result = json.loads(search_tool("def main", task_id="t1", offset=offset, limit=50))
            self.assertNotIn("_warning", result)
            self.assertNotIn("error", result)

    @patch("tools.file_tools._get_file_ops", return_value=_make_fake_file_ops())
    def test_read_between_searches_resets_consecutive(self, _mock_ops):
        """A read_file call between searches resets search consecutive counter."""
        search_tool("def main", task_id="t1")
        search_tool("def main", task_id="t1")
        # A read changes the last_key, resetting consecutive for the search
        read_file_tool("/tmp/test.py", task_id="t1")
        result = json.loads(search_tool("def main", task_id="t1"))
        self.assertNotIn("_warning", result)
        self.assertNotIn("error", result)


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
