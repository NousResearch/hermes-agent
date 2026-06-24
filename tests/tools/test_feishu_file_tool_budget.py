"""Feishu low-token behavior for file tools."""

import json
from unittest.mock import patch

from tools.file_tools import (
    notify_other_tool_call,
    read_file_tool,
    search_tool,
    _read_tracker,
    _read_tracker_lock,
)


class FakeReadResult:
    def __init__(self, content="1|hello\n"):
        self.content = content

    def to_dict(self):
        return {
            "content": self.content,
            "file_size": len(self.content),
            "total_lines": len(self.content.splitlines()),
            "truncated": False,
        }


class FakeSearchResult:
    def __init__(self):
        self.matches = []

    def to_dict(self):
        return {
            "total_count": 30,
            "files": [f"/tmp/file_{i}.py" for i in range(30)],
            "truncated": True,
        }


class FakeFileOps:
    def __init__(self):
        self.read_calls = []
        self.search_calls = []

    def read_file(self, path, offset, limit):
        self.read_calls.append((path, offset, limit))
        return FakeReadResult()

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return FakeSearchResult()


def _clear_task(task_id):
    with _read_tracker_lock:
        _read_tracker.pop(task_id, None)


def test_feishu_read_file_default_and_explicit_limit_are_narrowed(tmp_path):
    task_id = "feishu-read-limit"
    _clear_task(task_id)
    target = tmp_path / "sample.txt"
    target.write_text("hello\n" * 300)
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        read_file_tool(str(target), task_id=task_id, platform="feishu")
        read_file_tool(str(target), limit=999, task_id=task_id, platform="feishu")

    assert fake.read_calls[0][2] == 120
    assert fake.read_calls[1][2] == 200


def test_cli_read_file_keeps_existing_default_limit(tmp_path):
    task_id = "cli-read-limit"
    _clear_task(task_id)
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        read_file_tool(str(target), task_id=task_id, platform="cli")

    assert fake.read_calls[0][2] == 500


def test_feishu_search_files_defaults_to_files_only_and_limit_twenty(tmp_path):
    task_id = "feishu-search-limit"
    _clear_task(task_id)
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        search_tool("needle", path=str(tmp_path), task_id=task_id, platform="feishu")

    call = fake.search_calls[0]
    assert call["limit"] == 20
    assert call["output_mode"] == "files_only"


def test_feishu_file_tool_budget_blocks_fifth_read_or_search(tmp_path):
    task_id = "feishu-budget"
    _clear_task(task_id)
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        read_file_tool(str(target), offset=1, task_id=task_id, platform="feishu")
        read_file_tool(str(target), offset=2, task_id=task_id, platform="feishu")
        search_tool("a", path=str(tmp_path), task_id=task_id, platform="feishu")
        search_tool("b", path=str(tmp_path), task_id=task_id, platform="feishu")
        result = json.loads(search_tool("c", path=str(tmp_path), task_id=task_id, platform="feishu"))

    assert result["error"].startswith("BLOCKED: Feishu low-token file budget")
    assert result["recommendation"] == "Switch to execute_code for batched local diagnosis."


def test_non_read_tool_resets_feishu_file_tool_budget(tmp_path):
    task_id = "feishu-budget-reset"
    _clear_task(task_id)
    target = tmp_path / "sample.txt"
    target.write_text("hello\n")
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        for i in range(4):
            read_file_tool(str(target), offset=i + 1, task_id=task_id, platform="feishu")
        notify_other_tool_call(task_id)
        result = json.loads(read_file_tool(str(target), offset=10, task_id=task_id, platform="feishu"))

    assert "error" not in result
