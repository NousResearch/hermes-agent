"""Feishu low-token behavior for file tools."""

import json
from pathlib import Path
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

    def to_dict(self, **_kwargs):
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


def test_feishu_read_file_blocks_raw_cron_jobs_json(tmp_path):
    task_id = "feishu-read-cron-jobs-json"
    _clear_task(task_id)
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text('{"jobs": []}\n')
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        result = json.loads(read_file_tool(str(jobs_path), task_id=task_id, platform="feishu"))

    assert result["error"].startswith("BLOCKED: Feishu should not read raw cron/jobs.json")
    assert result["recommendation"] == "Use cronjob(action='list') or execute_code with a targeted jobs filter."
    assert fake.read_calls == []


def test_feishu_deep_read_file_blocks_raw_cron_jobs_json(tmp_path):
    task_id = "feishu-deep-read-cron-jobs-json"
    _clear_task(task_id)
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text('{"jobs": []}\n')
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        result = json.loads(read_file_tool(str(jobs_path), task_id=task_id, platform="feishu_deep"))

    assert result["error"].startswith("BLOCKED: Feishu should not read raw cron/jobs.json")
    assert fake.read_calls == []


def test_cli_read_file_allows_raw_cron_jobs_json(tmp_path):
    task_id = "cli-read-cron-jobs-json"
    _clear_task(task_id)
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir()
    jobs_path = cron_dir / "jobs.json"
    jobs_path.write_text('{"jobs": []}\n')
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        result = json.loads(read_file_tool(str(jobs_path), task_id=task_id, platform="cli"))

    assert "error" not in result
    assert fake.read_calls[0][0] == str(jobs_path)


def test_feishu_search_files_defaults_to_files_only_and_limit_twenty(tmp_path):
    task_id = "feishu-search-limit"
    _clear_task(task_id)
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        search_tool("needle", path=str(tmp_path), task_id=task_id, platform="feishu")

    call = fake.search_calls[0]
    assert call["limit"] == 20
    assert call["output_mode"] == "files_only"


def test_feishu_search_files_blocks_default_home_root(monkeypatch):
    task_id = "feishu-search-home-root"
    _clear_task(task_id)
    monkeypatch.setattr(
        "tools.file_tools._resolve_base_dir",
        lambda _task_id="default", **_kwargs: Path.home(),
    )

    result = json.loads(search_tool("needle", task_id=task_id, platform="feishu"))

    assert result["error"].startswith("BLOCKED: Feishu low-token search_files refused")
    assert str(Path.home().resolve()) in result["error"]
    assert "narrow path" in result["recommendation"]


def test_feishu_search_files_blocks_explicit_home_root():
    task_id = "feishu-search-explicit-home-root"
    _clear_task(task_id)

    result = json.loads(search_tool("needle", path=str(Path.home()), task_id=task_id, platform="feishu"))

    assert result["error"].startswith("BLOCKED: Feishu low-token search_files refused")


def test_feishu_deep_search_files_also_blocks_explicit_home_root():
    task_id = "feishu-deep-search-explicit-home-root"
    _clear_task(task_id)

    result = json.loads(search_tool("needle", path=str(Path.home()), task_id=task_id, platform="feishu_deep"))

    assert result["error"].startswith("BLOCKED: Feishu low-token search_files refused")


def test_feishu_search_files_allows_explicit_narrow_path(tmp_path):
    task_id = "feishu-search-narrow-path"
    _clear_task(task_id)
    fake = FakeFileOps()

    with patch("tools.file_tools._get_file_ops", return_value=fake):
        raw = search_tool("needle", path=str(tmp_path), task_id=task_id, platform="feishu")
        result = json.loads(raw.split("\n\n[Hint:", 1)[0])

    assert "error" not in result
    assert fake.search_calls[0]["path"] == str(tmp_path)


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
