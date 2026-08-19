"""Bounded read_file behavior for cloud-backed or wedged filesystems."""

import json
import threading
import time
from unittest.mock import MagicMock

from tools import file_tools
from tools.file_operations import ReadResult


def test_read_file_timeout_returns_actionable_structured_error(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def _wedged_read(*_args, **_kwargs):
        started.set()
        release.wait()
        return "late result"

    monkeypatch.setattr(file_tools, "_read_file_tool_impl", _wedged_read)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)

    before = time.monotonic()
    try:
        payload = json.loads(file_tools.read_file_tool("/cloud/project/file.md"))
    finally:
        release.set()

    assert started.is_set()
    assert time.monotonic() - before < 1.0
    assert payload["error_type"] == "tool_timeout"
    assert payload["timeout_seconds"] == 0.05
    assert payload["path"] == "/cloud/project/file.md"
    assert "local clone/direct source" in payload["error"]
    assert "instead of retrying the same read" in payload["error"]


def test_read_file_timeout_can_be_disabled(monkeypatch):
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: None)
    monkeypatch.setattr(
        file_tools,
        "_read_file_tool_impl",
        lambda path, offset, limit, task_id: f"{path}|{offset}|{limit}|{task_id}",
    )

    assert file_tools.read_file_tool("notes.md", 3, 7, "task-1") == (
        "notes.md|3|7|task-1"
    )


def test_late_read_does_not_publish_bookkeeping(tmp_path, monkeypatch):
    path = tmp_path / "cloud.md"
    path.write_text("local placeholder")
    release = threading.Event()
    read_returned = threading.Event()
    record_read = MagicMock()

    class _WedgedOps:
        def read_file(self, *_args, **_kwargs):
            release.wait()
            read_returned.set()
            return ReadResult(content="late content", total_lines=1, file_size=12)

    task_id = "late-read-task"
    with file_tools._read_tracker_lock:
        file_tools._read_tracker.pop(task_id, None)

    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _WedgedOps())
    monkeypatch.setattr(file_tools, "_file_ops_uses_host_paths", lambda _ops: True)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(file_tools.file_state, "record_read", record_read)

    try:
        payload = json.loads(file_tools.read_file_tool(str(path), task_id=task_id))
        assert payload["error_type"] == "tool_timeout"
        release.set()
        assert read_returned.wait(timeout=1)
        time.sleep(0.05)

        with file_tools._read_tracker_lock:
            task_data = file_tools._read_tracker.get(task_id)
            assert task_data is None or (
                task_data["last_key"] is None
                and task_data["consecutive"] == 0
                and task_data["dedup"] == {}
                and task_data["read_timestamps"] == {}
            )
        record_read.assert_not_called()
    finally:
        release.set()
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_timeout_wins_before_bookkeeping_commit(tmp_path, monkeypatch):
    path = tmp_path / "cloud.md"
    path.write_text("local placeholder")
    commit_attempted = threading.Event()
    release_commit = threading.Event()
    worker_finished = threading.Event()
    call_finished = threading.Event()
    payload_holder = {}
    record_read = MagicMock()
    real_impl = file_tools._read_file_tool_impl
    real_try_begin_commit = file_tools._ReadAbandonState.try_begin_commit

    def _observed_impl(*args, **kwargs):
        try:
            return real_impl(*args, **kwargs)
        finally:
            worker_finished.set()

    def _blocked_try_begin_commit(self):
        commit_attempted.set()
        assert release_commit.wait(timeout=1)
        return real_try_begin_commit(self)

    class _FastOps:
        def read_file(self, *_args, **_kwargs):
            return ReadResult(content="late content", total_lines=1, file_size=12)

    def _call_read_file():
        payload_holder["payload"] = json.loads(
            file_tools.read_file_tool(str(path), task_id=task_id)
        )
        call_finished.set()

    task_id = "bookkeeping-race-task"
    with file_tools._read_tracker_lock:
        file_tools._read_tracker.pop(task_id, None)

    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _FastOps())
    monkeypatch.setattr(file_tools, "_file_ops_uses_host_paths", lambda _ops: True)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(file_tools, "_read_file_tool_impl", _observed_impl)
    monkeypatch.setattr(
        file_tools._ReadAbandonState,
        "try_begin_commit",
        _blocked_try_begin_commit,
    )
    monkeypatch.setattr(file_tools.file_state, "record_read", record_read)

    caller = threading.Thread(target=_call_read_file)
    caller.start()
    try:
        assert commit_attempted.wait(timeout=1)
        assert call_finished.wait(timeout=1)
        assert payload_holder["payload"]["error_type"] == "tool_timeout"
        release_commit.set()
        caller.join(timeout=1)
        assert worker_finished.wait(timeout=1)

        with file_tools._read_tracker_lock:
            task_data = file_tools._read_tracker.get(task_id)
            assert task_data is None or (
                task_data["last_key"] is None
                and task_data["consecutive"] == 0
                and task_data["dedup"] == {}
                and task_data["read_timestamps"] == {}
            )
        record_read.assert_not_called()
    finally:
        release_commit.set()
        caller.join(timeout=1)
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_commit_winner_returns_real_result_instead_of_timeout(tmp_path, monkeypatch):
    path = tmp_path / "cloud.md"
    path.write_text("local placeholder")
    commit_claimed = threading.Event()
    release_commit = threading.Event()
    call_finished = threading.Event()
    payload_holder = {}
    record_read = MagicMock()
    real_try_begin_commit = file_tools._ReadAbandonState.try_begin_commit

    def _claimed_then_blocked(self):
        claimed = real_try_begin_commit(self)
        assert claimed
        commit_claimed.set()
        assert release_commit.wait(timeout=1)
        return claimed

    class _FastOps:
        def read_file(self, *_args, **_kwargs):
            return ReadResult(content="on-time content", total_lines=1, file_size=15)

    task_id = "bookkeeping-commit-winner-task"
    with file_tools._read_tracker_lock:
        file_tools._read_tracker.pop(task_id, None)

    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _FastOps())
    monkeypatch.setattr(file_tools, "_file_ops_uses_host_paths", lambda _ops: True)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(
        file_tools._ReadAbandonState,
        "try_begin_commit",
        _claimed_then_blocked,
    )
    monkeypatch.setattr(file_tools.file_state, "record_read", record_read)

    def _call_read_file():
        payload_holder["payload"] = json.loads(
            file_tools.read_file_tool(str(path), task_id=task_id)
        )
        call_finished.set()

    caller = threading.Thread(target=_call_read_file)
    caller.start()
    try:
        assert commit_claimed.wait(timeout=1)
        time.sleep(0.1)
        assert not call_finished.is_set()
        release_commit.set()
        assert call_finished.wait(timeout=1)
        caller.join(timeout=1)

        payload = payload_holder["payload"]
        assert payload["content"] == "on-time content"
        assert "error_type" not in payload
        with file_tools._read_tracker_lock:
            task_data = file_tools._read_tracker[task_id]
            assert task_data["last_key"] == ("read", str(path), 1, 2000)
            assert task_data["consecutive"] == 1
        record_read.assert_called_once()
        assert record_read.call_args.kwargs["mtime"] == path.stat().st_mtime
    finally:
        release_commit.set()
        caller.join(timeout=1)
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_unavailable_mtime_does_not_restat_after_commit(tmp_path, monkeypatch):
    path = tmp_path / "cloud.md"
    path.write_text("local placeholder")
    getmtime = MagicMock(side_effect=OSError("metadata unavailable"))
    record_read = MagicMock()

    class _FastOps:
        def read_file(self, *_args, **_kwargs):
            return ReadResult(content="content", total_lines=1, file_size=7)

    task_id = "bookkeeping-no-mtime-task"
    with file_tools._read_tracker_lock:
        file_tools._read_tracker.pop(task_id, None)

    monkeypatch.setattr(file_tools, "_get_file_ops", lambda _task_id: _FastOps())
    monkeypatch.setattr(file_tools, "_file_ops_uses_host_paths", lambda _ops: True)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(file_tools.os.path, "getmtime", getmtime)
    monkeypatch.setattr(file_tools.file_state, "record_read", record_read)

    try:
        payload = json.loads(file_tools.read_file_tool(str(path), task_id=task_id))
        assert payload["content"] == "content"
        assert "error_type" not in payload
        getmtime.assert_called_once_with(str(path))
        record_read.assert_not_called()
    finally:
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_late_dedup_hit_does_not_publish_after_timeout(tmp_path, monkeypatch):
    path = tmp_path / "cloud.md"
    path.write_text("cached content")
    release_stat = threading.Event()
    stat_started = threading.Event()
    worker_finished = threading.Event()
    real_impl = file_tools._read_file_tool_impl
    cached_mtime = path.stat().st_mtime
    task_id = "late-dedup-hit-task"
    dedup_key = (str(path), 1, 2000)
    executor = file_tools.DaemonThreadPoolExecutor(max_workers=1)
    admission = threading.BoundedSemaphore(1)

    with file_tools._read_tracker_lock:
        file_tools._read_tracker[task_id] = {
            "last_key": None,
            "consecutive": 0,
            "read_history": set(),
            "dedup": {dedup_key: cached_mtime},
            "dedup_hits": {},
            "read_timestamps": {},
            "not_found": {},
        }

    def _wedged_getmtime(_path):
        stat_started.set()
        assert release_stat.wait(timeout=1)
        return cached_mtime

    def _observed_impl(*args, **kwargs):
        try:
            return real_impl(*args, **kwargs)
        finally:
            worker_finished.set()

    real_submit = executor.submit

    def _submit_after_worker_reaches_stat(*args, **kwargs):
        future = real_submit(*args, **kwargs)
        assert stat_started.wait(timeout=1)
        return future

    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(file_tools, "_read_file_tool_impl", _observed_impl)
    monkeypatch.setattr(file_tools, "_read_file_executor", executor)
    monkeypatch.setattr(file_tools, "_read_file_admission", admission)
    monkeypatch.setattr(executor, "submit", _submit_after_worker_reaches_stat)
    monkeypatch.setattr(file_tools.os.path, "getmtime", _wedged_getmtime)

    try:
        payload = json.loads(file_tools.read_file_tool(str(path), task_id=task_id))
        assert stat_started.is_set()
        assert payload["error_type"] == "tool_timeout"

        release_stat.set()
        assert worker_finished.wait(timeout=1)
        with file_tools._read_tracker_lock:
            assert file_tools._read_tracker[task_id]["dedup_hits"] == {}
    finally:
        release_stat.set()
        executor.shutdown(wait=True, cancel_futures=True)
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_late_negative_cache_existence_check_does_not_publish_after_timeout(
    tmp_path, monkeypatch
):
    path = tmp_path / "created-later.md"
    path.write_text("real content")
    release_exists = threading.Event()
    exists_started = threading.Event()
    worker_finished = threading.Event()
    real_impl = file_tools._read_file_tool_impl
    task_id = "late-negative-cache-task"
    cache_key = ("read", str(path))

    file_tools._record_not_found(
        "read", str(path), task_id, '{"error":"File not found: cached"}'
    )

    def _wedged_exists(_path):
        exists_started.set()
        assert release_exists.wait(timeout=1)
        return True

    def _observed_impl(*args, **kwargs):
        try:
            return real_impl(*args, **kwargs)
        finally:
            worker_finished.set()

    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.05)
    monkeypatch.setattr(file_tools, "_read_file_tool_impl", _observed_impl)
    monkeypatch.setattr(file_tools.os.path, "exists", _wedged_exists)

    try:
        payload = json.loads(file_tools.read_file_tool(str(path), task_id=task_id))
        assert exists_started.is_set()
        assert payload["error_type"] == "tool_timeout"

        release_exists.set()
        assert worker_finished.wait(timeout=1)
        with file_tools._read_tracker_lock:
            assert cache_key in file_tools._read_tracker[task_id]["not_found"]
    finally:
        release_exists.set()
        with file_tools._read_tracker_lock:
            file_tools._read_tracker.pop(task_id, None)


def test_timed_out_reads_use_a_process_wide_bounded_worker_pool(monkeypatch):
    release = threading.Event()
    started_lock = threading.Lock()
    started = 0
    executor = file_tools.DaemonThreadPoolExecutor(max_workers=4)
    admission = threading.BoundedSemaphore(4)

    def _wedged_read(*_args, **_kwargs):
        nonlocal started
        with started_lock:
            started += 1
        release.wait()
        return "late result"

    monkeypatch.setattr(file_tools, "_read_file_tool_impl", _wedged_read)
    monkeypatch.setattr(file_tools, "_resolve_read_file_timeout", lambda: 0.1)
    monkeypatch.setattr(file_tools, "_read_file_executor", executor)
    monkeypatch.setattr(file_tools, "_read_file_admission", admission)

    try:
        payloads = [
            json.loads(file_tools.read_file_tool(f"/cloud/stuck-{index}.md"))
            for index in range(6)
        ]
        assert all(payload["error_type"] == "tool_timeout" for payload in payloads)
        with started_lock:
            assert started == 4
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)
