"""Concurrency admission tests for expensive filename walks."""

from concurrent.futures import ThreadPoolExecutor
import threading
import types

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import SearchResult, ShellFileOperations
from tools.file_operations_search import (
    _ACTIVE_FILENAME_SEARCH_ROOTS,
    _FILENAME_SEARCH_ADMISSION,
    _normalized_filename_search_root,
)
from tools.interrupt import set_interrupt


class RemoteEnvironment:
    is_local = False
    cwd = "/workspace"

    def execute(self, command, **kwargs):
        raise AssertionError(f"unexpected backend command: {command}")


def _operations(env, scan):
    operations = ShellFileOperations(env)
    operations._resolve_command = lambda command: "/usr/bin/rg" if command == "rg" else None
    operations._search_files_rg = types.MethodType(scan, operations)
    return operations


def test_same_backend_class_and_root_serialize_five_filename_walks():
    entered = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0
    completed = 0

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        nonlocal active, maximum_active, completed
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            entered.set()
        assert release.wait(5)
        with counter_lock:
            active -= 1
            completed += 1
        return SearchResult(files=[str(path)], total_count=1)

    operations = [_operations(RemoteEnvironment(), scan) for _ in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [
            pool.submit(operation._search_files, "*.py", "/repo", 50, 0)
            for operation in operations
        ]
        assert entered.wait(5)
        release.set()
        results = [future.result(timeout=5) for future in futures]

    assert all(result.error is None for result in results)
    assert completed == 5
    assert maximum_active == 1


def test_different_roots_can_enter_filename_walks_together():
    both_entered = threading.Barrier(2)

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        both_entered.wait(5)
        return SearchResult(files=[str(path)], total_count=1)

    first = _operations(RemoteEnvironment(), scan)
    second = _operations(RemoteEnvironment(), scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first._search_files, "*.py", "/one", 50, 0),
            pool.submit(second._search_files, "*.py", "/two", 50, 0),
        ]
        assert [future.result(timeout=5).error for future in futures] == [None, None]


def test_different_backend_classes_can_walk_the_same_root_together():
    class OtherRemoteEnvironment(RemoteEnvironment):
        pass

    both_entered = threading.Barrier(2)

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        both_entered.wait(5)
        return SearchResult(files=[str(path)], total_count=1)

    first = _operations(RemoteEnvironment(), scan)
    second = _operations(OtherRemoteEnvironment(), scan)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(first._search_files, "*.py", "/same", 50, 0),
            pool.submit(second._search_files, "*.py", "/same", 50, 0),
        ]
        assert [future.result(timeout=5).error for future in futures] == [None, None]


def test_overlapping_multi_root_sets_are_claimed_atomically(monkeypatch):
    first_entered = threading.Event()
    release_first = threading.Event()
    second_waiting = threading.Event()
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if path == ["/a", "/b"]:
                first_entered.set()
        if path == ["/a", "/b"]:
            assert release_first.wait(5)
        with lock:
            active -= 1
        return SearchResult(files=[str(path)], total_count=1)

    first = _operations(RemoteEnvironment(), scan)
    second = _operations(RemoteEnvironment(), scan)
    original_wait = _FILENAME_SEARCH_ADMISSION.wait

    def observed_wait(timeout=None):
        second_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(_FILENAME_SEARCH_ADMISSION, "wait", observed_wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(first._search_files, "*.py", ["/a", "/b"], 50, 0)
        assert first_entered.wait(5)
        second_future = pool.submit(second._search_files, "*.py", ["/b", "/c"], 50, 0)
        assert second_waiting.wait(5)
        release_first.set()
        assert first_future.result(timeout=5).error is None
        assert second_future.result(timeout=5).error is None

    assert maximum_active == 1


def test_interrupted_waiter_returns_without_dispatch_or_late_dispatch(monkeypatch):
    holder_entered = threading.Event()
    release_holder = threading.Event()
    waiter_waiting = threading.Event()
    waiter_tid = []
    dispatches = []

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        dispatches.append(threading.get_ident())
        holder_entered.set()
        assert release_holder.wait(5)
        return SearchResult(files=[str(path)], total_count=1)

    holder = _operations(RemoteEnvironment(), scan)
    waiter = _operations(RemoteEnvironment(), scan)

    original_wait = _FILENAME_SEARCH_ADMISSION.wait

    def observed_wait(timeout=None):
        waiter_waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(_FILENAME_SEARCH_ADMISSION, "wait", observed_wait)

    def run_waiter():
        waiter_tid.append(threading.get_ident())
        return waiter._search_files("*.py", "/repo", 50, 0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder_future = pool.submit(holder._search_files, "*.py", "/repo", 50, 0)
        assert holder_entered.wait(5)
        waiter_future = pool.submit(run_waiter)
        assert waiter_waiting.wait(5)
        set_interrupt(True, waiter_tid[0])
        try:
            interrupted = waiter_future.result(timeout=5)
            assert "interrupted" in (interrupted.error or "").lower()
            assert len(dispatches) == 1
            release_holder.set()
            assert holder_future.result(timeout=5).error is None
            assert len(dispatches) == 1
        finally:
            set_interrupt(False, waiter_tid[0])
            release_holder.set()


def test_interrupt_published_after_final_sample_prevents_filename_dispatch(monkeypatch):
    sampled_clear = threading.Event()
    resume_acquire = threading.Event()
    worker_tid = []
    dispatches = []

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        dispatches.append(threading.get_ident())
        return SearchResult(files=[str(path)], total_count=1)

    operations = _operations(RemoteEnvironment(), scan)
    original_is_interrupted = __import__(
        "tools.interrupt", fromlist=["is_interrupted"]
    ).is_interrupted

    def pause_after_clear_sample():
        interrupted = original_is_interrupted()
        if not interrupted and threading.get_ident() == worker_tid[0]:
            sampled_clear.set()
            assert resume_acquire.wait(5)
        return interrupted

    monkeypatch.setattr(
        "tools.file_operations_search.tool_interrupt.is_interrupted",
        pause_after_clear_sample,
    )

    def run_search():
        worker_tid.append(threading.get_ident())
        return operations._search_files("*.py", "/repo", 50, 0)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_search)
        assert sampled_clear.wait(5)
        set_interrupt(True, worker_tid[0])
        resume_acquire.set()
        try:
            result = future.result(timeout=5)
        finally:
            set_interrupt(False, worker_tid[0])
            resume_acquire.set()

    assert "interrupted" in (result.error or "").lower()
    assert dispatches == []
    assert _ACTIVE_FILENAME_SEARCH_ROOTS == set()


def test_empty_filename_roots_are_rejected_before_engine_resolution():
    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        raise AssertionError("filename engine dispatched")

    operations = _operations(RemoteEnvironment(), scan)
    operations._resolve_command = lambda command: (_ for _ in ()).throw(
        AssertionError(f"engine resolution attempted: {command}")
    )

    result = operations._search_files("*.py", [], 50, 0)

    assert "at least one search root" in (result.error or "").lower()
    assert _ACTIVE_FILENAME_SEARCH_ROOTS == set()


@pytest.mark.parametrize("raised", [Exception, KeyboardInterrupt, SystemExit, BaseException])
def test_admission_releases_after_every_base_exception_path(raised):
    attempts = 0

    def scan(self, pattern, path, limit, offset, order, rg_executable=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise raised("engine failed")
        return SearchResult(files=[str(path)], total_count=1)

    operations = _operations(RemoteEnvironment(), scan)
    with pytest.raises(raised, match="engine failed"):
        operations._search_files("*.py", "/repo", 50, 0)

    result = operations._search_files("*.py", "/repo", 50, 0)
    assert result.error is None
    assert attempts == 2
    assert _ACTIVE_FILENAME_SEARCH_ROOTS == set()


def test_remote_roots_are_normalized_lexically_against_backend_cwd(monkeypatch):
    env = RemoteEnvironment()
    monkeypatch.setattr(
        "tools.file_operations.os.path.abspath",
        lambda path: (_ for _ in ()).throw(AssertionError("controller resolution used")),
    )

    relative = _normalized_filename_search_root(env, "repo/../repo", "/controller")
    absolute = _normalized_filename_search_root(env, "/workspace/repo", "/controller")

    assert relative == "/workspace/repo"
    assert absolute == relative


@pytest.mark.windows_only
def test_windows_local_root_spellings_share_one_normalized_key():
    env = LocalEnvironment.__new__(LocalEnvironment)
    env.cwd = "C:/Repo"

    native = _normalized_filename_search_root(env, r"C:\Repo\src\..", "C:/ignored")
    msys = _normalized_filename_search_root(env, "/c/Repo", "C:/ignored")

    assert native == msys
