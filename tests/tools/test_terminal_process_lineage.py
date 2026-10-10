"""Local launch evidence stays unresolved, including after direct-child exit."""
import sqlite3
import sys

import pytest

from agent.chat_execution_context import bind_chat_execution
from gateway.chat_execution_recovery import ChatExecutionStore
from tools import process_registry as module
from tools.process_registry import ProcessRegistry

pytestmark = pytest.mark.platforms("linux")


def started(home):
    store = ChatExecutionStore(home)
    assert store.start("key", 1, "physical-session")
    assert store.worker_started("key", 1)
    return store


def isolated_registry(monkeypatch):
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_scope_argv", lambda *a: [sys.executable, "-c", "import signal; signal.pause()"])
    return registry


def cleanup(registry, processes):
    registry.kill_all()
    for proc in processes:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
        if proc.stdout:
            proc.stdout.close()
    for session in list(registry._running.values()) + list(registry._finished.values()):
        if session._reader_thread:
            session._reader_thread.join(5)
            assert not session._reader_thread.is_alive()


@pytest.mark.parametrize("phase", ["normal", "reserve", "start", "spawn", "lineage", "missing_fingerprint"])
def test_actual_pipe_producer_persistence_boundaries(tmp_path, monkeypatch, phase):
    store = started(tmp_path)
    registry = isolated_registry(monkeypatch)
    processes, launches, reaps = [], [], []
    real_popen = module.subprocess.Popen
    def popen(*args, **kwargs):
        row = ChatExecutionStore(tmp_path).inspect_resources("key")[0]
        assert (row["kind"], row["state"]) == ("terminal_local_process", "started")
        launches.append("launch")
        if phase == "spawn":
            raise OSError("injected launch failure")
        proc = real_popen(*args, **kwargs)
        processes.append(proc)
        return proc
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    methods = {"reserve": "reserve_concurrent_tool_resource", "start": "start_concurrent_tool_resource",
               "lineage": "record_terminal_process_lineage"}
    def broken(*a, **kw):
        raise sqlite3.OperationalError("injected durable failure")
    if phase in methods:
        monkeypatch.setattr(ChatExecutionStore, methods[phase], broken)
    if phase == "missing_fingerprint":
        monkeypatch.setattr(registry, "_safe_host_start_time", lambda pid: None)
    real_reap = registry._reap_untracked
    def reap(session, proc):
        reaps.append(proc.pid)
        real_reap(session, proc)
    monkeypatch.setattr(registry, "_reap_untracked", reap)
    errors = {"reserve": sqlite3.OperationalError, "start": sqlite3.OperationalError,
              "spawn": OSError, "lineage": sqlite3.OperationalError, "missing_fingerprint": RuntimeError}
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            if phase in errors:
                with pytest.raises(errors[phase]):
                    registry.spawn_local("fixture", cwd=str(tmp_path), use_pty=False)
            else:
                session = registry.spawn_local("fixture", cwd=str(tmp_path), use_pty=False)
                row = store.inspect_resources("key")[0]
                lineage = store.inspect_terminal_process_lineage("key", 1, row["resource_id"])
                assert lineage is not None
                assert lineage["pid"] == session.pid
                assert lineage["host_start_time"] == session.host_start_time
                assert lineage["session_id"] == "physical-session"
                assert lineage["process_session_id"] == session.id
                processes[0].terminate()
                processes[0].wait(timeout=5)
                assert session._reader_thread is not None
                session._reader_thread.join(5)
                assert not session._reader_thread.is_alive()
                assert session.exited
                assert store.inspect_resources("key")[0]["state"] == "started"
                assert not store.finish_concurrent_tool_resource(
                    "key", 1, row["resource_id"], "physical-session", "terminal_local_process")
    finally:
        cleanup(registry, processes)
    assert launches == ([] if phase in {"reserve", "start"} else ["launch"])
    assert len(reaps) == (1 if phase in {"lineage", "missing_fingerprint"} else 0)
    expected = {"reserve": [], "start": ["reserved"], "normal": ["started"],
                "spawn": ["started"], "lineage": ["started"], "missing_fingerprint": ["started"]}
    assert [row["state"] for row in store.inspect_resources("key")] == expected[phase]
    assert store.inspect("key", 1)["effects"] == "unknown"


@pytest.mark.parametrize("later", ["fence", "successor", "parent_finished"])
def test_stale_before_spawn_refuses_launch(tmp_path, monkeypatch, later):
    from tools import process_registry_chat_execution as adapter
    store = started(tmp_path)
    registry = isolated_registry(monkeypatch)
    called = []
    real_start = adapter.start_local_process
    def start(handle):
        changes = {"fence": lambda: store.begin_recovery("key", 1, "fence"),
                   "successor": lambda: store.reserve("key", 1, 1),
                   "parent_finished": lambda: store.worker_finished("key", 1)}
        assert changes[later]()
        real_start(handle)
    monkeypatch.setattr(adapter, "start_local_process", start)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **kw: called.append("launch"))
    try:
        with bind_chat_execution(tmp_path, "key", 1):
            with pytest.raises(RuntimeError, match="start refused"):
                registry.spawn_local("fixture", cwd=str(tmp_path))
    finally:
        cleanup(registry, [])
    assert called == []
    assert store.inspect_resources("key")[0]["state"] == "reserved"


@pytest.mark.parametrize("later", ["fence", "successor", "different_profile"])
def test_late_lineage_uses_original_identity_and_survives_handoff(tmp_path, monkeypatch, later):
    from tools import process_registry_chat_execution as adapter
    from agent.chat_execution_context import bind_chat_execution as bind
    home, other = tmp_path / "home", tmp_path / "other"
    store, other_store = started(home), started(other)
    registry = isolated_registry(monkeypatch)
    processes = []
    real_record = adapter.record_local_process
    real_popen = module.subprocess.Popen
    def popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        processes.append(proc)
        return proc
    def record(handle, process_id, pid, fingerprint):
        if later == "fence":
            assert store.worker_finished("key", 1)
            assert store.begin_recovery("key", 1, "fence")
        if later == "successor":
            assert store.reserve("key", 1, 1) == 2
        with bind(other, "key", 1):
            real_record(handle, process_id, pid, fingerprint)
    monkeypatch.setattr(adapter, "record_local_process", record)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    try:
        with bind(home, "key", 1):
            session = registry.spawn_local("fixture", cwd=str(tmp_path), owner_task_id="original-owner")
        session.owner_task_id = "new-owner"
        row = store.inspect_resources("key")[0]
        lineage = ChatExecutionStore(home).inspect_terminal_process_lineage("key", 1, row["resource_id"])
        assert lineage is not None
        assert lineage["pid"] == processes[0].pid
        assert lineage["session_id"] == "physical-session"
        assert other_store.inspect_resources("key") == []
    finally:
        cleanup(registry, processes)
    assert ChatExecutionStore(home).inspect_resources("key")[0]["state"] == "started"


@pytest.mark.parametrize("field,value", [("pid", 0), ("pid", True), ("host_start_time", None),
    ("host_start_time", 0), ("host_start_time", True), ("host_start_time", "123"),
    ("session_id", "foreign"), ("resource_id", "foreign")])
def test_lineage_rejects_unknown_or_foreign_identity(tmp_path, field, value):
    store = started(tmp_path)
    assert store.reserve_concurrent_tool_resource("key", 1, "resource", "terminal_local_process")
    assert store.start_concurrent_tool_resource("key", 1, "resource", "physical-session", "terminal_local_process")
    params = dict(session_key="key", generation=1, resource_id="resource", session_id="physical-session",
                  process_session_id="process", pid=123, host_start_time=456)
    params[field] = value
    assert not store.record_terminal_process_lineage(**params)
    assert store.inspect_terminal_process_lineage("key", 1, "resource") is None
    assert store.inspect_resources("key")[0]["state"] == "started"


def test_lineage_idempotence_refuses_recycled_fingerprint_or_rebinding(tmp_path):
    store = started(tmp_path)
    assert store.reserve_concurrent_tool_resource("key", 1, "resource", "terminal_local_process")
    assert store.start_concurrent_tool_resource("key", 1, "resource", "physical-session", "terminal_local_process")
    args = ("key", 1, "resource", "physical-session", "process", 123, 456)
    assert store.record_terminal_process_lineage(*args)
    assert store.record_terminal_process_lineage(*args)
    assert not store.record_terminal_process_lineage(*args[:-1], 789)
    assert not store.record_terminal_process_lineage(*args[:-2], 999, 456)
    original = store.inspect_terminal_process_lineage("key", 1, "resource")
    assert original is not None and original["pid"] == 123 and original["host_start_time"] == 456
    assert store.inspect_resources("key")[0]["state"] == "started"
