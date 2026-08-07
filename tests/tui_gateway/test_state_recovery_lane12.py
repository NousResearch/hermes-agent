"""Crash/restart recovery contracts found by the lane-12 E2E campaign."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from tui_gateway import host_supervisor as host_mod
from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor
from tui_gateway.turn_marker import read_turn_marker, record_turn_start


def _marker_path(home: Path) -> Path:
    return home / "desktop" / "interrupted_turns.json"


def test_malformed_sibling_marker_does_not_block_new_turn_record(tmp_path):
    path = _marker_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "damaged": {
                    "attempts": 0,
                    "prompt": "old",
                    "started_at": "not-a-timestamp",
                }
            }
        ),
        encoding="utf-8",
    )

    record_turn_start(tmp_path, "current", "finish the current task")

    assert read_turn_marker(tmp_path, "current") is not None
    assert read_turn_marker(tmp_path, "damaged") is None


@pytest.mark.parametrize("started_at", [math.nan, math.inf, -math.inf])
def test_nonfinite_marker_timestamp_is_never_recovered(tmp_path, started_at):
    path = _marker_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "session": {
                    "attempts": 0,
                    "prompt": "stale operation",
                    "started_at": started_at,
                }
            }
        ),
        encoding="utf-8",
    )

    assert read_turn_marker(tmp_path, "session") is None


def test_far_future_marker_is_not_auto_continued(tmp_path, monkeypatch):
    path = _marker_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "session": {
                    "attempts": 0,
                    "prompt": "repeat an old side effect",
                    "started_at": time.time() + 24 * 3600,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    started: list[str] = []

    class RecordingThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            started.append("thread")

    monkeypatch.setattr(server.threading, "Thread", RecordingThread)
    session = {
        "profile_home": str(tmp_path),
        "history_lock": threading.Lock(),
        "running": False,
    }

    assert server._maybe_schedule_auto_continue("sid", session, "session") is None
    assert started == []


def test_marker_entry_cap_includes_the_new_turn(tmp_path, monkeypatch):
    now = time.time()
    path = _marker_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                f"old-{i}": {
                    "attempts": 0,
                    "prompt": f"prompt {i}",
                    "started_at": now - i,
                }
                for i in range(32)
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("tui_gateway.turn_marker.time.time", lambda: now)

    record_turn_start(tmp_path, "new", "new prompt")

    entries = json.loads(path.read_text(encoding="utf-8"))
    assert len(entries) == 32
    assert "new" in entries


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX durability")
def test_marker_replace_fsyncs_payload_and_parent_directory(tmp_path, monkeypatch):
    from tui_gateway import turn_marker

    synced_modes: list[int] = []
    real_fsync = turn_marker.os.fsync

    def tracking_fsync(fd: int):
        synced_modes.append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(turn_marker.os, "fsync", tracking_fsync)

    record_turn_start(tmp_path, "session", "durable prompt")

    assert any(stat.S_ISREG(mode) for mode in synced_modes)
    assert any(stat.S_ISDIR(mode) for mode in synced_modes)


def _concurrent_marker_writer(home: str, key: str, barrier) -> None:
    """Force two processes through the same pre-write snapshot on old code."""
    from tui_gateway import turn_marker

    original_load = turn_marker._load

    def synchronized_load(path):
        value = original_load(path)
        try:
            barrier.wait(timeout=0.25)
        except threading.BrokenBarrierError:
            pass
        return value

    turn_marker._load = synchronized_load
    turn_marker.record_turn_start(home, key, f"prompt {key}")


@pytest.mark.skipif(os.name == "nt", reason="fork-only deterministic race harness")
def test_marker_updates_are_serialized_across_processes(tmp_path):
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    workers = [
        ctx.Process(target=_concurrent_marker_writer, args=(str(tmp_path), key, barrier))
        for key in ("one", "two")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=5)
        assert worker.exitcode == 0

    assert read_turn_marker(tmp_path, "one") is not None
    assert read_turn_marker(tmp_path, "two") is not None


def _write_fake_host(path: Path, *, build_sha: str = "expected", control: str = "ack") -> None:
    path.write_text(
        f"""
import json, os, sys, time
print(json.dumps({{'type':'hello','host_pid':os.getpid(),'boot_id':'boot','build_sha':{build_sha!r},'hermes_home':os.environ.get('HERMES_HOME','')}}), flush=True)
for raw in sys.stdin:
    frame=json.loads(raw)
    if frame.get('type') == 'shutdown':
        print(json.dumps({{'type':'shutdown.ack','request_id':frame.get('request_id')}}), flush=True)
        break
    if frame.get('type') == 'control':
        if {control!r} == 'crash':
            os._exit(9)
        if {control!r} == 'hang':
            time.sleep(60)
        print(json.dumps({{'type':'control.ack','request_id':frame.get('request_id')}}), flush=True)
""".strip(),
        encoding="utf-8",
    )


def _wait_dead(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not host_mod._pid_alive(pid):
            return True
        time.sleep(0.02)
    return not host_mod._pid_alive(pid)


def test_invalid_hello_cleans_up_spawned_child(tmp_path):
    script = tmp_path / "wrong_host.py"
    _write_fake_host(script, build_sha="wrong")
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json",
        argv=[sys.executable, str(script)],
        expected_build_sha="expected",
        autostart=False,
    )
    spawned: list[int] = []
    original_validate = supervisor._validate_hello

    def validate():
        spawned.append(supervisor.pid)
        original_validate()

    supervisor._validate_hello = validate

    with pytest.raises(RuntimeError, match="build mismatch"):
        supervisor.start()

    assert spawned and spawned[0] > 0
    assert _wait_dead(spawned[0]), "hello-rejected child survived startup failure"


def test_registry_persistence_failure_cleans_up_spawned_child(tmp_path, monkeypatch):
    script = tmp_path / "host.py"
    _write_fake_host(script)
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json",
        argv=[sys.executable, str(script)],
        expected_build_sha="expected",
        autostart=False,
    )
    spawned: list[int] = []

    def fail_registry():
        spawned.append(supervisor.pid)
        raise OSError("disk full")

    monkeypatch.setattr(supervisor, "_persist_registry", fail_registry)

    with pytest.raises(OSError, match="disk full"):
        supervisor.start()

    assert spawned and spawned[0] > 0
    assert _wait_dead(spawned[0]), "unregistered child survived registry failure"


def test_compute_host_crash_resolves_pending_control_waiter(tmp_path):
    script = tmp_path / "crash_host.py"
    _write_fake_host(script, control="crash")
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json",
        argv=[sys.executable, str(script)],
        expected_build_sha="expected",
        respawn_max=0,
        autostart=False,
    )
    result: list[object] = []
    supervisor.start()

    def call_control():
        try:
            result.append(
                supervisor.control(
                    "sid", route_name="session.save", timeout=5.0
                )
            )
        except Exception as exc:  # old behavior strands the waiter to timeout
            result.append(exc)

    worker = threading.Thread(target=call_control)
    worker.start()
    worker.join(timeout=1.5)
    try:
        assert not worker.is_alive(), "control waiter remained stranded after crash"
        assert isinstance(result[0], dict)
        assert result[0]["type"] == "control.error"
        assert result[0]["reason"] == "crash"
    finally:
        supervisor.shutdown()


def test_control_send_failure_does_not_leak_waiter(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json", autostart=False
    )
    monkeypatch.setattr(supervisor, "start", lambda: None)
    monkeypatch.setattr(
        supervisor,
        "_send_frame",
        lambda _frame: (_ for _ in ()).throw(BrokenPipeError("closed")),
    )

    with pytest.raises(BrokenPipeError):
        supervisor.control("sid", route_name="session.save", timeout=0.1)

    assert supervisor._pending_controls == {}


def test_stale_exit_cannot_remove_replacement_registry(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"host_pid": 222}), encoding="utf-8")
    supervisor = HostSupervisor(registry_path=registry, autostart=False)
    old_proc = type("OldProc", (), {"pid": 111, "wait": lambda self: 9})()
    supervisor._proc = old_proc
    monkeypatch.setattr(supervisor, "_maybe_respawn_after_crash", lambda: None)

    supervisor._wait_for_exit(old_proc)

    assert json.loads(registry.read_text(encoding="utf-8"))["host_pid"] == 222


def test_failed_orphan_termination_blocks_duplicate_start(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"host_pid": 111}), encoding="utf-8")
    supervisor = HostSupervisor(registry_path=registry, autostart=False)
    monkeypatch.setattr(host_mod, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(supervisor, "_pid_matches_compute_host", lambda _pid: True)
    monkeypatch.setattr(supervisor, "_terminate_pid", lambda *_a, **_kw: False)
    spawned: list[str] = []
    monkeypatch.setattr(
        supervisor, "_spawn_locked", lambda *, reason: spawned.append(reason)
    )

    with pytest.raises(RuntimeError, match="could not terminate"):
        supervisor.start()

    assert spawned == []
    assert registry.exists()


def test_shutdown_resolves_pending_control_waiter(tmp_path, monkeypatch):
    script = tmp_path / "hang_host.py"
    _write_fake_host(script, control="hang")
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json",
        argv=[sys.executable, str(script)],
        expected_build_sha="expected",
        autostart=False,
    )
    monkeypatch.setattr(host_mod, "_SHUTDOWN_TIMEOUT_SECS", 0.1)
    result: list[object] = []
    supervisor.start()

    def call_control():
        try:
            result.append(
                supervisor.control("sid", route_name="session.save", timeout=5.0)
            )
        except Exception as exc:
            result.append(exc)

    worker = threading.Thread(target=call_control)
    worker.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not supervisor._pending_controls:
        time.sleep(0.01)
    supervisor.shutdown()
    worker.join(timeout=1)

    assert not worker.is_alive(), "control waiter remained stranded during shutdown"
    assert isinstance(result[0], dict)
    assert result[0]["type"] == "control.error"
    assert result[0]["reason"] == "shutdown"


def test_transient_respawn_failure_is_retried(tmp_path, monkeypatch):
    supervisor = HostSupervisor(
        registry_path=tmp_path / "registry.json", respawn_max=3, autostart=False
    )
    calls: list[str] = []

    def spawn(*, reason: str):
        calls.append(reason)
        if len(calls) == 1:
            raise OSError("temporary fork failure")
        supervisor._proc = object()  # only proves the retry path was invoked

    class InlineThread:
        def __init__(self, target=None, **_kwargs):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(supervisor, "_spawn_locked", spawn)
    monkeypatch.setattr(host_mod, "_Thread", InlineThread)
    monkeypatch.setattr(host_mod.time, "sleep", lambda _delay: None)

    supervisor._maybe_respawn_after_crash()

    assert calls == ["crash", "crash"]
