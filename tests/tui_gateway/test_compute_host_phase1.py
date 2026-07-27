import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tui_gateway.compute_host import ComputeHost, _default_workers
from tui_gateway.host_supervisor import (
    MUTATOR_ROUTE_TABLE,
    HostSupervisor,
    append_log_record,
)


def _json_lines(out: io.StringIO) -> list[dict]:
    frames = []
    for line in out.getvalue().splitlines():
        if line.strip():
            frames.append(json.loads(line))
    return frames


def _wait_for_frame(out: io.StringIO, predicate, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in _json_lines(out):
            if predicate(frame):
                return frame
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for frame; saw={_json_lines(out)}")


def test_compute_host_workers_inherit_tui_pool_env_or_8(monkeypatch):
    monkeypatch.delenv("HERMES_TUI_RPC_POOL_WORKERS", raising=False)
    monkeypatch.delenv("HERMES_COMPUTE_HOST_WORKERS", raising=False)
    assert _default_workers() == 8

    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "11")
    assert _default_workers() == 11

    # Dead-RC tombstone: malformed env falls back to 8, not the old except-branch 4.
    monkeypatch.setenv("HERMES_TUI_RPC_POOL_WORKERS", "not-an-int")
    assert _default_workers() == 8


def test_compute_host_frame_protocol_round_trip():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=2, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-1",
                "prompt": "hello",
                "delta_count": 3,
                "delay_s": 0,
            }
        )

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-1")
        assert end["history_version"] == 1
        frames = _json_lines(out)
        assert [f["type"] for f in frames if f.get("request_id") == "turn-1"] == [
            "turn.started",
            "delta",
            "delta",
            "delta",
            "turn.end",
        ]
    finally:
        host.close()


def test_compute_host_interrupt_control_is_not_queued_behind_turn():
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    try:
        host.handle_frame({"type": "session.seed", "sid": "alpha", "request_id": "seed", "history": []})
        host.handle_frame(
            {
                "type": "turn.start",
                "sid": "alpha",
                "request_id": "turn-slow",
                "prompt": "hello",
                "delta_count": 200,
                "delay_s": 0.01,
            }
        )
        _wait_for_frame(out, lambda f: f.get("type") == "delta" and f.get("request_id") == "turn-slow")

        host.handle_frame({"type": "interrupt", "sid": "alpha", "request_id": "stop-1"})
        ack = _wait_for_frame(out, lambda f: f.get("type") == "interrupt.ack" and f.get("request_id") == "stop-1")
        assert ack["applied"] is True

        end = _wait_for_frame(out, lambda f: f.get("type") == "turn.end" and f.get("request_id") == "turn-slow")
        assert end["interrupted"] is True
        typed = [f["type"] for f in _json_lines(out)]
        assert typed.index("interrupt.ack") < typed.index("turn.end")
    finally:
        host.close()


def test_compute_host_flushes_sessions_on_orphan_shutdown(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    session = {"session_key": "key"}
    calls: list[tuple[dict, str]] = []
    server._sessions["flush-sid"] = session
    monkeypatch.setattr(
        server,
        "_finalize_session",
        lambda sess, end_reason="tui_close": calls.append((sess, end_reason)),
    )
    try:
        host.flush_all_sessions(reason="orphan")
        assert calls == [(session, "compute_host_orphan")]
    finally:
        server._sessions.pop("flush-sid", None)
        host.close()


def test_compute_host_parent_guard_exits_when_parent_pid_changes(monkeypatch):
    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    host._parent_pid = 111
    monkeypatch.setattr(os, "getppid", lambda: 222)

    def _exit(code):
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", _exit)

    with pytest.raises(SystemExit) as exc_info:
        host._parent_guard_loop()

    assert exc_info.value.code == 0
    orphan = next(frame for frame in _json_lines(out) if frame.get("type") == "orphan")
    assert orphan["old_ppid"] == 111
    assert orphan["ppid"] == 222
    assert isinstance(orphan["host_ns"], int)


def test_mutator_route_table_matches_prd_inventory():
    assert MUTATOR_ROUTE_TABLE == {
        "prompt.submit": "turn-path",
        "session.interrupt": "turn-path",
        "reload.mcp": "run-concurrent",
        "session.save": "run-concurrent",
        "session.compress": "idle-gated",
        "prompt.submit.truncate": "idle-gated",
        "slash.model": "idle-gated",
        "slash.personality": "idle-gated",
        "slash.prompt": "idle-gated",
        "slash.compress": "idle-gated",
        "session.reset": "idle-gated",
        "session.history.reload": "idle-gated",
        "slash.retry": "idle-gated",
    }


def test_compute_host_compress_control_runs_identity_guard_in_host(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)

    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        context_compressor = None

    session = {
        "agent": _Agent(),
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }
    calls: dict[str, object] = {}

    def _compress(sess, focus_topic=None, **_kwargs):
        assert sess is session
        calls["compress_focus"] = focus_topic
        with sess["history_lock"]:
            sess["history"] = [{"role": "summary", "content": "compressed in host"}]
            sess["history_version"] = 3

    def _sync(sid, sess):
        assert sess is session
        calls["sync"] = sid
        sess["session_key"] = "after-key"

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _sync)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {
            "model": "host-model",
            "provider": "host-provider",
            "usage": {"total": 2},
        },
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-1",
                "route_name": "slash.compress",
                "command": "/compress focus",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == {"compress_focus": "focus", "sync": "sid"}
    assert ack["route_name"] == "slash.compress"
    assert ack["session_key"] == "after-key"
    assert ack["history_version"] == 3
    assert ack["message_count"] == 1
    assert ack["session_info"]["model"] == "host-model"


def test_compute_host_session_compress_returns_structured_result(monkeypatch):
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    session = {
        "agent": None,
        "session_key": "host-key",
        "history": [{"role": "user", "content": "preserved"}],
        "history_lock": threading.Lock(),
        "history_version": 3,
        "running": False,
    }
    calls: list[dict] = []

    def compress_handler(_rid, params):
        calls.append(params)
        return {
            "result": {
                "status": "aborted",
                "messages": [{"role": "user", "content": "preserved"}],
                "summary": {"aborted": True, "headline": "Compression aborted"},
            }
        }

    server._sessions["sid"] = session
    monkeypatch.setitem(server._methods, "session.compress", compress_handler)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {"model": "host-model"})

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-structured",
                "route_name": "session.compress",
                "command": "/compress auth",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda frame: frame.get("type") == "control.ack" and frame.get("request_id") == "compress-structured",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == [{"session_id": "sid", "focus_topic": "auth"}]
    assert ack["result"]["status"] == "aborted"
    assert ack["result"]["summary"]["aborted"] is True
    assert ack["session_key"] == "host-key"
    assert ack["history_version"] == 3
    assert ack["message_count"] == 1
    assert ack["session_info"] == {"model": "host-model"}


def test_append_log_record_single_write_lines(tmp_path):
    path = tmp_path / "agent.log"

    def writer(i: int) -> None:
        append_log_record(path, f"line-{i:03d}-" + ("x" * 2000))

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 32
    assert sorted(line.split("-", 2)[1] for line in lines) == [f"{i:03d}" for i in range(32)]
    assert all(line.endswith("x" * 2000) for line in lines)


def test_supervisor_startup_reconcile_pid_reuse_guard(tmp_path, monkeypatch):
    registry = tmp_path / "dashboard-compute-host.json"
    registry.write_text(json.dumps({"host_pid": os.getpid(), "boot_id": "stale"}), encoding="utf-8")

    killed: list[int] = []
    supervisor = HostSupervisor(registry_path=registry, argv=[sys.executable, "-c", ""], autostart=False)
    monkeypatch.setattr(supervisor, "_pid_matches_compute_host", lambda _pid: False)
    monkeypatch.setattr(supervisor, "_terminate_pid", lambda pid, **_kw: killed.append(pid))

    result = supervisor.reconcile_startup_orphan()

    assert result == "pid-reuse-ignored"
    assert killed == []
    assert not registry.exists()


def test_supervisor_crash_emits_turn_error_and_respawns(tmp_path):
    script = tmp_path / "fake_host.py"
    script.write_text(
        """
import json, os, sys
print(json.dumps({'type':'hello','host_pid':os.getpid(),'boot_id':'boot-1','build_sha':'test','hermes_home':os.environ.get('HERMES_HOME','')}), flush=True)
for raw in sys.stdin:
    frame=json.loads(raw)
    if frame.get('type') == 'shutdown':
        print(json.dumps({'type':'shutdown.ack','request_id':frame.get('request_id')}), flush=True)
        break
    if frame.get('type') == 'turn.start':
        print(json.dumps({'type':'turn.started','sid':frame.get('sid'),'request_id':frame.get('request_id')}), flush=True)
        sys.stdout.flush()
        os._exit(7)
""".strip(),
        encoding="utf-8",
    )
    registry = tmp_path / "dashboard-compute-host.json"
    completions: list[dict] = []
    rpc_events: list[dict] = []
    supervisor = HostSupervisor(
        registry_path=registry,
        argv=[sys.executable, str(script)],
        rpc_sink=rpc_events.append,
        respawn_max=2,
        heartbeat_secs=1,
        expected_build_sha="test",
        autostart=False,
    )
    try:
        supervisor.start()
        supervisor.submit_turn(
            {"type": "turn.start", "sid": "sid-1", "request_id": "turn-1", "text": "hello"},
            on_complete=completions.append,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not completions:
            time.sleep(0.02)
        assert completions, "host crash did not complete pending turn"
        assert completions[0]["type"] == "turn.error"
        assert completions[0]["reason"] == "crash"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not supervisor.is_running():
            time.sleep(0.02)
        assert supervisor.is_running()
    finally:
        supervisor.shutdown()


def _make_compress_host_session(events: list) -> dict:
    class _Agent:
        model = "host-model"
        provider = "host-provider"
        tools = []
        _cached_system_prompt = ""
        session_input_tokens = 1
        session_output_tokens = 1
        session_prompt_tokens = 1
        session_completion_tokens = 1
        session_total_tokens = 2
        session_api_calls = 1
        session_id = "rotated-id"

    agent = _Agent()
    agent.context_compressor = type("ContextEngineStub", (), {})()
    agent.context_compressor.on_session_start = (
        lambda *_args, **_kwargs: events.append("notify")
    )
    return {
        "agent": agent,
        "session_key": "before-key",
        "history": [
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "before"},
        ],
        "history_lock": threading.Lock(),
        "history_version": 2,
        "running": False,
        "manual_compression_lock": threading.Lock(),
    }


def test_compute_host_compress_control_notifies_engine_after_commit(monkeypatch):
    """The compute-host slash.compress route must fire the context-engine
    boundary hook exactly once, and only AFTER the host commits the compressed
    history + session-key sync (salvaged #65670, extended to this route)."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
        finalize_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)

    def _compress(sess, focus_topic=None, **_kwargs):
        # Simulate agent._compress_context(defer_context_engine_notification=True)
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )
        with sess["history_lock"]:
            sess["history"] = [{"role": "summary", "content": "compressed"}]
            sess["history_version"] = 3

    def _sync(sid, sess):
        events.append("sync")
        sess["session_key"] = "after-key"

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _sync)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-1",
                "route_name": "slash.compress",
                "command": "/compress",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    # Exactly one notification, after the session-key commit.
    assert events == ["sync", "notify"]
    assert ack["session_key"] == "after-key"
    # Nothing pending leaks onto the agent for a later compress to misfire.
    assert (
        finalize_context_engine_compression_notification(
            session["agent"], committed=True
        )
        is False
    )


def test_compute_host_compress_control_failure_discards_notification(monkeypatch):
    """When the host-side compress mirror fails after compression queued the
    boundary notification, the pending hook must be discarded — never left to
    fire against a boundary the host rejected."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
        finalize_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)

    def _compress(sess, focus_topic=None, **_kwargs):
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic host commit failure")

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", _boom)
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compress-2",
                "route_name": "slash.compress",
                "command": "/compress",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compress-2",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert events == []
    assert "live session sync failed" in str(ack.get("output") or "")
    # The pending notification was discarded, not left on the agent.
    assert (
        finalize_context_engine_compression_notification(
            session["agent"], committed=True
        )
        is False
    )
    assert events == []


def test_compute_host_compact_alias_routes_to_compress_mirror(monkeypatch):
    """slash.compress control frames forward the user's raw alias verbatim;
    /compact must reach the compress mirror (and its deferred-notification
    finalize wiring), not silently no-op."""
    from agent.conversation_compression import (
        _queue_context_engine_compression_notification,
    )
    from tui_gateway import server

    out = io.StringIO()
    host = ComputeHost(stdout=out, max_workers=1, heartbeat_secs=0)
    events: list[str] = []
    session = _make_compress_host_session(events)
    calls: dict[str, object] = {}

    def _compress(sess, focus_topic=None, **_kwargs):
        calls["focus"] = focus_topic
        _queue_context_engine_compression_notification(
            sess["agent"],
            new_session_id="rotated-id",
            old_session_id="before-key",
        )

    server._sessions["sid"] = session
    monkeypatch.setenv("HERMES_COMPUTE_HOST_CHILD", "1")
    monkeypatch.setattr(server, "_compress_session_history", _compress)
    monkeypatch.setattr(
        server, "_sync_session_key_after_compress", lambda *_a: events.append("sync")
    )
    monkeypatch.setattr(server, "_emit", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda _agent, _session=None: {"model": "host-model", "usage": {"total": 2}},
    )

    try:
        host.handle_frame(
            {
                "type": "control",
                "sid": "sid",
                "request_id": "compact-1",
                "route_name": "slash.compress",
                "command": "/compact focus topic",
            }
        )
        ack = _wait_for_frame(
            out,
            lambda f: f.get("type") == "control.ack" and f.get("request_id") == "compact-1",
        )
    finally:
        server._sessions.pop("sid", None)
        host.close()

    assert calls == {"focus": "focus topic"}
    assert events == ["sync", "notify"]
    assert ack["route_name"] == "slash.compress"
