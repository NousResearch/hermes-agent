"""A retained Desktop session consumes a real terminal completion without a human prompt.

Regression for #143959 / #143927: disconnect is not finalization, and an
ordinary status reader must not erase the recovered child's pending lost wake.
These tests keep the process registry, poller, batch and turn admission real;
only the model and unrelated UI/config refresh work are replaced.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def consumer(tmp_path, monkeypatch):
    from tools import process_registry as processes
    from hermes_cli.active_sessions import ActiveSessionLease

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    server = importlib.import_module("tui_gateway.server")
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    for name in ("_maybe_fire_tui_loop_tick", "_maybe_fire_tui_heartbeat_tick", "_notif_poll_kanban",
                 "_wire_callbacks", "_sync_agent_model_with_config", "_register_session_cwd"):
        monkeypatch.setattr(server, name, lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_get_usage", lambda a: {})
    frames, calls = [], []
    stop, reached = threading.Event(), threading.Event()
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: frames.append(a))
    key, sid = "retained-process-owner", "retained-ui"

    def model(message, **kwargs):
        calls.append((message, kwargs))
        reached.set()
        stop.set()
        return {"final_response": "REPORTED-BACK", "messages": [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "REPORTED-BACK"}]}

    agent = SimpleNamespace(session_id=key, clear_interrupt=lambda: None, run_conversation=model,
                            _process_owner_task_ids={key}, _session_db=None)
    session: dict = dict(session_key=key, agent=agent, history=[], history_version=0,
                   history_lock=threading.RLock(), running=False, attached_images=[], image_counter=0,
                   cols=80, slash_worker=None, show_reasoning=False, tool_progress_mode="all",
                   inflight_turn=None, last_active=0.0, created_at=0.0,
                   active_session_lease=ActiveSessionLease(
                       lease_id="fixture-borrowed", session_id=key, surface="desktop", enabled=False))
    monkeypatch.setitem(server._sessions, sid, session)
    yield SimpleNamespace(server=server, registry=registry, session=session, sid=sid,
                          stop=stop, reached=reached, frames=frames, calls=calls)
    stop.set()
    if thread := session.get("_run_thread"):
        thread.join(10)
        assert not thread.is_alive()
    server._sessions.pop(sid, None)


def _finish_child(c, tmp_path):
    """adopt_local avoids platform service managers while using the real reader and exit publisher."""
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(1); print('REAL-RESULT')"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=tmp_path, env={k: os.environ[k] for k in ("PATH", "HOME", "HERMES_HOME", "TMPDIR") if k in os.environ})
    tracked = c.registry.adopt_local(child, command="fixture child", cwd=str(tmp_path),
        task_id=c.session["session_key"], owner_task_id=c.session["session_key"],
        session_key=c.session["session_key"], notify_on_complete=True)
    try:
        assert not c.server._session_is_reapable(c.sid, c.session)
        assert child.stdin is not None
        child.stdin.write("x")
        child.stdin.flush()
        assert tracked._completion_event.wait(10)
        tracked._reader_thread.join(10)
        assert not tracked._reader_thread.is_alive()
        assert tracked.exit_code == 0
        assert c.registry.completion_queue.qsize() == 1
        assert not c.server._session_is_reapable(c.sid, c.session)
        return tracked
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        for stream in (child.stdin, child.stdout):
            if stream and not stream.closed:
                stream.close()


def _deliver(c):
    worker = threading.Thread(target=c.server._notification_poller_loop,
                              args=(c.stop, c.sid, c.session), daemon=True)
    worker.start()
    try:
        assert c.reached.wait(10), "the retained owner's poller must reach the model without a new prompt"
    finally:
        c.stop.set()
        worker.join(10)
        assert not worker.is_alive()
        if turn := c.session.get("_run_thread"):
            turn.join(10)
            assert not turn.is_alive()


@pytest.mark.parametrize("detached", [False, True], ids=["quiet-live", "disconnected-retained"])
def test_retained_owner_reports_real_process_completion(consumer, tmp_path, detached):
    c = consumer
    peer = SimpleNamespace(_closed=False)
    assert c.server._attach_session_transport(c.session, peer)
    assert c.server._session_has_live_transport(c.session)
    if detached:
        peer._closed = True
        assert not c.server._detach_session_transport(c.session, peer)
        c.session["transport"] = c.server._detached_ws_transport
        assert not c.server._session_has_live_transport(c.session)
    tracked = _finish_child(c, tmp_path)
    _deliver(c)
    assert len(c.calls) == 1
    assert tracked.id in c.calls[0][0] and "REAL-RESULT" in c.calls[0][0]
    assert c.registry.completion_queue.empty()
    assert any(frame[0] == "message.complete" and frame[2]["text"] == "REPORTED-BACK" for frame in c.frames)
    assert not c.session.get("_finalized")
    if detached:
        assert c.server._attach_session_transport(c.session, SimpleNamespace(_closed=False))
        assert c.session["history"][-1]["content"] == "REPORTED-BACK"


def test_stop_holds_same_completion_until_owner_returns(consumer, tmp_path):
    c = consumer
    c.session["transport"] = c.server._detached_ws_transport
    _finish_child(c, tmp_path)
    original = c.registry.completion_queue.queue[0]
    c.session["_turn_cancel_requested"] = True
    assert c.server._session_is_reapable(c.sid, c.session)
    # Stop a poller at its real shutdown drain, not a substituted claim/queue outcome.
    c.stop.set()
    c.server._notification_poller_loop(c.stop, c.sid, c.session)
    assert c.calls == []
    assert not any(frame[0] == "message.start" for frame in c.frames)
    assert c.registry.completion_queue.qsize() == 1
    assert c.registry.completion_queue.queue[0] is original
    assert not c.session["running"]
    # A later explicit prompt clears this latch; reattachment alone must not clear it.
    assert c.server._attach_session_transport(c.session, SimpleNamespace(_closed=False))
    assert c.session["_turn_cancel_requested"]
    c.session["_turn_cancel_requested"] = False
    c.stop.clear()
    _deliver(c)
    assert len(c.calls) == 1 and "REAL-RESULT" in c.calls[0][0]
    assert c.registry.completion_queue.empty()
    c.server._notification_poller_loop(c.stop, c.sid, c.session)
    assert len(c.calls) == 1, "an empty subsequent shutdown pass must not replay the completion"


@pytest.fixture()
def recovered(consumer, tmp_path, monkeypatch):
    from tools import process_registry as processes

    c = consumer
    c.session["transport"] = c.server._detached_ws_transport
    checkpoint = tmp_path / "recovery.json"
    monkeypatch.setattr(processes, "CHECKPOINT_PATH", checkpoint)
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=tmp_path, env={k: os.environ[k] for k in ("PATH", "HOME", "HERMES_HOME", "TMPDIR") if k in os.environ})
    try:
        start = c.registry._safe_host_start_time(child.pid)
        assert start is not None
        entry = dict(session_id="proc_recovered_consumer", command="fixture recovered child",
                     pid=child.pid, pid_scope="host", host_start_time=start,
                     task_id=c.session["session_key"], owner_task_id=c.session["session_key"],
                     session_key=c.session["session_key"], notify_on_complete=True)
        checkpoint.write_text(json.dumps([entry]))
        assert c.registry.recover_from_checkpoint() == 1
        assert c.registry.poll(entry["session_id"])["status"] == "running"
        assert not c.server._session_is_reapable(c.sid, c.session)
        assert child.stdin is not None
        child.stdin.write(b"x")
        child.stdin.flush()
        assert child.wait(timeout=10) == 0
        yield c, entry, checkpoint
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=10)
        if child.stdin:
            child.stdin.close()


@pytest.mark.parametrize("first_observer", ["sweeper", "owner-keepalive", "foreign-keepalive", "poll"])
def test_recovered_exit_reaches_owner_regardless_of_first_observer(recovered, first_observer):
    c, entry, checkpoint = recovered
    if first_observer == "sweeper":
        c.registry._sweep_detached_once()
    elif first_observer == "poll":
        assert c.registry.poll(entry["session_id"])["status"] == "not_found"
    else:
        observed = c.session if first_observer == "owner-keepalive" else {"session_key": "foreign-owner"}
        c.server._session_owns_pending_process_notification(observed)
    c.registry._sweep_detached_once()
    assert c.registry.completion_queue.qsize() == 1, "an earlier status refresh must not erase the lost wake"
    event = c.registry.completion_queue.queue[0]
    assert (event["session_id"], event["session_key"], event["owner_task_id"]) == (
        entry["session_id"], entry["session_key"], entry["owner_task_id"])
    assert event["completion_reason"] == "lost" and event["exit_code"] is None
    assert c.registry.poll(entry["session_id"])["status"] == "not_found"
    assert not c.server._session_is_reapable(c.sid, c.session), "the queued notice still protects its owner"
    receipt = Path(os.environ["HERMES_HOME"]) / "logs" / "process-results" / f"{entry['session_id']}.json"
    saved = json.loads(receipt.read_text())
    assert saved["completion_reason"] == "lost" and saved["exit_code"] is None
    _deliver(c)
    assert len(c.calls) == 1 and "marked lost" in c.calls[0][0]
    assert entry["session_id"] in c.calls[0][0]
    assert any(frame[0] == "message.complete" and frame[2]["text"] == "REPORTED-BACK" for frame in c.frames)
    assert c.registry.completion_queue.empty()
    # A stale checkpoint may survive a crash; the real receipt prevents a duplicate offer.
    checkpoint.write_text(json.dumps([entry]))
    assert c.registry.recover_from_checkpoint() == 0
    c.registry._sweep_detached_once()
    assert c.registry.completion_queue.empty()


def test_refresh_keeps_owner_protected_until_lost_wake_is_published(recovered, monkeypatch):
    c, entry, _checkpoint = recovered
    publishing, release = threading.Event(), threading.Event()
    publish = c.registry._publish_lost_session

    def paused_publish(session):
        publishing.set()
        assert release.wait(10)
        publish(session)

    monkeypatch.setattr(c.registry, "_publish_lost_session", paused_publish)
    assert c.registry.poll(entry["session_id"])["status"] == "not_found"
    worker = threading.Thread(target=c.registry._sweep_detached_once, daemon=True)
    worker.start()
    try:
        assert publishing.wait(10)
        # Another observer cannot retire the notice while publication owns it.
        c.registry._sweep_detached_once()
        assert c.registry.completion_queue.empty()
        assert not c.server._session_is_reapable(c.sid, c.session)
    finally:
        release.set()
        worker.join(10)
        assert not worker.is_alive()
    assert c.registry.completion_queue.qsize() == 1
    _deliver(c)
    assert len(c.calls) == 1 and "marked lost" in c.calls[0][0]


def test_foreign_query_defers_receipt_to_original_sweeper_context(recovered, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    c, entry, checkpoint = recovered
    owner_home = Path(os.environ["HERMES_HOME"])
    query_home = tmp_path / "query-profile"
    query_home.mkdir()
    checkpoint_before = checkpoint.read_bytes()
    token = set_hermes_home_override(query_home)
    try:
        assert c.registry.poll(entry["session_id"])["status"] == "not_found"
        assert c.registry.completion_queue.empty()
        assert checkpoint.read_bytes() == checkpoint_before
        assert not (query_home / "logs" / "process-results").exists()
    finally:
        reset_hermes_home_override(token)
    # The query did not assume the event owner's receipt-writing authority.
    c.registry._sweep_detached_once()
    assert (owner_home / "logs" / "process-results" / f"{entry['session_id']}.json").exists()
    assert not (query_home / "logs" / "process-results").exists()
    _deliver(c)
    assert len(c.calls) == 1 and entry["session_id"] in c.calls[0][0]


def test_sweeper_preserves_committed_kill_and_pending_scope_stop(recovered):
    c, entry, _checkpoint = recovered
    tracked = c.registry._running[entry["session_id"]]
    # Commit a synthetic kill receipt through the real exit publisher. No PID is signalled.
    c.registry._finish_exited(tracked, -15, reason="killed", source="process.kill")
    tracked._scope_stop_pending = True
    c.registry._sweep_detached_once()
    assert tracked._scope_stop_pending
    assert tracked.exit_code == -15 and tracked.completion_reason == "killed"
    assert tracked.termination_source == "process.kill"
    assert c.registry._finished[tracked.id] is tracked
    event = c.registry.completion_queue.get_nowait()
    assert event["exit_code"] == -15 and event["completion_reason"] == "killed"
    assert c.registry.completion_queue.empty()
