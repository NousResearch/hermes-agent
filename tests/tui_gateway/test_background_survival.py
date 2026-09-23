"""Transport loss cannot retire a parent before its terminal result is delivered."""
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from agent.secret_scope import set_multiplex_active
from agent.session_persistence import SessionPersistenceMixin
from agent.client_lifecycle import ClientLifecycleMixin
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from gateway.session_context import set_session_vars, clear_session_vars
from tools import process_registry as processes
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server
from tui_gateway import event_replay
from tests.tui_gateway.test_auto_continue import turn_env, marker_home, _session


@pytest.mark.parametrize("cleanup", ["orphan", "idle", "lru", "backend"])
def test_detached_process_completion_keeps_owner_until_delivery(
    turn_env, marker_home, monkeypatch, cleanup,
):
    # Use real reader threads; only the model is replaced, never lifecycle/dispatch.
    monkeypatch.setattr(server.threading, "Thread", server._RealThread)
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    monkeypatch.setattr(server, "_start_usage_ticker", lambda *a: (threading.Event(), SimpleNamespace(join=lambda: None)))
    timers = []
    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)
        def start(self): pass
        def cancel(self): pass
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    event_replay.reset_replay_state()
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    home_a, home_b = marker_home / "a", marker_home / "b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    db_a, db_b = SessionDB(home_a / "state.db"), SessionDB(home_b / "state.db")
    for db in (db_a, db_b):
        db.create_session("parent", source="tui")
    monkeypatch.setattr(server, "_get_db", lambda: db_a)
    set_multiplex_active(True)
    calls = []
    followup_entered, release_followup = threading.Event(), threading.Event()
    original_followup = server._run_post_turn_followups
    def followup(*args):
        followup_entered.set()
        assert release_followup.wait(10)
        return original_followup(*args)
    monkeypatch.setattr(server, "_run_post_turn_followups", followup)
    class Agent(SessionPersistenceMixin, SimpleNamespace):
        pass
    agent = Agent(session_id="parent", _process_owner_task_ids={"parent"}, clear_interrupt=lambda: None,
                  _session_db=db_b, _session_db_created=True, _last_flushed_db_idx=0)
    def run(message, **kwargs):
        assert get_hermes_home() == home_b
        calls.append(message)
        messages = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "child result received"}]
        agent._persist_session(messages)
        return {"final_response": "child result received", "messages": messages}
    agent.run_conversation = run
    transport = object()
    session = _session(agent=agent, session_key="parent", profile_home=str(home_b),
                       transport=transport, last_active=1, created_at=1)
    foreign = _session(agent=SimpleNamespace(_process_owner_task_ids={"parent"}),
                       session_key="parent", profile_home=str(home_a), transport=object())
    monkeypatch.setattr(server, "_sessions", {"tab": session, "foreign": foreign})
    proc = subprocess.Popen([sys.executable, "-u", "-c", "import sys; sys.stdin.readline(); print('CHILD_RESULT')"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True, text=True)
    with server._session_profile_runtime_scope(session):
        tokens = set_session_vars(session_id="parent", ui_session_id="tab")
        try:
            child = registry.adopt_local(proc, command="controlled child", cwd=str(home_b),
                                         task_id="shared-environment", owner_task_id="parent", session_key="parent")
        finally:
            clear_session_vars(tokens)
    def protected():
        if cleanup == "orphan":
            server._pending_ws_reaps["tab"].callback()
            return server._sessions.get("tab") is session
        if cleanup == "idle":
            return not server._session_is_evictable("tab", session, 10**12)
        if cleanup == "lru":
            return not server._session_is_lru_evictable("tab", session)
        from hermes_cli.web_server_idle_exit import turn_in_flight
        return turn_in_flight() is True
    try:
        assert server._close_sessions_for_transport(transport) == (0, 1)
        assert protected(), "automatic cleanup retired a parent with a running child"
        for scoped in (foreign, session, foreign):
            with server._session_profile_runtime_scope(scoped):
                assert server._session_has_background_processes(session)
                assert not server._session_has_background_processes(foreign)
        assert proc.poll() is None
        # The process retains its raw spawning owner while compression rotates
        # the durable parent key. The other profile deliberately has the old key.
        db_b.end_session("parent", "compression")
        db_b.create_session("tip", source="tui", parent_session_id="parent")
        agent.session_id = session["session_key"] = "tip"
        proc.stdin.write("finish\n")
        proc.stdin.flush()
        assert child._completion_event.wait(10)
        event = registry.completion_queue.get(timeout=5)
        # A long engineering job must not become prunable at the moment it
        # finishes, including while its event is held outside the queue.
        child.started_at = time.time() - processes.FINISHED_TTL_SECONDS - 1
        with registry._lock:
            registry._prune_if_needed()
            # Exercise capacity pressure separately, with a fresh receipt so
            # the TTL branch cannot explain its survival.
            child.started_at = time.time()
            with monkeypatch.context() as capacity:
                capacity.setattr(processes, "MAX_PROCESSES", 1)
                registry._prune_if_needed()
        # The queue is empty: the poller has dequeued, but not claimed a turn yet.
        assert protected(), "exited child lost its parent before notification admission"
        deferred = []
        server._notif_handle_ready("foreign", foreign, [event], set(), registry, format_process_notification, deferred)
        assert deferred == [event] and not calls
        def refuse(*args, **kwargs):
            server._notif_release_turn(session)
            return False
        with monkeypatch.context() as patch:
            patch.setattr(server, "_run_prompt_submit", refuse)
            server._notif_handle_ready("tab", session, [event], set(), registry, format_process_notification, [])
        assert not registry.is_completion_consumed(child.id)
        assert registry.completion_queue.get(timeout=5) == event
        assert protected(), "refused admission must retain the result and owner"
        server._notif_handle_ready("tab", session, [event], set(), registry, format_process_notification, [])
        assert followup_entered.wait(10)
        assert not session["running"] and registry.is_completion_consumed(child.id)
        assert protected(), "post-turn continuation handoff is still active work"
        release_followup.set()
        session["_run_thread"].join(10)
        assert not session["_run_thread"].is_alive()
        assert len(calls) == 1 and "CHILD_RESULT" in calls[0]
        assert any(m.get("content") == "child result received" for m in session["history"])
        assert db_a.get_messages("parent") == []
        assert [m["content"] for m in db_b.get_messages("tip")].count("child result received") == 1
        old_timer = server._pending_ws_reaps["tab"]
        server._rebind_live_transport("tab", session, object())
        old_timer.callback()  # cancelled callback already dispatched on another thread
        assert server._sessions["tab"] is session
        replay = event_replay.events_since("tab", 0)
        assert sum(e["type"] == "message.complete" for e in replay) == 1
        assert event_replay.events_since("tab", replay[-1]["seq"]) == []
        server._notif_handle_ready("tab", session, [event], set(), registry, format_process_notification, [])
        assert len(calls) == 1, "reconnect delivered a second continuation"
        assert server._close_sessions_for_transport(session["transport"]) == (0, 1)
        server._pending_ws_reaps["tab"].callback()
        assert "tab" not in server._sessions, "finished empty session leaked"
        assert [m["content"] for m in db_b.get_messages("tip")].count("child result received") == 1
    finally:
        release_followup.set()
        if proc.poll() is None:
            proc.stdin.write("cleanup\n")
            proc.stdin.flush()
        proc.wait(timeout=10)
        server._cancel_ws_orphan_reap("tab")
        set_multiplex_active(False)
        db_a.close()
        db_b.close()
        event_replay.reset_replay_state()


@pytest.mark.parametrize("reason", ["tui_close", "tui_shutdown"])
def test_explicit_close_releases_owned_pending_results(monkeypatch, tmp_path, reason):
    registry = processes.ProcessRegistry()
    monkeypatch.setattr(processes, "process_registry", registry)
    # Both records are older than cache retention, but only the explicitly
    # closed owner relinquishes its pending result.
    for owner in ("owner", "foreign"):
        registry._finished[owner] = processes.ProcessSession(
            id=owner, command="completed child", task_id="shared-environment", owner_task_id=owner,
            exited=True, exit_code=0, started_at=1, notify_on_complete=True, origin_ui_session_id=owner,
            profile_home=str(tmp_path.resolve()))
    class Agent(ClientLifecycleMixin):
        session_id = "parent"
        _process_owner_task_ids = {"owner"}
        def close(self):
            self._close_task_resources("owner")
    session = _session(agent=Agent(), session_key="parent", profile_home=str(tmp_path),
                       transport=server._detached_ws_transport)
    monkeypatch.setattr(server, "_sessions", {"tab": session})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    assert server._session_has_background_processes(session)
    assert server._close_session_by_id("tab", end_reason=reason)
    assert "tab" not in server._sessions
    assert registry.is_completion_consumed("owner")
    assert not registry.is_completion_consumed("foreign")
    with registry._lock:
        registry._prune_if_needed()
    assert set(registry._finished) == {"foreign"}
