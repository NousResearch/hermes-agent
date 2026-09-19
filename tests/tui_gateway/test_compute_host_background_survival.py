"""A compute-host terminal and its autonomous follow-up outlive the client's turn."""
from pathlib import Path
import io
import subprocess
import sys
import threading
import time
import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor
from tests.tui_gateway.test_isolated_orphan_activity import _Timer, _session


def _until(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    pause = threading.Event()
    while time.monotonic() < deadline:
        if predicate():
            return
        pause.wait(0.02)
    assert predicate(), "timed out waiting for child/bridge event"


@pytest.mark.parametrize("ending", ["complete", "close", "crash"])
def test_isolated_idle_parent_survives_terminal_and_notification(tmp_path, monkeypatch, ending):
    sid = "background-host"
    home = tmp_path / "home"
    home.mkdir()
    session = _session(sid)
    session.update(profile_home=str(home), source="desktop")
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    frames = []
    original_write = server.write_json
    monkeypatch.setattr(server, "write_json", lambda f: frames.append(f) or original_write(f))
    supervisor = HostSupervisor(
        argv=[sys.executable, str(Path(__file__).resolve()), "child", str(tmp_path)],
        registry_path=tmp_path / "host.json", env={"HERMES_HOME": str(home)},
        expected_hermes_home=str(home), rpc_sink=server._relay_compute_host_rpc,
        heartbeat_secs=1, respawn_max=0, autostart=False)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *args: supervisor)
    try:
        assert server._submit_prompt_to_compute_host("request", sid, session, "launch")["result"]["turn_isolation"]
        _until(lambda: not session["running"])
        assert session["agent"] is None
        assert session.get("_compute_host_pending_work"), supervisor._stderr_tail
        monkeypatch.setattr(server.threading, "Timer", _Timer)
        server._schedule_ws_orphan_reap(sid)
        timer = server._pending_ws_reaps[sid]
        timer.callback()
        assert sid in server._sessions
        assert not server._session_is_lru_evictable(sid, session)
        from hermes_cli.web_server_idle_exit import turn_in_flight
        assert turn_in_flight() is True
        if ending == "crash":
            token = session["_compute_host_work_token"]
            (tmp_path / "crash").touch()
            _until(lambda: not supervisor.is_running())
            _until(lambda: not session.get("_compute_host_pending_work"))
            _until(lambda: any("background completion is unavailable" in str(f) for f in frames))
            assert "_compute_host_turn_id" not in session
            assert "_compute_host_activity_ns" not in session
            session.update(_compute_host_work_token="new-generation", _compute_host_pending_work=True)
            server._relay_compute_host_rpc({"method": "compute_host.lost", "params": {
                "session_id": sid, "work_token": token, "message": "late old-host death"}})
            assert session["_compute_host_work_token"] == "new-generation"
            assert session["_compute_host_pending_work"] is True
            server._relay_compute_host_rpc({"method": "compute_host.work", "params": {
                "session_id": sid, "work_token": token, "pending_work": True}})
            assert session["_compute_host_pending_work"] is True
            server._relay_compute_host_rpc({"method": "compute_host.lost", "params": {
                "session_id": sid, "work_token": "new-generation", "message": "replacement exited"}})
            assert not session.get("_compute_host_pending_work")
            server._pending_ws_reaps[sid].callback()
            assert sid not in server._sessions
            return
        if ending == "close":
            assert server._close_session_by_id(sid, end_reason="tui_close")
            assert (tmp_path / "owner-closed").exists(), supervisor._stderr_tail
            assert not any("BACKGROUND_DELIVERED" in str(f) for f in frames)
            return
        (tmp_path / "release").touch()
        _until(lambda: not session.get("_compute_host_pending_work"))
        payload = server._live_session_payload(sid, session)
        from hermes_state import SessionDB
        with SessionDB(home / "state.db") as db:
            contents = [m["content"] for m in db.get_messages(sid)]
        assert contents.count("BACKGROUND_DELIVERED") == 1
        assert "BACKGROUND_DELIVERED" in str(payload["messages"])
        assert sum(f.get("params", {}).get("type") == "message.complete" for f in frames) == 2
        assert not any(f.get("method") == "compute_host.work" for f in frames)
        # A superseded work snapshot must not pin a new generation forever.
        server._relay_compute_host_rpc({"method": "compute_host.work", "params": {
            "session_id": sid, "work_token": "old-generation", "pending_work": True}})
        assert not session.get("_compute_host_pending_work")
        server._pending_ws_reaps[sid].callback()
        assert sid not in server._sessions
    finally:
        (tmp_path / "release").touch()
        server._cancel_ws_orphan_reap(sid)
        supervisor.shutdown()


def test_real_host_death_during_foreground_settles_only_its_generation(tmp_path, monkeypatch):
    sid = "foreground-host"
    home = tmp_path / "home"
    home.mkdir()
    session = _session(sid)
    session.update(profile_home=str(home), source="desktop")
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    frames = []
    monkeypatch.setattr(server, "write_json", lambda frame: frames.append(frame) or True)
    supervisor = HostSupervisor(
        argv=[sys.executable, str(Path(__file__).resolve()), "child", str(tmp_path)],
        registry_path=tmp_path / "host.json", env={"HERMES_HOME": str(home)},
        expected_hermes_home=str(home), rpc_sink=server._relay_compute_host_rpc,
        heartbeat_secs=1, respawn_max=0, autostart=False)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *args: supervisor)
    try:
        assert server._submit_prompt_to_compute_host(
            "request", sid, session, "block-foreground")["result"]["turn_isolation"]
        _until(lambda: (tmp_path / "foreground-started").exists())
        old_token = session["_compute_host_work_token"]
        assert session["running"] is True
        (tmp_path / "foreground-crash").touch()
        _until(lambda: session["running"] is False)
        assert any("compute host exited" in str(frame) for frame in frames)
        assert "_compute_host_turn_id" not in session
        assert not session.get("_compute_host_pending_work")

        session.update(_compute_host_turn_id="replacement", _compute_host_work_token="replacement",
                       _compute_host_pending_work=True, running=True)
        server._relay_compute_host_rpc({"method": "compute_host.lost", "params": {
            "session_id": sid, "work_token": old_token, "message": "late old generation"}})
        assert session["_compute_host_turn_id"] == "replacement"
        assert session["_compute_host_work_token"] == "replacement"
        assert session["_compute_host_pending_work"] is True
        assert session["running"] is True
    finally:
        supervisor.shutdown()


def test_submit_turn_sends_to_the_host_registered_as_its_owner(tmp_path, monkeypatch):
    class Proc:
        def __init__(self):
            self.stdin = io.StringIO()
            self.pid = 1

        def poll(self):
            return None

    supervisor = HostSupervisor(registry_path=tmp_path / "host.json", autostart=False)
    old = Proc()
    replacement = Proc()
    supervisor._proc = old
    send_reached = threading.Event()
    replacement_attempted = threading.Event()
    replacement_started = threading.Event()
    crossed_boundary = []
    original_send = supervisor._send_frame

    def paused_send(frame):
        send_reached.set()
        assert replacement_attempted.wait(timeout=5)
        original_send(frame)

    monkeypatch.setattr(supervisor, "_send_frame", paused_send)
    submitter = threading.Thread(target=supervisor.submit_turn, args=({
        "sid": "session", "request_id": "request", "turn_id": "generation"},))
    submitter.start()
    assert send_reached.wait(timeout=5)

    def install_replacement():
        acquired = supervisor._lock.acquire(blocking=False)
        crossed_boundary.append(acquired)
        replacement_attempted.set()
        if not acquired:
            supervisor._lock.acquire()
        try:
            supervisor._proc = replacement
            replacement_started.set()
        finally:
            supervisor._lock.release()

    replacer = threading.Thread(target=install_replacement)
    replacer.start()
    submitter.join(timeout=5)
    replacer.join(timeout=5)

    assert not submitter.is_alive() and not replacer.is_alive()
    assert replacement_started.is_set()
    assert crossed_boundary == [False]
    assert supervisor._pending_turn_hosts["request"] is old
    assert '"request_id":"request"' in old.stdin.getvalue()
    assert replacement.stdin.getvalue() == ""


def test_late_old_waiter_settles_old_ownership_without_touching_replacement(tmp_path):
    class Proc:
        def __init__(self, code=17):
            self.code = code
            self.release = threading.Event()
            self.wait_returned = threading.Event()
            self.stdin = io.StringIO()
            self.pid = 1

        def wait(self):
            self.release.wait(timeout=5)
            self.wait_returned.set()
            return self.code

        def poll(self):
            return None

    frames = []
    callbacks = []
    supervisor = HostSupervisor(
        registry_path=tmp_path / "host.json", rpc_sink=frames.append,
        respawn_max=0, autostart=False)
    old = Proc()
    replacement = Proc()
    supervisor._proc = old
    supervisor.registry_path.write_text("old", encoding="utf-8")
    supervisor._pending_turns["old-request"] = (
        "old-session", None)
    supervisor._pending_turn_hosts["old-request"] = old
    supervisor._session_work_tokens[old] = {"old-session": "old-token"}

    def start_replacement():
        with supervisor._lock:
            if supervisor._proc is None:
                supervisor._proc = replacement
                supervisor.registry_path.write_text("replacement", encoding="utf-8")

    supervisor.start = start_replacement

    def replace_from_crash_callback(frame):
        callbacks.append(("old", frame))
        supervisor.submit_turn(
            {"sid": "new-session", "request_id": "new-request", "turn_id": "new-token"},
            on_complete=lambda result: callbacks.append(("new", result)))

    supervisor._pending_turns["old-request"] = ("old-session", replace_from_crash_callback)

    waiter = threading.Thread(target=supervisor._wait_for_exit, args=(old,))
    waiter.start()
    old.release.set()
    waiter.join(timeout=5)
    assert not waiter.is_alive()
    assert supervisor._proc is replacement
    assert supervisor.registry_path.read_text(encoding="utf-8") == "replacement"
    assert supervisor._stopped_respawning is False
    assert supervisor._restart_times == []
    assert "new-request" in supervisor._pending_turns
    assert supervisor._pending_turn_hosts["new-request"] is replacement
    assert supervisor._session_work_tokens[replacement] == {"new-session": "new-token"}
    assert callbacks == [("old", {
        "type": "turn.error", "sid": "old-session", "request_id": "old-request",
        "reason": "crash", "message": "compute host exited with code 17"})]
    assert any(frame.get("params", {}).get("work_token") == "old-token" for frame in frames)
    assert not any(frame.get("params", {}).get("session_id") == "new-session" for frame in frames)


def _child(directory):
    import socket
    from types import SimpleNamespace
    from agent.session_persistence import SessionPersistenceMixin
    from agent.client_lifecycle import ClientLifecycleMixin
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from tools.process_registry import process_registry
    from tui_gateway.compute_host import run_host

    def no_network(*args, **kwargs):
        raise AssertionError("test child must not contact a provider")
    socket.socket.connect = no_network

    class Agent(SessionPersistenceMixin, ClientLifecycleMixin):
        def __init__(self, sid):
            self.session_id = sid
            self._process_owner_task_ids = set()
            self._session_db = SessionDB(get_hermes_home() / "state.db")
            self._session_db.create_session(sid, source="desktop")
            self._session_db_created = True
            self._last_flushed_db_idx = 0

        def clear_interrupt(self): pass

        def close(self):
            self._close_task_resources(self.session_id)
            if hasattr(self, "proc"):
                self.proc.wait(timeout=5)
            Path(directory, "owner-closed").touch()

        def run_conversation(self, message, **kwargs):
            owner = kwargs.get("task_id") or self.session_id
            self._process_owner_task_ids.add(owner)
            if message == "block-foreground":
                Path(directory, "foreground-started").touch()
                _until(lambda: Path(directory, "foreground-crash").exists(), timeout=60)
                import os
                os._exit(19)
            if message == "launch":
                proc = subprocess.Popen([sys.executable, "-u", "-c",
                    "import sys; sys.stdin.readline(); print('CHILD_RESULT')"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True)
                self.proc = proc
                process_registry.adopt_local(proc, command="isolated child", cwd=directory,
                    task_id=owner, owner_task_id=owner, session_key=self.session_id)
                def release():
                    _until(lambda: any(Path(directory, name).exists() for name in ("release", "crash")), timeout=60)
                    if proc.poll() is None:
                        proc.stdin.write("finish\n")
                        proc.stdin.flush()
                    proc.wait(timeout=10)
                    if Path(directory, "crash").exists():
                        import os
                        os._exit(18)
                threading.Thread(target=release, daemon=True).start()
                response = "launched"
            else:
                assert "CHILD_RESULT" in message
                response = "BACKGROUND_DELIVERED"
            messages = list(kwargs.get("conversation_history") or []) + [
                {"role": "user", "content": message}, {"role": "assistant", "content": response}]
            self._persist_session(messages)
            return {"final_response": response, "messages": messages}

    def init(sid, key, agent, history, **kwargs):
        session = _session(sid)
        session.update(agent=agent, running=False, image_counter=0, slash_worker=None,
                       show_reasoning=False, tool_progress_mode="all", profile_home=str(get_hermes_home()))
        server._sessions[sid] = session
        session["_notif_stop"] = server._start_notification_poller(sid, session)

    server._make_agent = lambda sid, *args, **kwargs: Agent(sid)
    server._init_session = init
    server._wire_callbacks = lambda *args: None
    server._sync_agent_model_with_config = lambda *args: None
    server._register_session_cwd = lambda *args: None
    server._tts_stream_begin = lambda *args: None
    server._sync_session_key_after_compress = lambda *args, **kwargs: None
    server._get_usage = lambda *args: {}
    server._start_usage_ticker = lambda *args: (threading.Event(), SimpleNamespace(join=lambda: None))
    run_host(stdout=sys.__stdout__)


if __name__ == "__main__":
    _child(sys.argv[2])
