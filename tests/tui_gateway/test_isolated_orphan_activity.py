"""Detached Desktop/TUI turns use child-owned activity, not process heartbeats."""

from pathlib import Path
import sys
import threading
import time

import pytest

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor


class _Timer:
    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback

    def start(self):
        pass

    def cancel(self):
        pass


def _session(sid):
    return dict(agent=None, agent_ready=threading.Event(), session_key=sid,
                history=[], history_version=0, history_lock=threading.Lock(),
                running=True, transport=server._detached_ws_transport,
                attached_images=[], cols=80, source="desktop", inflight_turn=None)


@pytest.mark.parametrize("mode", ["fresh", "stale", "missing", "previous"])
def test_real_child_detached_turn_activity(tmp_path, monkeypatch, mode):
    """Real supervisor pipes, child admission/turn thread, bridge and orphan timer.

    Only the agent/provider and environment-heavy UI side effects are stubbed in
    the child. Its activity writer and snapshot contract are the production ones.
    """
    sid = "detached-turn"
    session = _session(sid)
    forwarded = []
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "write_json", lambda msg: forwarded.append(msg) or True)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {"turn_isolation": True})
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 30.0)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20.0)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *args: False)
    monkeypatch.setattr(server, "_session_cwd", lambda s: str(tmp_path))
    home = tmp_path / "home"
    home.mkdir()
    supervisor = HostSupervisor(
        argv=[sys.executable, str(Path(__file__).resolve()), mode, str(tmp_path)],
        registry_path=tmp_path / "host.json", env={"HERMES_HOME": str(home)},
        expected_hermes_home=str(home), rpc_sink=server._relay_compute_host_rpc,
        heartbeat_secs=1, autostart=False)
    monkeypatch.setattr(server, "_get_compute_host_supervisor", lambda *args: supervisor)
    try:
        response = server._submit_prompt_to_compute_host("request", sid, session, "work")
        assert response["result"]["turn_isolation"] is True
        deadline = time.monotonic() + 12
        while not (tmp_path / "provider-started").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "provider-started").exists(), supervisor._stderr_tail
        # Give the actual child-to-parent sampler a bounded opportunity to arrive.
        deadline = time.monotonic() + 3
        while not server._ws_orphan_turn_activity_is_fresh(session) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert supervisor.is_running()
        assert session["agent"] is None
        assert server._ws_orphan_turn_activity_is_fresh(session) is (mode == "fresh")
        monkeypatch.setattr(server.threading, "Timer", _Timer)
        server._schedule_ws_orphan_reap(sid)
        server._pending_ws_reaps[sid].callback()
        assert bool(session.get("_client_gone_interrupt_requested")) is (mode != "fresh")
        assert server._pending_ws_reaps[sid].delay == (
            20.0 if mode == "fresh" else server._WS_ORPHAN_INTERRUPT_REAP_POLL_S)
        assert not any(m.get("method") == "compute_host.activity" for m in forwarded)
        if mode != "fresh":
            deadline = time.monotonic() + 5
            while session["running"] and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not session["running"], "stale child must receive and settle the real interrupt"
        if mode == "fresh":
            old_token = session["_compute_host_turn_id"]
            old_request = next(iter(supervisor._pending_turns))
            (tmp_path / "release").touch()
            deadline = time.monotonic() + 5
            while session["running"] and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not session["running"]
            assert "_compute_host_activity_ns" not in session
            (tmp_path / "release").unlink()
            (tmp_path / "provider-started").unlink()
            session["running"] = True
            # Same sid and caller rid, same child/agent, but NO new activity.
            server._submit_prompt_to_compute_host("request", sid, session, "next")
            assert session["_compute_host_turn_id"] != old_token
            new_token = session["_compute_host_turn_id"]
            # A delayed terminal frame cannot resolve the new caller-rid reuse.
            supervisor._handle_host_frame({"type": "turn.end", "sid": sid, "request_id": old_request})
            assert session["running"]
            assert session["_compute_host_turn_id"] == new_token
            deadline = time.monotonic() + 5
            while not (tmp_path / "provider-started").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (tmp_path / "provider-started").exists()
            # Also replay a delayed sample from the previous dispatch.
            server._relay_compute_host_rpc({"method": "compute_host.activity", "params": {
                "session_id": sid, "turn_id": old_token, "activity_ns": time.perf_counter_ns()}})
            deadline = time.monotonic() + 3
            while "_compute_host_activity_ns" not in session and time.monotonic() < deadline:
                time.sleep(0.02)
            assert "_compute_host_activity_ns" in session
            assert not server._ws_orphan_turn_activity_is_fresh(session)
            server._pending_ws_reaps[sid].callback()
            assert session["_client_gone_interrupt_requested"]
    finally:
        supervisor.shutdown()


@pytest.mark.parametrize("change", ["none", "other-session", "old-turn", "not-running", "stale", "missing"])
def test_activity_relay_is_fenced_and_ages(monkeypatch, change):
    session = _session("session")
    session.update(_compute_host_active=True, _compute_host_turn_id="new-turn")
    monkeypatch.setattr(server, "_sessions", {"session": session})
    monkeypatch.setattr(server, "_WS_ORPHAN_ACTIVITY_STALE_S", 30)
    monkeypatch.setattr(server, "write_json", lambda msg: pytest.fail("internal activity leaked to client"))
    params: dict = dict(session_id="session", turn_id="new-turn", activity_ns=time.perf_counter_ns())
    if change == "other-session":
        params["session_id"] = "other"
    elif change == "old-turn":
        params["turn_id"] = "old-turn"
    elif change == "not-running":
        session["running"] = False
    elif change == "stale":
        params["activity_ns"] -= 31_000_000_000
    elif change == "missing":
        params["activity_ns"] = None
    server._relay_compute_host_rpc({"jsonrpc": "2.0", "method": "compute_host.activity", "params": params})
    assert server._ws_orphan_turn_activity_is_fresh(session) is (change == "none")
    if change == "none":
        # Repeated delivery is an observation of the same clock, not a refresh.
        monkeypatch.setattr(server.time, "perf_counter_ns", lambda: params["activity_ns"] + 31_000_000_000)
        server._relay_compute_host_rpc({"method": "compute_host.activity", "params": params})
        assert not server._ws_orphan_turn_activity_is_fresh(session)


def _run_child(mode, directory):
    import socket
    from agent.activity_tracking import ActivityTrackingMixin
    from agent.session_activity import build_activity_snapshot
    from tui_gateway.compute_host import run_host

    def no_network(*args, **kwargs):
        raise AssertionError("test child must not contact a provider")
    socket.socket.connect = no_network

    class Agent(ActivityTrackingMixin):
        def __init__(self, sid):
            self.session_id = sid
            self._interrupt = threading.Event()
            if mode == "previous":
                self._touch_activity("previous turn")

        def get_activity_summary(self):
            return build_activity_snapshot(last_activity_at=getattr(self, "_last_activity_ts", None),
                                           last_activity_description="test provider")

        def clear_interrupt(self):
            self._interrupt.clear()

        def interrupt(self, **kwargs):
            self._interrupt.set()

        def run_conversation(self, *args, **kwargs):
            Path(directory, "provider-started").touch()
            deadline = time.monotonic() + 20
            while not self._interrupt.wait(0.05) and time.monotonic() < deadline:
                if Path(directory, "release").exists():
                    break
                if mode == "fresh" and args[0] != "next":
                    self._touch_activity("provider wait")
                elif mode == "stale":
                    self._last_activity_ts = time.time() - 3600
            return {"final_response": "done", "interrupted": self._interrupt.is_set()}

    def init(sid, key, agent, history, **kwargs):
        s = _session(sid)
        s.update(agent=agent, running=False, transport=None,
                 image_counter=0, slash_worker=None, show_reasoning=False,
                 tool_progress_mode="all")
        server._sessions[sid] = s

    server._make_agent = lambda sid, *a, **kw: Agent(sid)
    server._init_session = init
    server._wire_callbacks = lambda *a: None
    server._sync_agent_model_with_config = lambda *a: None
    server._register_session_cwd = lambda *a: None
    server._tts_stream_begin = lambda: None
    server._sync_session_key_after_compress = lambda *a, **kw: None
    server._get_usage = lambda *a: {}
    run_host(stdout=sys.__stdout__)


if __name__ == "__main__":
    _run_child(sys.argv[1], sys.argv[2])
