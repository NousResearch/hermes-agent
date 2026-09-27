"""Detached Desktop/TUI turns use child-owned activity, not process heartbeats."""

import threading
import time

import pytest

from tui_gateway import server


def _session(sid):
    return dict(agent=None, agent_ready=threading.Event(), session_key=sid,
                history=[], history_version=0, history_lock=threading.Lock(),
                running=True, transport=server._detached_ws_transport,
                attached_images=[], cols=80, source="desktop", inflight_turn=None)


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
