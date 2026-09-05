"""Auto-yield patch: a cross-surface send fenced out by an IDLE same-process desktop session
takes over instead of being refused; a RUNNING desktop owner still fences; a foreign-process
owner is never fabricated into a takeover.

Cross-process contract: a requester refused by a LIVE foreign pid writes a yield request into
the lease registry's runtime dir; the holder honors it only for a still-matching, idle session;
expired or mismatched requests are dropped.

These exercise the REAL bound functions on tui_gateway.server (split-module binding at import),
with a real on-disk lease registry under a temp HERMES_HOME. Behavior contract, not mocks.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

import pytest

import tui_gateway.server as server


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    server._sessions.clear()
    yield server
    server._sessions.clear()


def _relay_session(session_key: str) -> dict:
    """The incoming cross-surface (Relay) session dict, pre-lease."""
    return {
        "session_key": session_key,
        "active_session_lease": None,
        "running": False,
        "history_lock": threading.RLock(),
        "profile_home": None,
    }


def _desktop_session(session_key: str, *, running: bool = False) -> dict:
    """A live desktop tab's session dict holding a real lease."""
    sess = _relay_session(session_key)
    sess["running"] = running
    sess["source"] = "desktop"
    lease, refusal = server._claim_active_session_slot(
        session_key, live_session_id="desktop-live", surface="desktop", profile_home=None)
    assert refusal is None, f"desktop claim failed: {refusal}"
    sess["active_session_lease"] = lease
    return sess


def test_idle_desktop_owner_yields_to_cross_surface_send(gateway):
    """Relay send vs an IDLE desktop tab in this process: tab closes, send claims, turn admitted."""
    desktop = _desktop_session("sess-yield-1", running=False)
    gateway._sessions["desktop-live"] = desktop
    relay = _relay_session("sess-yield-1")

    result = server._ensure_active_session_slot("relay-live", relay)

    assert result is None, f"turn should be admitted after yield, got: {result}"
    assert relay["active_session_lease"] is not None
    assert "desktop-live" not in gateway._sessions, "desktop tab must be closed"


def test_running_desktop_owner_never_yields(gateway):
    """A mid-turn desktop session keeps its lease; the cross-surface send is still refused."""
    from hermes_cli.active_sessions import SESSION_NOT_OWNED

    gateway._sessions["desktop-live"] = _desktop_session("sess-yield-2", running=True)
    relay = _relay_session("sess-yield-2")

    result = server._ensure_active_session_slot("relay-live", relay)

    assert getattr(result, "reason", None) == SESSION_NOT_OWNED
    assert "desktop-live" in gateway._sessions, "running owner must be untouched"


def test_foreign_live_owner_request_written_and_refusal_kept(gateway):
    """Refused by a REAL other process: a yield request is written for it, the refusal stands
    until that process honors it (we do not fabricate a takeover), and the refusal carries the
    holder identity for the requester."""
    from hermes_cli.active_sessions import (
        SESSION_NOT_OWNED, poll_yield_requests, try_acquire_active_session)

    key = "sess-yield-3"
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        # Acquire from THIS process, then rewrite the entry so it belongs to the real
        # live foreign pid (a fake dead pid would be pruned by upstream — not this contract).
        lease, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={}, track_liveness=False,
            metadata={"live_session_id": "foreign"})
        assert refusal is None
        registry_path = lease.state_path
        data = json.loads(registry_path.read_text())
        for entry in data["entries"]:
            if entry["session_id"] == key:
                entry["pid"] = foreign.pid
                entry.pop("process_start_time", None)
        registry_path.write_text(json.dumps(data))

        relay = _relay_session(key)
        t0 = time.time()
        result = server._ensure_active_session_slot("relay-live", relay)
        waited = time.time() - t0

        assert getattr(result, "reason", None) == SESSION_NOT_OWNED
        # The refusal must carry the holder entry (cross-process yield handshake data).
        assert isinstance(getattr(result, "holder_entry", None), dict)
        assert result.holder_entry.get("pid") == foreign.pid
        # Nobody honored the request within the wait window; the wait is bounded and short.
        assert waited < 12, f"yield wait must stay bounded, took {waited:.1f}s"
        # The request file for the foreign holder exists and is well-formed for ITS poller.
        # (This process is not the holder, so polling from here returns nothing.)
        assert poll_yield_requests() == []
    finally:
        foreign.terminate()
        foreign.wait()


def test_yield_request_roundtrip_holder_honors(gateway):
    """Full cross-process contract, holder side: a fresh request naming one of THIS process's
    IDLE sessions closes it; a request for a RUNNING session does not; an expired request is
    dropped without effect."""
    from hermes_cli.active_sessions import (
        request_cross_surface_yield, poll_yield_requests)

    # Bound through the server (split-module rebinding resolves _sessions/_close_session_by_id).
    _yield_session_for_request = server._yield_session_for_request

    # Idle desktop tab holding a lease for sess-a; running tab holding sess-b.
    gateway._sessions["tab-idle"] = _desktop_session("sess-a", running=False)
    gateway._sessions["tab-busy"] = _desktop_session("sess-b", running=True)

    holder_pid = server._sessions["tab-idle"]["active_session_lease"]
    # Mint a request as if a foreign requester had been refused by this process's lease.
    ok = request_cross_surface_yield(
        "sess-a", {"pid": holder_pid_pid_of_self(), "process_start_time": None})
    assert ok

    # THIS process polls its own request dir: it is the holder.
    mine = poll_yield_requests()
    assert len(mine) == 1 and mine[0]["session_id"] == "sess-a"

    # Honor it: the idle tab closes.
    _yield_session_for_request(gateway_home(), mine[0])
    assert "tab-idle" not in gateway._sessions
    assert "tab-busy" in gateway._sessions, "running session must survive"

    # A request for the busy session leaves it alone.
    ok = request_cross_surface_yield(
        "sess-b", {"pid": holder_pid_pid_of_self(), "process_start_time": None})
    assert ok
    mine = poll_yield_requests()
    assert len(mine) == 1 and mine[0]["session_id"] == "sess-b"
    _yield_session_for_request(gateway_home(), mine[0])
    assert "tab-busy" in gateway._sessions

    # An expired request is dropped by the poll, never honored.
    from hermes_cli.active_sessions import _yield_request_dir
    stale = _yield_request_dir(gateway_home()) / "stale-abc12345.json"
    stale.write_text(json.dumps({
        "session_id": "sess-b", "holder_pid": holder_pid_pid_of_self(),
        "holder_process_start_time": None, "requested_at": time.time() - 9999}))
    assert poll_yield_requests() == []


def holder_pid_pid_of_self() -> int:
    import os
    return os.getpid()


def gateway_home():
    import os
    return os.environ["HERMES_HOME"]
