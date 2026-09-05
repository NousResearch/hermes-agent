"""Auto-yield patch: a cross-surface send fenced out by an IDLE same-process desktop session
takes over instead of being refused; a RUNNING desktop owner still fences; a foreign-process
owner is never fabricated into a takeover.

These exercise the REAL bound functions on tui_gateway.server (split-module binding at import),
with a real on-disk lease registry under a temp HERMES_HOME. Behavior contract, not mocks.
"""

from __future__ import annotations

import json
import threading

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


def test_no_local_owner_refusal_propagates(gateway):
    """A lease held by another LIVE process still refuses; auto-yield must not fabricate a takeover."""
    import subprocess

    from hermes_cli.active_sessions import SESSION_NOT_OWNED, try_acquire_active_session

    key = "sess-yield-3"
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, track_liveness=False,
        metadata={"live_session_id": "foreign"})
    assert refusal is None
    # Rewrite the registry so the entry belongs to a REAL other process (a fake pid would
    # read as dead and be pruned — which is correct upstream behavior, not this contract).
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        registry_path = lease.state_path
        data = json.loads(registry_path.read_text())
        for entry in data["entries"]:
            if entry["session_id"] == key:
                entry["pid"] = foreign.pid
                entry.pop("process_start_time", None)
        registry_path.write_text(json.dumps(data))

        relay = _relay_session(key)
        result = server._ensure_active_session_slot("relay-live", relay)

        assert getattr(result, "reason", None) == SESSION_NOT_OWNED
    finally:
        foreign.terminate()
        foreign.wait()
