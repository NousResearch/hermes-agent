"""Preemptible cross-surface leases: a cross-surface send fenced out by an IDLE
desktop session takes over (now by ATOMIC STEAL inside the registry flock — epoch
bump, displaced close — instead of the retired 8s yield-request dance); a RUNNING
desktop owner still fences (SESSION_BUSY); a foreign-process owner is never
fabricated into a takeover.

Cross-process compat contract (kept): OLD-code requesters still write yield-request
files; the NEW holder's lease maintenance watcher honors them only for a
still-matching, idle session, requeues busy ones with the ORIGINAL requested_at, and
drops expired or mismatched requests.

These exercise the REAL bound functions on tui_gateway.server (split-module binding at
import), with a real on-disk lease registry under a temp HERMES_HOME. Behavior
contract, not mocks. Deeper preemptible-lease coverage (steal tiers, watcher startup,
displacement detection) lives in test_lease_preempt.py.
"""

from __future__ import annotations

import json
import os
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
        "history": [],
        "history_version": 0,
        "agent": None,
        "slash_worker": None,
        "profile_home": None,
        "source": "webui",
        "created_at": time.time(),
        "last_active": time.time(),
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


def _registry_entry(session_key: str):
    from hermes_cli import active_sessions as AS
    state = AS._state_path(None)
    if not state.exists():
        return None
    return next(
        (e for e in json.loads(state.read_text())["entries"]
         if e.get("session_id") == session_key), None)


def _self_pid() -> int:
    return os.getpid()


def test_idle_desktop_owner_yields_to_cross_surface_send(gateway):
    """Relay send vs an IDLE desktop tab in this process: the claim STEALS the lease
    atomically (epoch bump), the turn is admitted, and the displaced tab closes."""
    desktop = _desktop_session("sess-yield-1", running=False)
    # The desktop's last turn finished: publish idle (busy cleared beside running=False).
    server._lease_turn_settled(desktop)
    holder_epoch = int(_registry_entry("sess-yield-1").get("epoch") or 1)
    gateway._sessions["desktop-live"] = desktop
    relay = _relay_session("sess-yield-1")

    result = server._ensure_active_session_slot("relay-live", relay)

    assert result is None, f"turn should be admitted after steal, got: {result}"
    assert relay["active_session_lease"] is not None
    entry = _registry_entry("sess-yield-1")
    assert entry["pid"] == _self_pid()
    assert entry["lease_id"] == relay["active_session_lease"].lease_id
    assert int(entry["epoch"]) == holder_epoch + 1, "a steal must bump the fencing epoch"
    assert entry["busy"] is True and entry["busy_kind"] == "user"
    # The displaced tab closes (the lease maintenance watcher detects the theft within
    # one tick); event-synced poll with a wide margin per AGENTS.md timing rules.
    deadline = time.monotonic() + 10.0
    while "desktop-live" in gateway._sessions and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "desktop-live" not in gateway._sessions, "displaced desktop tab must close"


def test_running_desktop_owner_never_yields(gateway):
    """A mid-turn desktop session keeps its lease; the cross-surface send is refused
    fast with SESSION_BUSY — including the dead-watcher variant (stale heartbeat),
    which must STILL never steal a visibly streaming turn."""
    from hermes_cli import active_sessions as AS
    from hermes_cli.active_sessions import _lock_path, _state_path, _FileLock, _write_entries

    def _age_heartbeat(age_s: float) -> None:
        state = _state_path(None)
        with _FileLock(_lock_path(None)):
            entries = json.loads(state.read_text())["entries"]
            for entry in entries:
                if entry.get("session_id") == "sess-yield-2":
                    entry["heartbeat_at"] = time.time() - age_s
            _write_entries(state, entries)

    desktop = _desktop_session("sess-yield-2", running=True)
    # A live admitted turn: publish busy_kind='user' with fresh activity.
    assert server._lease_admission_check("desktop-live", desktop) is None
    assert _registry_entry("sess-yield-2")["busy"] is True
    gateway._sessions["desktop-live"] = desktop

    relay = _relay_session("sess-yield-2")
    t0 = time.monotonic()
    result = server._ensure_active_session_slot("relay-live", relay)
    elapsed = time.monotonic() - t0
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert "mid-turn" in str(result)
    assert elapsed < 2.0, f"fresh-turn refusal must be fast, took {elapsed:.2f}s"
    assert "desktop-live" in gateway._sessions, "running owner must be untouched"

    # Dead-watcher variant: heartbeat 120s stale — STILL refused, never stolen (F1 pin).
    _age_heartbeat(120.0)
    relay2 = _relay_session("sess-yield-2")
    result = server._ensure_active_session_slot("relay-live-2", relay2)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    entry = _registry_entry("sess-yield-2")
    assert entry["lease_id"] == desktop["active_session_lease"].lease_id, (
        "a streaming turn is never stolen, in any heartbeat state")
    assert "desktop-live" in gateway._sessions


def test_foreign_live_owner_refused_fast_no_request_files(gateway):
    """Refused by a REAL other process (registry entry shaped like old-code output —
    no new fields): SESSION_NOT_OWNED in one fast claim, holder identity attached, no
    takeover fabricated, and NO yield-request file written by the new-code requester."""
    from hermes_cli import active_sessions as AS
    from hermes_cli.active_sessions import try_acquire_active_session

    key = "sess-yield-3"
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        # Acquire from THIS process, then rewrite the entry so it belongs to the real
        # live foreign pid AND carries no preemptible-lease fields (old-code shape).
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
                for field in ("epoch", "busy", "busy_kind", "busy_detail",
                              "busy_since", "activity_at", "heartbeat_at"):
                    entry.pop(field, None)
        registry_path.write_text(json.dumps(data))

        relay = _relay_session(key)
        t0 = time.monotonic()
        result = server._ensure_active_session_slot("relay-live", relay)
        waited = time.monotonic() - t0

        assert getattr(result, "reason", None) == AS.SESSION_NOT_OWNED
        assert "restart once" in str(result)
        # The refusal must carry the holder entry (cross-process handshake data).
        assert isinstance(getattr(result, "holder_entry", None), dict)
        assert result.holder_entry.get("pid") == foreign.pid
        # Single fast claim — the 8s dance is retired; nothing is fabricated.
        assert waited < 2.0, f"refusal must be fast, took {waited:.2f}s"
        # New requesters never write yield-request files to acquire.
        req_dir = AS._yield_request_dir(None)
        leftover = list(req_dir.glob("*.json")) if req_dir.exists() else []
        assert leftover == [], f"no request files may be written by the new code: {leftover}"
    finally:
        foreign.terminate()
        foreign.wait()


def test_yield_request_roundtrip_holder_honors(gateway):
    """Full cross-process compat contract, holder side: a fresh request naming one of
    THIS process's IDLE sessions closes it; a request for a RUNNING session does not;
    an expired request is dropped without effect."""
    from hermes_cli.active_sessions import (
        request_cross_surface_yield, poll_yield_requests)

    # Bound through the server (split-module rebinding resolves _sessions/_close_session_by_id).
    _yield_session_for_request = server._yield_session_for_request

    # Idle desktop tab holding a lease for sess-a; running tab holding sess-b.
    gateway._sessions["tab-idle"] = _desktop_session("sess-a", running=False)
    server._lease_turn_settled(gateway._sessions["tab-idle"])
    gateway._sessions["tab-busy"] = _desktop_session("sess-b", running=True)

    # Mint a request as if a foreign requester had been refused by this process's lease.
    ok = request_cross_surface_yield(
        "sess-a", {"pid": _self_pid(), "process_start_time": None})
    assert ok

    # THIS process polls its own request dir: it is the holder.
    mine = poll_yield_requests()
    assert len(mine) == 1 and mine[0]["session_id"] == "sess-a"

    # Honor it: the idle tab closes.
    _yield_session_for_request(gateway_home(), mine[0])
    assert "tab-idle" not in gateway._sessions
    assert "tab-busy" in gateway._sessions, "running session must survive"

    # A request for the busy session leaves it alone and REQUEUES itself (retry when
    # idle) with the ORIGINAL requested_at (bounded TTL chain).
    t0 = time.time()
    ok = request_cross_surface_yield(
        "sess-b", {"pid": _self_pid(), "process_start_time": None})
    assert ok
    mine = poll_yield_requests()
    assert len(mine) == 1 and mine[0]["session_id"] == "sess-b"
    _yield_session_for_request(gateway_home(), mine[0])
    assert "tab-busy" in gateway._sessions
    requeued = poll_yield_requests()
    assert len(requeued) == 1 and requeued[0]["session_id"] == "sess-b", \
        "busy session must requeue its yield request for a retry"
    assert requeued[0]["requested_at"] == pytest.approx(t0, abs=2.0), \
        "requeue must preserve the ORIGINAL requested_at (bounded TTL chain)"
    _yield_session_for_request(gateway_home(), requeued[0])  # consume the requeue

    # An expired request is dropped by the poll, never honored.
    from hermes_cli.active_sessions import _yield_request_dir
    stale = _yield_request_dir(gateway_home()) / "stale-abc12345.json"
    stale.write_text(json.dumps({
        "session_id": "sess-b", "holder_pid": _self_pid(),
        "holder_process_start_time": None, "requested_at": time.time() - 9999}))
    # Drain any requeued requests first so the assertion isolates the stale file.
    poll_yield_requests()
    assert poll_yield_requests() == []
    assert not stale.exists(), "expired request files must be unlinked"


def gateway_home():
    return os.environ["HERMES_HOME"]


def test_poller_leaves_fresh_foreign_requests_alone(tmp_path, monkeypatch):
    """Every backend sweeps every home, so a poller must NOT consume a request addressed to
    another live pid — only its own, or expired/corrupt ones. A fresh own-pid file of a
    FUTURE protocol also survives (its future-version reader may need it)."""
    from hermes_cli.active_sessions import _yield_request_dir, poll_yield_requests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    req_dir = _yield_request_dir()
    req_dir.mkdir(parents=True, exist_ok=True)
    (req_dir / "foreign-live.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": 424242, "holder_process_start_time": None,
        "requested_at": time.time()}))
    (req_dir / "future-own.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": _self_pid(), "protocol": 99,
        "requested_at": time.time()}))
    mine = poll_yield_requests()
    assert mine == []
    assert (req_dir / "foreign-live.json").exists(), "foreign request must survive for its owner"
    assert (req_dir / "future-own.json").exists(), "fresh future-protocol own-pid file survives"
