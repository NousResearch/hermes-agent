"""Preemptible cross-surface session leases (#auto-yield → preemptible leases).

A cross-surface send against a live foreign holder either STEALS atomically inside
``try_acquire_active_session``'s existing flock (idle holder, stalled user turn,
bg-review past grace, heartbeat-dead holder) or refuses fast with a truthful
SESSION_BUSY message. The displaced holder discovers the loss via the lease
maintenance watcher (heartbeats + epoch fence) and closes with end_reason
``lease_preempted`` (broadcast ``session.reclaimed``; the state.db row is preserved
for the new owner).

Real-subprocess holders exercise the PRODUCTION startup path: a fresh interpreter
imports ``tui_gateway.server``, whose module bottom starts the maintenance watcher
after the split-module register loop. The old suite's blind spot (calling the
server-rebound watcher functions directly — a path production never uses) is why
the previous NameError never failed CI; these tests never call the watcher except
through a cold import.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import tui_gateway.server as server
from hermes_cli import active_sessions as AS
from tui_gateway import session_reaper
from tui_gateway.turn_marker import clear_turn_marker, read_turn_marker, record_turn_start


# ── Shared harness ──────────────────────────────────────────────────────────


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    server._sessions.clear()
    yield server
    server._sessions.clear()


def _session_dict(session_key: str, *, running: bool = False, source: str | None = None) -> dict:
    """A live session dict shaped so the real teardown funnel works against it."""
    return {
        "session_key": session_key,
        "active_session_lease": None,
        "running": running,
        "history_lock": threading.RLock(),
        "history": [],
        "history_version": 0,
        "agent": None,
        "slash_worker": None,
        "profile_home": None,
        "source": source or "tui",
        "transport": None,
        "created_at": time.time(),
        "last_active": time.time(),
    }


def _registry_entries(home) -> list[dict]:
    state = AS._state_path(home if home else None)
    if not state.exists():
        return []
    return json.loads(state.read_text())["entries"]


def _entry_for(home, session_id: str) -> dict | None:
    return next((e for e in _registry_entries(home) if e.get("session_id") == session_id), None)


def _mutate_entry(home, session_id: str, mutate) -> None:
    """Registry surgery under the real lock (holder-published clock shaping for tier tests)."""
    state = AS._state_path(home if home else None)
    with AS._FileLock(AS._lock_path(home if home else None)):
        entries = json.loads(state.read_text())["entries"]
        for entry in entries:
            if entry.get("session_id") == session_id:
                mutate(entry)
        AS._write_entries(state, entries)


def _wait_for(predicate, *, timeout: float = 30.0, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    pytest.fail(f"timed out waiting for {what}")


# ── Real-subprocess holder (production cold import of tui_gateway.server) ──

_HOLDER_SCRIPT = r'''
import contextlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.environ["REPO_ROOT"])
os.environ["HERMES_HOME"] = os.environ["CHILD_HOME"]

mode = os.environ["HOLDER_MODE"]
out_dir = Path(os.environ["OUT_DIR"])
session_key = os.environ["SESSION_KEY"]
child_sid = "child-tab"

# Production cold-import: the module bottom starts the lease maintenance watcher AFTER the
# split-module register loop — the exact startup path production uses.
import tui_gateway.server as server
from hermes_cli import active_sessions as AS

# Capture INFO logs (Auto-yield: honored, Stole active session lease, …) for the parent.
logging.getLogger().addHandler(logging.FileHandler(str(out_dir / "child.log")))
logging.getLogger().setLevel(logging.INFO)

# Record broadcasts + state.db end_session calls so the parent can assert the displaced
# close reclaimed the client surface WITHOUT ending the row the new owner holds.
def _append(name, payload):
    with open(out_dir / name, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")

_orig_broadcast = server._broadcast_global_event
def _recording_broadcast(event, payload=None):
    _append("events.jsonl", {"event": event, "payload": payload})
server._broadcast_global_event = _recording_broadcast

class _RecDB:
    def get_session(self, target):
        return {"id": target, "source": os.environ.get("CHILD_SOURCE", "desktop")}
    def end_session(self, target, reason):
        _append("db_end.jsonl", {"id": target, "reason": reason})

@contextlib.contextmanager
def _fake_session_db(_session):
    yield _RecDB()
server._session_db = _fake_session_db

session = {
    "session_key": session_key, "active_session_lease": None,
    "running": os.environ.get("CHILD_RUNNING") == "1",
    "history_lock": threading.RLock(), "history": [], "history_version": 0,
    "agent": None, "slash_worker": None, "profile_home": None,
    "source": os.environ.get("CHILD_SOURCE", "desktop"), "transport": None,
    "created_at": time.time(), "last_active": time.time(),
}
server._sessions[child_sid] = session

def _signal(name, payload=None):
    (out_dir / name).write_text(json.dumps(payload or {}), encoding="utf-8")

def _watch_close():
    while True:
        if child_sid not in server._sessions:
            _signal("session_closed", {"sid": child_sid})
            return
        time.sleep(0.05)
threading.Thread(target=_watch_close, daemon=True).start()

if mode == "freeze_revalidate":
    # Instrument the flock boundary: the FIRST lock entry after this point writes BOUNDARY
    # and blocks until GO — freezing the holder between "decided to admit" and "flock taken".
    boundary = Path(os.environ["BOUNDARY_FILE"])
    go = Path(os.environ["GO_FILE"])
    original_enter = AS._FileLock.__enter__
    armed = {"freeze": False}

    def instrumented_enter(self):
        if armed["freeze"]:
            boundary.write_text("boundary", encoding="utf-8")
            deadline = time.monotonic() + 60
            while not go.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("frozen at flock boundary past deadline")
                time.sleep(0.02)
        return original_enter(self)

    AS._FileLock.__enter__ = instrumented_enter

    lease, refusal = AS.try_acquire_active_session(session_id=session_key, surface="desktop", config={})
    assert refusal is None, str(refusal)
    session["active_session_lease"] = lease
    _signal("ready", {"pid": os.getpid(), "lease_id": lease.lease_id})
    # Admission's epoch-fenced mark-busy: armed AFTER ready so the initial acquire runs
    # free; the revalidate then freezes BEFORE taking the flock, and a steal may land
    # in that gap.
    armed["freeze"] = True
    refreshed, refusal = AS.revalidate_active_session(lease, mark_busy=True, busy_kind="user")
    armed["freeze"] = False
    _signal("revalidate_result", {"reason": getattr(refusal, "reason", None), "held": refreshed is not None})
else:
    # raw_idle: a real lease with NO busy mark — the holder finished its last turn.
    lease, refusal = AS.try_acquire_active_session(session_id=session_key, surface="desktop", config={})
    assert refusal is None, str(refusal)
    session["active_session_lease"] = lease
    if os.environ.get("CHILD_ADMIT") == "1":
        # A live admitted turn: publish busy_kind='user' through the real admission helper.
        refusal = server._ensure_active_session_slot(child_sid, session)
        assert refusal is None, str(refusal)
    _signal("ready", {"pid": os.getpid(), "lease_id": lease.lease_id})

deadline = time.monotonic() + 120
release = Path(os.environ["RELEASE_FILE"])
while not release.exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("holder timed out waiting for release")
    time.sleep(0.02)
with contextlib.suppress(Exception):
    lease.release()
'''


def _spawn_holder(
    tmp_path: Path, *, mode: str, session_key: str, running: bool = False,
    admit: bool = False, source: str = "desktop", env_extra: dict | None = None,
) -> tuple[subprocess.Popen, dict]:
    """Spawn a real holder backend; returns (proc, env) with READY_FILE etc."""
    repo_root = Path(__file__).resolve().parents[2]
    child_home = tmp_path / "child-home"
    child_home.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env.update({
        "HERMES_HOME": str(child_home),
        "CHILD_HOME": str(child_home),
        "REPO_ROOT": str(repo_root),
        "OUT_DIR": str(out_dir),
        "HOLDER_MODE": mode,
        "SESSION_KEY": session_key,
        "CHILD_SOURCE": source,
        "CHILD_RUNNING": "1" if running else "0",
        "CHILD_ADMIT": "1" if admit else "0",
        "RELEASE_FILE": str(tmp_path / "release"),
    })
    env.update(env_extra or {})
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc, env


def _wait_holder_ready(proc: subprocess.Popen, env: dict, label: str = "holder") -> dict:
    ready = Path(env["OUT_DIR"]) / "ready"
    deadline = time.monotonic() + 90.0
    while not ready.exists():
        if proc.poll() is not None:
            out, err = proc.communicate()
            pytest.fail(f"{label} exited early\nstdout: {out}\nstderr: {err}")
        if time.monotonic() >= deadline:
            proc.kill()
            out, err = proc.communicate()
            pytest.fail(f"timed out waiting for {label} ready\nstdout: {out}\nstderr: {err}")
        time.sleep(0.05)
    return json.loads(ready.read_text())


def _stop_holder(proc: subprocess.Popen, env: dict) -> None:
    Path(env["RELEASE_FILE"]).touch()
    if proc.poll() is None:
        proc.kill()
    proc.communicate()


def _child_events(env) -> list[dict]:
    path = Path(env["OUT_DIR"]) / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _child_db_ends(env) -> list[dict]:
    path = Path(env["OUT_DIR"]) / "db_end.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ── THE regression pin: production-startup watcher ─────────────────────────


def test_maintenance_watcher_production_startup(tmp_path):
    """Cold import of tui_gateway.server must start a LIVE lease maintenance watcher:
    the holder's lease heartbeats within two ticks and a minted (old-code shaped)
    yield-request file is honored — the exact path whose module-namespace NameError
    silently ate every yield request ever written. Never call the rebound functions
    directly here; the subprocess is the production startup path."""
    key = "sess-watch-startup"
    proc, env = _spawn_holder(tmp_path, mode="raw_idle", session_key=key)
    try:
        ready = _wait_holder_ready(proc, env)
        home = Path(env["CHILD_HOME"])
        initial = _entry_for(home, key)
        assert initial is not None and "heartbeat_at" in initial, (
            "new-code acquire must publish a heartbeat")
        h0 = initial["heartbeat_at"]
        # Two ticks of the 1.5s watcher: the heartbeat value must MOVE (only the
        # maintenance watcher refreshes an idle lease's heartbeat after acquire).
        _wait_for(
            lambda: (_entry_for(home, key) or {}).get("heartbeat_at") not in (None, h0),
            timeout=8.0, what="watcher heartbeat refresh")

        # Old-code requester shape: a hand-minted request file with no protocol key.
        req_dir = AS._yield_request_dir(home)
        req_dir.mkdir(parents=True, exist_ok=True)
        (req_dir / "old-requester.json").write_text(json.dumps({
            "session_id": key, "holder_pid": ready["pid"],
            "holder_process_start_time": None, "requested_at": time.time()}))
        _wait_for(
            (Path(env["OUT_DIR"]) / "session_closed").exists,
            timeout=8.0, what="yield-request honored (session closed)")
        _wait_for(lambda: _entry_for(home, key) is None, timeout=8.0, what="lease released")
        # The honored INFO line trails session_closed (it logs after the close funnel
        # returns) — poll for it rather than asserting on a possibly-unflushed log.
        _wait_for(
            lambda: "Auto-yield: honored" in (Path(env["OUT_DIR"]) / "child.log").read_text(),
            timeout=8.0, what="'Auto-yield: honored' log line")
    finally:
        _stop_holder(proc, env)


# ── Steal tiers (real idle holder vs this-process requester) ───────────────


def test_idle_holder_stolen_in_single_claim(tmp_path):
    key = "sess-steal-idle"
    proc, env = _spawn_holder(tmp_path, mode="raw_idle", session_key=key)
    try:
        ready = _wait_holder_ready(proc, env)
        holder_entry = _entry_for(Path(env["CHILD_HOME"]), key)
        relay = _session_dict(key)
        relay["profile_home"] = str(env["CHILD_HOME"])
        server._sessions["relay-live"] = relay

        t0 = time.monotonic()
        result = server._ensure_active_session_slot("relay-live", relay)
        elapsed = time.monotonic() - t0

        assert result is None, f"idle holder must be stolen in one claim, got: {result}"
        assert elapsed < 0.5, f"single-claim steal took {elapsed:.3f}s (C1 pin)"
        stolen = _entry_for(None, key) if False else _entry_for(Path(env["CHILD_HOME"]), key)
        assert stolen["pid"] == os.getpid()
        assert stolen["lease_id"] == relay["active_session_lease"].lease_id
        assert stolen["epoch"] == int(holder_entry.get("epoch") or 0) + 1
        assert stolen["busy"] is True and stolen["busy_kind"] == "user"
        assert not AS.poll_yield_requests(registry_home=env["CHILD_HOME"]), (
            "new requesters never write yield-request files to acquire")
    finally:
        _stop_holder(proc, env)
        server._sessions.clear()


def test_running_user_turn_never_stolen_any_heartbeat_state(tmp_path):
    """F1 pin: busy_kind=user with FRESH activity is refused in EVERY heartbeat state —
    including a dead maintenance watcher (heartbeat 120s stale). Visible streaming is
    never stealable; the refusal is fast and leaves the holder's entry untouched."""
    key = "sess-f1-user-turn"
    proc, env = _spawn_holder(tmp_path, mode="raw_idle", session_key=key, admit=True)
    try:
        _wait_holder_ready(proc, env)
        home = Path(env["CHILD_HOME"])
        _wait_for(
            lambda: (e := _entry_for(home, key)) is not None and e.get("busy") is True,
            what="admission busy mark")
        for heartbeat_age in (0.0, 120.0):
            _mutate_entry(home, key, lambda e, ha=heartbeat_age: e.update(
                heartbeat_at=time.time() - ha,
                activity_at=time.time(), busy=True, busy_kind="user"))
            before = json.dumps(_entry_for(home, key), sort_keys=True)
            relay = _session_dict(key)
            relay["profile_home"] = str(env["CHILD_HOME"])
            server._sessions[f"relay-live-{heartbeat_age}"] = relay
            t0 = time.monotonic()
            result = server._ensure_active_session_slot(f"relay-live-{heartbeat_age}", relay)
            elapsed = time.monotonic() - t0
            assert getattr(result, "reason", None) == AS.SESSION_BUSY, (
                f"heartbeat_age={heartbeat_age}: {result}")
            assert "mid-turn" in str(result)
            assert elapsed < 0.5, f"refusal took {elapsed:.3f}s"
            assert json.dumps(_entry_for(home, key), sort_keys=True) == before, (
                "holder entry must be byte-identical after a refused steal")
    finally:
        _stop_holder(proc, env)
        server._sessions.clear()


def test_stall_tier(gateway, monkeypatch):
    """busy_kind=user with stale activity: >60s is the blocked-approval stall shape
    (stealable, the flagship case); <=60s refuses honestly; a 0 threshold disables
    stall-steal entirely."""
    from hermes_cli.active_sessions import try_acquire_active_session

    def holder_entry(key, activity_age):
        lease, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={},
            metadata={"live_session_id": "holder"})
        assert refusal is None
        _mutate_entry(None, key, lambda e: e.update(
            busy=True, busy_kind="user", activity_at=time.time() - activity_age,
            heartbeat_at=time.time()))
        return lease

    # Stalled past the threshold: stolen.
    key = "sess-stall-61"
    holder_entry(key, 61.0)
    relay = _session_dict(key)
    server._sessions["relay-1"] = relay
    assert server._ensure_active_session_slot("relay-1", relay) is None
    assert _entry_for(None, key)["pid"] == os.getpid()

    # Stale but inside the window: honest refusal, never 'mid-turn'.
    key = "sess-stall-30"
    holder_entry(key, 30.0)
    relay = _session_dict(key)
    server._sessions["relay-2"] = relay
    result = server._ensure_active_session_slot("relay-2", relay)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert "no visible progress" in str(result)

    # HERMES_LEASE_STALL_ACTIVITY_S=0 reverts to never-interrupt.
    monkeypatch.setattr(AS, "HERMES_LEASE_STALL_ACTIVITY_S", 0.0)
    key = "sess-stall-off"
    holder_entry(key, 9999.0)
    relay = _session_dict(key)
    server._sessions["relay-3"] = relay
    result = server._ensure_active_session_slot("relay-3", relay)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert _entry_for(None, key)["pid"] == os.getpid(), "stall-steal disabled must not steal"


def test_bg_review_marks_auto_busy_with_grace(gateway, monkeypatch):
    """busy_kind='auto' (bg-review) refuses inside its 90s grace and becomes stealable
    past it; the agent-side hook marks at spawn and clears at completion."""
    from hermes_cli.active_sessions import try_acquire_active_session

    def auto_busy_holder(key, since_age):
        lease, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={}, metadata={"live_session_id": "holder"})
        assert refusal is None
        _mutate_entry(None, key, lambda e: e.update(
            busy=True, busy_kind="auto", busy_detail="bg_review",
            busy_since=time.time() - since_age, heartbeat_at=time.time()))
        return lease

    key = "sess-bg-fresh"
    auto_busy_holder(key, 5.0)
    relay = _session_dict(key)
    server._sessions["relay-1"] = relay
    result = server._ensure_active_session_slot("relay-1", relay)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert "background review" in str(result)

    key = "sess-bg-past-grace"
    auto_busy_holder(key, AS.HERMES_LEASE_AUTO_BUSY_GRACE_S + 5.0)
    relay = _session_dict(key)
    server._sessions["relay-2"] = relay
    assert server._ensure_active_session_slot("relay-2", relay) is None

    # Agent-side hooks: mark-at-spawn / clear-at-completion drive the session's lease.
    calls: list[tuple] = []

    class _Agent:
        _session_lease_busy_hook = staticmethod(
            lambda kind, detail: calls.append(("hook", kind, detail)))

    class _NoAgent:
        pass

    from agent.background_review import _lease_auto_busy_begin, _lease_auto_busy_end
    agent = _Agent()
    _lease_auto_busy_begin(agent)
    _lease_auto_busy_end(agent)
    assert calls == [("hook", "auto", "bg_review"), ("hook", None, "bg_review")]
    _lease_auto_busy_begin(_NoAgent())  # no hook installed (CLI/gateway) → no-op


def test_legacy_holder_missing_heartbeat_refused_fast(tmp_path):
    """An entry shaped exactly like old-code output (no epoch/busy/heartbeat fields)
    is NEVER stolen from and refused in ONE claim — the 8s yield-request dance is gone."""
    key = "sess-legacy-holder"
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        home = tmp_path / "legacy-home"
        state = AS._state_path(home)
        state.parent.mkdir(parents=True, exist_ok=True)
        old_shaped = {
            "lease_id": "legacy-lease", "session_id": key, "surface": "desktop",
            "pid": foreign.pid, "process_start_time": None,
            "started_at": time.time(), "updated_at": time.time(), "track_liveness": True,
            "metadata": {"live_session_id": "legacy"},
        }
        with AS._FileLock(AS._lock_path(home)):
            AS._write_entries(state, [old_shaped])
        before = json.dumps(old_shaped, sort_keys=True)

        os.environ["HERMES_HOME"] = str(tmp_path / "requester-home")
        relay = _session_dict(key)
        relay["profile_home"] = str(home)
        server._sessions["relay-live"] = relay
        t0 = time.monotonic()
        result = server._ensure_active_session_slot("relay-live", relay)
        elapsed = time.monotonic() - t0

        assert getattr(result, "reason", None) == AS.SESSION_NOT_OWNED
        assert "restart once" in str(result)
        assert elapsed < 0.5, f"legacy refusal took {elapsed:.3f}s"
        assert json.dumps(_entry_for(home, key), sort_keys=True) == before, (
            "legacy holder's entry must be preserved verbatim")
        assert not AS._yield_request_dir(home).exists() or not list(
            AS._yield_request_dir(home).glob("*.json")), (
            "new requesters never write yield-request files to acquire")
    finally:
        foreign.terminate()
        foreign.wait()
        server._sessions.clear()


def test_stale_heartbeat_escape(gateway):
    """Idle + heartbeat stale >30s (dead maintenance watcher) becomes stealable —
    tonight's permanently-unyieldable state is now bounded; inside (6s,30s] it refuses
    with honest 'lease stale' text, never claiming mid-turn."""
    from hermes_cli.active_sessions import try_acquire_active_session

    def idle_holder(key, heartbeat_age):
        lease, refusal = try_acquire_active_session(
            session_id=key, surface="desktop", config={}, metadata={"live_session_id": "holder"})
        assert refusal is None
        _mutate_entry(None, key, lambda e: e.update(
            busy=False, heartbeat_at=time.time() - heartbeat_age))
        return lease

    key = "sess-hb-31"
    idle_holder(key, AS.HERMES_LEASE_HEARTBEAT_STALE_STEAL_S + 1.0)
    relay = _session_dict(key)
    server._sessions["relay-1"] = relay
    assert server._ensure_active_session_slot("relay-1", relay) is None
    assert _entry_for(None, key)["pid"] == os.getpid()

    key = "sess-hb-10"
    idle_holder(key, 10.0)
    relay = _session_dict(key)
    server._sessions["relay-2"] = relay
    result = server._ensure_active_session_slot("relay-2", relay)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert "lease stale" in str(result)
    assert "mid-turn" not in str(result)


def test_acquisition_marks_busy_no_pingpong(gateway):
    """A steal marks busy='user' at acquisition: a THIRD surface claiming immediately
    after must be refused instead of ping-ponging ownership within the first turn."""
    from hermes_cli.active_sessions import try_acquire_active_session

    key = "sess-pingpong"
    idle, refusal = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, metadata={"live_session_id": "idle-tab"})
    assert refusal is None
    server._lease_turn_settled({**_session_dict(key), "active_session_lease": idle})

    surface_a = _session_dict(key)
    server._sessions["surface-a"] = surface_a
    assert server._ensure_active_session_slot("surface-a", surface_a) is None
    entry = _entry_for(None, key)
    assert entry["pid"] == os.getpid() and entry["busy"] is True and entry["busy_kind"] == "user"

    surface_b = _session_dict(key)
    server._sessions["surface-b"] = surface_b
    result = server._ensure_active_session_slot("surface-b", surface_b)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert _entry_for(None, key)["lease_id"] == surface_a["active_session_lease"].lease_id, (
        "no ownership oscillation before the first turn ends")


# ── Displaced-holder detection (real watcher via subprocess server import) ──


@pytest.mark.parametrize("source", ["desktop", "webui"])
def test_holder_detects_displacement_within_one_tick_and_closes(tmp_path, source):
    """Parent steals; the child's watcher detects the epoch/pid mismatch within one
    tick and closes with end_reason='lease_preempted': session.reclaimed broadcast,
    state.db row PRESERVED (no end_session) — for BOTH desktop and phone/webui
    surfaces (the guard must not be desktop-only)."""
    key = f"sess-displaced-{source}"
    proc, env = _spawn_holder(
        tmp_path, mode="raw_idle", session_key=key, running=True, source=source)
    try:
        _wait_holder_ready(proc, env)
        relay = _session_dict(key)
        relay["profile_home"] = str(env["CHILD_HOME"])
        server._sessions["relay-live"] = relay
        assert server._ensure_active_session_slot("relay-live", relay) is None

        out_dir = Path(env["OUT_DIR"])
        _wait_for((out_dir / "session_closed").exists, timeout=6.0,
                  what=f"displaced {source} session closed (one tick + close)")
        # session_closed fires at POP time (teardown start); the reclaim broadcast runs
        # after finalize, and any end_session call happens BEFORE it — so once the
        # broadcast lands, the preserve decision is final.
        _wait_for(
            lambda: [e for e in _child_events(env) if e["event"] == "session.reclaimed"],
            timeout=8.0, what="session.reclaimed broadcast for the displaced close")
        reclaimed = [e for e in _child_events(env) if e["event"] == "session.reclaimed"]
        assert reclaimed[0]["payload"]["reason"] == "lease_preempted"
        # C4: the displaced close must NOT end the state.db row the new owner holds.
        assert _child_db_ends(env) == [], f"displaced close ended the DB row: {_child_db_ends(env)}"
        assert _entry_for(Path(env["CHILD_HOME"]), key)["pid"] == os.getpid()
    finally:
        _stop_holder(proc, env)
        server._sessions.clear()


def test_displaced_next_admission_fenced(gateway):
    """After a steal, the displaced holder's NEXT admission is fenced locally: the
    taken-over refusal comes back, the session closes, no turn thread starts."""
    key = "sess-displaced-admission"
    holder = _session_dict(key)
    server._sessions["holder-live"] = holder
    assert server._ensure_active_session_slot("holder-live", holder) is None
    server._lease_turn_settled(holder)  # turn finished: the inter-turn idle gap

    thief = _session_dict(key)
    server._sessions["thief-live"] = thief
    assert server._ensure_active_session_slot("thief-live", thief) is None

    result = server._ensure_active_session_slot("holder-live", holder)
    assert result is not None and "taken over" in str(result)
    assert getattr(result, "reason", None) == AS.SESSION_DISPLACED
    assert "holder-live" not in server._sessions, "displaced session must close"
    assert "_run_thread" not in holder, "no turn thread may start on a displaced session"


# ── Mutual exclusion is the flock, not timing ──────────────────────────────


def test_no_double_writer_under_lock_interleave(tmp_path):
    """BOUNDARY_FILE freeze instrumentation: (a) the holder's revalidate-mark-busy
    frozen at the flock boundary while a steal lands in the gap → the holder's
    admission returns SESSION_DISPLACED and writes NOTHING; (b) the holder's busy
    mark confirmed first → the steal sees busy=user and refuses."""
    key = "sess-interleave"
    boundary = tmp_path / "boundary"
    go = tmp_path / "go"
    proc, env = _spawn_holder(
        tmp_path, mode="freeze_revalidate", session_key=key,
        env_extra={"BOUNDARY_FILE": str(boundary), "GO_FILE": str(go)})
    try:
        ready = _wait_holder_ready(proc, env)
        home = Path(env["CHILD_HOME"])
        _wait_for(boundary.exists, timeout=30.0, what="holder frozen at flock boundary")

        # The steal lands in the gap (the frozen holder holds NO lock yet).
        lease, refusal = AS.try_acquire_active_session(
            session_id=key, surface="webui", config={}, registry_home=home,
            metadata={"live_session_id": "thief-live"})
        assert lease is not None, f"steal in the frozen gap must succeed: {refusal}"
        stolen = _entry_for(home, key)
        assert stolen["pid"] == os.getpid() and stolen["epoch"] >= 2
        after_steal = json.dumps(_entry_for(home, key), sort_keys=True)

        go.write_text("go")
        result_file = Path(env["OUT_DIR"]) / "revalidate_result"
        _wait_for(result_file.exists, timeout=30.0, what="holder revalidate result")
        result = json.loads(result_file.read_text())
        assert result["reason"] == AS.SESSION_DISPLACED, result
        assert result["held"] is False
        # The displaced admission wrote NOTHING: the stealer's entry is byte-identical.
        assert json.dumps(_entry_for(home, key), sort_keys=True) == after_steal
    finally:
        go.touch()
        _stop_holder(proc, env)

    # (b) Sequential order: a confirmed busy mark makes the steal refuse.
    holder = _session_dict(key)
    server._sessions["holder-live"] = holder
    assert server._ensure_active_session_slot("holder-live", holder) is None
    before = json.dumps(_entry_for(None, key), sort_keys=True)
    thief = _session_dict(key)
    server._sessions["thief-live"] = thief
    result = server._ensure_active_session_slot("thief-live", thief)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert json.dumps(_entry_for(None, key), sort_keys=True) == before
    server._sessions.clear()


def test_compute_host_drain_is_gated(gateway, monkeypatch):
    """The compute-host queued-drain dispatch (no _run_prompt_submit, hence no admit
    gate) must mark busy_kind='user' through the same epoch-fenced revalidate — and a
    session displaced mid-queue drops the envelope instead of streaming on a stolen
    session."""
    key = "sess-compute-drain"
    session = _session_dict(key)
    session["queued_prompt"] = {"text": "queued hello", "transport": None}
    server._sessions["drain-live"] = session
    assert server._ensure_active_session_slot("drain-live", session) is None
    server._lease_turn_settled(session)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *a, **k: True)

    dispatched: list[dict] = []

    def _fake_submit(rid, sid, sess, text, **kwargs):
        dispatched.append({
            "entry": dict(_entry_for(None, key)),
            "running": bool(sess.get("running"))})
        return {"result": {"status": "streaming"}}

    monkeypatch.setattr(server, "_submit_prompt_to_compute_host", _fake_submit)
    assert server._drain_queued_prompt("rid", "drain-live", session) is True
    assert dispatched, "queued compute-host prompt must dispatch through the gate"
    assert dispatched[0]["entry"].get("busy") is True
    assert dispatched[0]["entry"].get("busy_kind") == "user"
    assert dispatched[0]["running"] is True

    # Displaced mid-queue: the isolated turn ends (its done-callback settles the lease),
    # then a thief steals, the envelope re-queues, and the drain must not dispatch.
    session["running"] = False
    server._lease_turn_settled(session)
    server._sessions.pop("drain-live", None)  # keep the live watcher out of the race
    thief = _session_dict(key)
    server._sessions["thief-live"] = thief
    assert server._ensure_active_session_slot("thief-live", thief) is None
    session["queued_prompt"] = {"text": "queued again", "transport": None}
    session["running"] = False
    dispatched.clear()
    emits: list = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emits.append((event, sid)))
    assert server._drain_queued_prompt("rid", "drain-live", session) is True
    assert dispatched == [], "a displaced session must not dispatch to the compute host"
    assert any(event == "error" for event, _sid in emits)
    assert session.get("running") is False


# ── Flag fidelity: every running=True setter is admission-gated ────────────


def test_running_setter_canary():
    """Source scan: every `session["running"] = True` site must sit inside a gated
    admission path. A future setter outside the gate publishes an idle-looking
    streaming turn — the exact double-writer hole the gate exists to close."""
    repo_root = Path(__file__).resolve().parents[2]
    gated = {
        ("methods_prompt.py", "_lock_in_submit_turn"),          # marks busy post-history_lock release
        ("prompt_turn.py", "_run_post_turn_followups"),         # dispatches via _run_prompt_submit (admit gate)
        ("session_auto_continue.py", "_maybe_schedule_auto_continue"),  # admission + pre-dispatch re-check
        ("session_auto_continue.py", "kickoff"),  # nested body of the above; same gate + re-check
        ("session_auto_continue.py", "_drain_queued_prompt"),   # inline → _run_prompt_submit; compute → _lease_admission_check
        ("session_notifications.py", "_notif_claim_turn"),      # all submits go through _run_prompt_submit
    }
    for filename in (
        "methods_prompt.py", "prompt_turn.py", "session_auto_continue.py",
        "session_notifications.py", "compute_host_bridge.py",
    ):
        source = (repo_root / "tui_gateway" / filename).read_text()
        current_fn = None
        for lineno, line in enumerate(source.splitlines(), 1):
            match = re.match(r"^(\s*)def (\w+)", line)
            if match:
                current_fn = match.group(2)
            if re.search(r"session\[.running.\]\s*=\s*True", line):
                assert (filename, current_fn) in gated, (
                    f"{filename}:{lineno} sets running=True inside {current_fn!r}, "
                    "which is not a known gated admission path — route it through "
                    "_lease_admission_check/_run_prompt_submit or extend this canary "
                    "deliberately")


# ── Busy clears in the turn finally; followups re-mark ─────────────────────


def test_busy_clears_in_turn_finally_and_followups_remark(gateway, monkeypatch):
    """The registry shows idle in the inter-turn gap (cleared beside running=False,
    BEFORE followups) and a followup dispatch re-marks busy_kind='user' at its own
    admission — a steal in the gap breaks the chain instead of stranding the phone."""
    source = Path(server.__file__).with_name("prompt_turn.py").read_text()
    finally_pos = source.index("session[\"running\"] = False", source.index("def run():"))
    settled_pos = source.index("_lease_turn_settled", finally_pos)
    finished_pos = source.index("tui turn finished", finally_pos)
    followups_pos = source.index("_run_post_turn_followups(rid, sid", finally_pos)
    assert finally_pos < settled_pos < finished_pos < followups_pos, (
        "busy must clear beside running=False, before the finished log and followups")

    key = "sess-finally-clear"
    session = _session_dict(key)
    server._sessions["live"] = session
    assert server._ensure_active_session_slot("live", session) is None
    assert _entry_for(None, key)["busy"] is True

    # Turn settles: busy cleared beside running=False.
    session["running"] = False
    server._lease_turn_settled(session)
    entry = _entry_for(None, key)
    assert entry.get("busy") is False and "busy_kind" not in entry

    # Followup admission re-marks busy='user' at its own gate.
    assert server._lease_admission_check("live", session) is None
    entry = _entry_for(None, key)
    assert entry["busy"] is True and entry["busy_kind"] == "user"


def test_invisible_chain_never_permanently_unyieldable(gateway, monkeypatch):
    """C2's core promise: chained followups re-marking busy=user with FRESH activity
    are never stolen; the moment activity goes stale past STALL_ACTIVITY_S the next
    claim steals; an ended chain leaves an idle gap that steals cleanly; a displaced
    chain's next admission is fenced. The phone's wait is bounded by the threshold,
    not by the chain."""
    key = "sess-chain"
    session = _session_dict(key)
    server._sessions["chain-live"] = session
    assert server._ensure_active_session_slot("chain-live", session) is None

    # Fresh-activity chain: N followups never yield.
    for _ in range(3):
        server._lease_admission_check("chain-live", session)
        thief_probe = _session_dict(key)
        server._sessions["probe"] = thief_probe
        result = server._ensure_active_session_slot("probe", thief_probe)
        server._sessions.pop("probe", None)
        assert getattr(result, "reason", None) == AS.SESSION_BUSY

    # Activity goes stale past the stall threshold → the next claim steals.
    _mutate_entry(None, key, lambda e: e.update(activity_at=time.time() - (AS.HERMES_LEASE_STALL_ACTIVITY_S + 1)))
    thief = _session_dict(key)
    server._sessions["thief-live"] = thief
    assert server._ensure_active_session_slot("thief-live", thief) is None

    # The displaced chain's next admission is fenced (no unbounded double-writer).
    result = server._lease_admission_check("chain-live", session)
    assert result is not None and getattr(result, "reason", None) == AS.SESSION_DISPLACED

    # Inter-turn idle gap steals cleanly: settle + steal is immediate.
    key2 = "sess-chain-gap"
    session2 = _session_dict(key2)
    server._sessions["gap-live"] = session2
    assert server._ensure_active_session_slot("gap-live", session2) is None
    server._lease_turn_settled(session2)
    thief2 = _session_dict(key2)
    server._sessions["gap-thief"] = thief2
    t0 = time.monotonic()
    assert server._ensure_active_session_slot("gap-thief", thief2) is None
    assert time.monotonic() - t0 < 2.5


# ── Turn markers: conditional delete + steal-path force-clear ──────────────


def test_turn_marker_conditional_delete_and_steal_clear(gateway, monkeypatch, tmp_path):
    home = tmp_path / "marker-home"

    # (a) A writer-identified clear cannot delete a DIFFERENT writer's marker.
    record_turn_start(home, "k", "prompt-a", writer={"lease_id": "lease-a", "epoch": 1})
    record_turn_start(home, "k", "prompt-a", writer={"lease_id": "lease-a", "epoch": 1})
    clear_turn_marker(home, "k", writer={"lease_id": "lease-b", "epoch": 2})
    marker = read_turn_marker(home, "k")
    assert marker is not None, "identity mismatch must no-op, not delete"
    clear_turn_marker(home, "k", writer={"lease_id": "lease-a", "epoch": 1})
    assert read_turn_marker(home, "k") is None

    # (b) The steal path's force-clear removes a FOREIGN marker (the 01:32:10
    # double-submit hazard: the resuming surface must not auto-continue the
    # displaced surface's in-flight prompt).
    record_turn_start(home, "k2", "prompt-b", writer={"lease_id": "lease-old", "epoch": 4})
    clear_turn_marker(home, "k2", force=True)
    assert read_turn_marker(home, "k2") is None

    # A legacy identity-less marker still clears for its owner (old markers never wedge).
    record_turn_start(home, "k3", "prompt-c")
    clear_turn_marker(home, "k3", writer={"lease_id": "lease-x", "epoch": 1})
    assert read_turn_marker(home, "k3") is None

    # (c) The auto-continue kickoff re-check: admission followed by a vanished or
    # replaced marker (or a bumped epoch) bails instead of dispatching.
    key = "sess-marker-kickoff"
    session = _session_dict(key)
    server._sessions["kick-live"] = session
    assert server._ensure_active_session_slot("kick-live", session) is None
    monkeypatch.setattr(server, "_session_home", lambda sess: home)
    record_turn_start(
        home, key, "interrupted prompt", writer={
            "lease_id": session["active_session_lease"].lease_id,
            "epoch": session["active_session_lease"].epoch})
    marker = read_turn_marker(home, key)
    assert marker is not None and marker.get("lease_id")

    # Marker vanished (cleared by a steal) → bail.
    clear_turn_marker(home, key, force=True)
    assert server._auto_continue_still_valid("kick-live", session, home, key, marker) is False
    # Marker replaced under us → bail.
    record_turn_start(home, key, "a different prompt", writer={"lease_id": "other", "epoch": 9})
    assert server._auto_continue_still_valid("kick-live", session, home, key, marker) is False
    # Marker intact + lease epoch unchanged → proceed.
    clear_turn_marker(home, key, force=True)
    record_turn_start(
        home, key, "interrupted prompt", writer={
            "lease_id": session["active_session_lease"].lease_id,
            "epoch": session["active_session_lease"].epoch})
    marker = read_turn_marker(home, key)
    assert server._auto_continue_still_valid("kick-live", session, home, key, marker) is True


# ── transfer never resurrects after a steal ────────────────────────────────


def test_transfer_does_not_resurrect_after_steal(tmp_path):
    home = tmp_path / "transfer-home"
    victim, refusal = AS.try_acquire_active_session(
        session_id="sess-transfer", surface="desktop", config={}, registry_home=home,
        track_liveness=True, metadata={"live_session_id": "victim"})
    assert refusal is None
    thief, refusal = AS.try_acquire_active_session(
        session_id="sess-transfer", surface="webui", config={}, registry_home=home,
        metadata={"live_session_id": "thief"})
    assert thief is not None, f"steal must succeed: {refusal}"

    # The displaced holder must not resurrect its lease onto the stolen session.
    assert AS.transfer_active_session(victim, session_id="sess-transfer") is False
    entries = _registry_entries(home)
    assert [e["lease_id"] for e in entries] == [thief.lease_id], "no registry append"

    # Mixed-commit pin: an epoch-LESS old-code owner must still win the guard.
    _mutate_entry(home, "sess-transfer", lambda e: e.pop("epoch", None))
    assert AS.transfer_active_session(victim, session_id="sess-transfer") is False
    assert len(_registry_entries(home)) == 1


# ── Registry skew: unknown fields survive old-code rewrites ────────────────


def test_unknown_fields_survive_old_code_rewrite(tmp_path):
    home = tmp_path / "skew-home"
    lease, refusal = AS.try_acquire_active_session(
        session_id="sess-skew", surface="desktop", config={}, registry_home=home)
    assert refusal is None
    state = AS._state_path(home)
    data = json.loads(state.read_text())
    new_fields = {k: v for k, v in data["entries"][0].items()
                  if k in ("epoch", "busy", "busy_kind", "busy_detail", "busy_since", "activity_at", "heartbeat_at")}
    assert "epoch" in new_fields and "heartbeat_at" in new_fields
    lease.release()

    # Simulated OLD writer: read → mutate its own known field → write the list back.
    data["entries"][0]["updated_at"] = time.time()
    data["entries"].append({
        "lease_id": "old-code-lease", "session_id": "sess-old", "surface": "cli",
        "pid": os.getpid(), "process_start_time": None,
        "started_at": time.time(), "updated_at": time.time()})
    state.write_text(json.dumps({"entries": data["entries"]}))

    kept = _entry_for(home, "sess-skew")
    for field, value in new_fields.items():
        assert kept.get(field) == value, f"{field} must round-trip verbatim through old writers"
    snapshot = AS.active_session_registry_snapshot(registry_home=home)
    assert any(e.get("session_id") == "sess-old" for e in snapshot)


# ── Old-requester compat honor path (repaired requeue) ─────────────────────


def test_old_requester_compat_honor_path(gateway, tmp_path):
    """A NEW holder honors old-code request files: idle closes fast; busy requeues
    with the ORIGINAL requested_at (bounded TTL) and its holder incarnation; expired
    and corrupt files are unlinked; fresh foreign files and fresh future-protocol
    own-pid files survive; an EXPIRED future-protocol file is unlinked."""
    home = tmp_path / "compat-home"
    req_dir = AS._yield_request_dir(home)
    req_dir.mkdir(parents=True, exist_ok=True)
    mine = {"pid": os.getpid(), "process_start_time": None}

    busy = _session_dict("sess-compat-busy")
    busy["running"] = True
    gateway._sessions["tab-busy"] = busy
    busy_lease, refusal = server._claim_active_session_slot(
        "sess-compat-busy", live_session_id="tab-busy", surface="desktop", profile_home=str(home))
    assert refusal is None, f"busy tab claim failed: {refusal}"
    busy["active_session_lease"] = busy_lease
    original_mint = time.time()
    assert AS.request_cross_surface_yield("sess-compat-busy", mine, registry_home=home)
    polled = AS.poll_yield_requests(registry_home=home)
    assert len(polled) == 1 and polled[0]["session_id"] == "sess-compat-busy"
    assert polled[0].get("protocol") == 2, "new requests carry a protocol version"

    server._yield_session_for_request(home, polled[0])
    assert "tab-busy" in gateway._sessions, "busy session must survive"
    requeued = AS.poll_yield_requests(registry_home=home)
    assert len(requeued) == 1 and requeued[0]["session_id"] == "sess-compat-busy"
    assert requeued[0]["requested_at"] == pytest.approx(original_mint, abs=2.0), (
        "requeue must preserve the ORIGINAL requested_at so the TTL stays bounded")
    assert requeued[0].get("holder_process_start_time") is not None, (
        "requeue must preserve the holder incarnation (987d564112 defect)")
    AS.poll_yield_requests(registry_home=home)  # consume the requeue

    # Idle session: honored immediately.
    idle = _session_dict("sess-compat-idle")
    gateway._sessions["tab-idle"] = idle
    idle_lease, refusal = server._claim_active_session_slot(
        "sess-compat-idle", live_session_id="tab-idle", surface="desktop", profile_home=str(home))
    assert refusal is None, f"idle tab claim failed: {refusal}"
    idle["active_session_lease"] = idle_lease
    server._lease_turn_settled(idle)
    assert AS.request_cross_surface_yield("sess-compat-idle", mine, registry_home=home)
    polled = AS.poll_yield_requests(registry_home=home)
    assert len(polled) == 1 and polled[0]["session_id"] == "sess-compat-idle"
    server._yield_session_for_request(home, polled[0])
    assert "tab-idle" not in gateway._sessions

    # Expired, corrupt, fresh-foreign, and future-protocol files.
    (req_dir / "expired.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": os.getpid(), "requested_at": time.time() - 9999}))
    (req_dir / "corrupt.json").write_text("{not json")
    (req_dir / "foreign.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": 424242, "requested_at": time.time()}))
    (req_dir / "future-fresh.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": os.getpid(), "protocol": 99, "requested_at": time.time()}))
    (req_dir / "future-expired.json").write_text(json.dumps({
        "session_id": "x", "holder_pid": os.getpid(), "protocol": 99,
        "requested_at": time.time() - 9999}))
    assert AS.poll_yield_requests(registry_home=home) == []
    assert not (req_dir / "expired.json").exists()
    assert not (req_dir / "corrupt.json").exists()
    assert not (req_dir / "future-expired.json").exists(), (
        "expired future-protocol files are unlinked (bounded future-file leakage)")
    assert (req_dir / "foreign.json").exists(), "fresh foreign files are left for their owner"
    assert (req_dir / "future-fresh.json").exists(), (
        "fresh own-pid future-protocol files survive for a future-version reader")


def test_requester_retry_policy(gateway, monkeypatch):
    """SESSION_BUSY with busy_kind='auto' inside grace: bounded 250ms retries, then a
    steal once the grace lapses (total well under tonight's ~9s). A user-fresh busy
    holder returns immediately with no retry burn."""
    from hermes_cli.active_sessions import try_acquire_active_session

    claims: list[float] = []
    real_claim = server._claim_active_session_slot

    def counting_claim(*args, **kwargs):
        claims.append(time.monotonic())
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(server, "_claim_active_session_slot", counting_claim)
    monkeypatch.setattr(AS, "HERMES_LEASE_AUTO_BUSY_GRACE_S", 1.0)

    key = "sess-retry-auto"
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, metadata={"live_session_id": "holder"})
    assert refusal is None
    _mutate_entry(None, key, lambda e: e.update(
        busy=True, busy_kind="auto", busy_detail="bg_review",
        busy_since=time.time() - 0.8, heartbeat_at=time.time()))

    relay = _session_dict(key)
    server._sessions["relay-live"] = relay
    t0 = time.monotonic()
    assert server._ensure_active_session_slot("relay-live", relay) is None, (
        "past-grace auto busy must be stolen by a bounded retry")
    elapsed = time.monotonic() - t0
    assert elapsed <= 2.5, f"bounded retry policy took {elapsed:.2f}s"
    assert len(claims) >= 2, "auto-busy inside grace must retry, not give up"

    key = "sess-retry-user"
    claims.clear()
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="desktop", config={}, metadata={"live_session_id": "holder"})
    assert refusal is None
    _mutate_entry(None, key, lambda e: e.update(
        busy=True, busy_kind="user", activity_at=time.time(), heartbeat_at=time.time()))
    relay = _session_dict(key)
    server._sessions["relay-2"] = relay
    t0 = time.monotonic()
    result = server._ensure_active_session_slot("relay-2", relay)
    elapsed = time.monotonic() - t0
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert elapsed < 0.5, f"user-fresh refusal must be immediate, took {elapsed:.2f}s"
    assert len(claims) == 1, "no retry burn against a user-fresh holder"


# ── Watcher loop observability ─────────────────────────────────────────────


def test_watcher_loop_logs_and_survives_exception(monkeypatch, caplog):
    """A raising tick is logged (WARNING + exc_info — never silently suppressed, the
    exact mechanism that hid the original NameError) and the next good tick still
    runs; the 60s alive-line contract exists in source."""
    ran_good: list[int] = []
    blew = {"once": False}

    def bad_then_good():
        if not blew["once"]:
            blew["once"] = True
            raise RuntimeError("tick blew up once")
        ran_good.append(1)

    state: dict = {}
    with caplog.at_level(logging.WARNING, logger="tui_gateway.session_reaper"):
        session_reaper._run_lease_tick_safely(bad_then_good, state)
    assert any("lease maintenance tick failed" in r.message for r in caplog.records)
    assert any(r.exc_info for r in caplog.records), "the failure must carry exc_info"

    session_reaper._run_lease_tick_safely(bad_then_good, state)  # survives; good tick runs
    assert ran_good == [1]

    import ast
    source = Path(session_reaper.__file__).read_text()
    assert "lease maintenance alive" in source, "60s alive-line contract must exist"
    module_ast = ast.parse(source)
    loop_src = next(
        ast.get_source_segment(source, node) for node in ast.walk(module_ast)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_lease_maintenance_watcher")
    assert "contextlib.suppress" not in loop_src, (
        "the watcher loop must not blanket-suppress tick failures")


def test_refusal_message_truthfulness():
    """The phone renders the refusal string verbatim: each state names what the
    desktop is actually doing — and an idle holder with a lagging watcher is never
    called 'mid-turn'."""
    now = time.time()
    fresh_user = {"surface": "desktop", "pid": 1, "started_at": now - 5, "busy": True,
                  "busy_kind": "user", "activity_at": now - 1, "heartbeat_at": now}
    stale_user = dict(fresh_user, activity_at=now - 40)
    auto = {"surface": "desktop", "pid": 1, "started_at": now - 5, "busy": True,
            "busy_kind": "auto", "busy_detail": "bg_review", "busy_since": now - 30,
            "heartbeat_at": now}
    legacy = {"surface": "desktop", "pid": 1, "started_at": now - 5}
    lagging = {"surface": "desktop", "pid": 1, "started_at": now - 5, "heartbeat_at": now - 20}

    msg = AS.session_already_owned_message("s", fresh_user)
    assert "mid-turn" in msg and "no visible progress" not in msg
    msg = AS.session_already_owned_message("s", stale_user)
    assert "no visible progress" in msg and "mid-turn ~" not in msg
    msg = AS.session_already_owned_message("s", auto)
    assert "background review" in msg
    msg = AS.session_already_owned_message("s", legacy)
    assert "restart once" in msg
    msg = AS.session_already_owned_message("s", lagging)
    assert "lease stale" in msg and "mid-turn" not in msg


# ── Review-fix round: confirmed defects from the adversarial review ────────


def test_bg_review_token_survives_turn_settle_transition(gateway, monkeypatch):
    """The REAL user→auto handover (not a pre-planted auto entry): a review spawned
    mid-turn takes its token, the turn's finally CONVERTS the live 'user' mark to
    'auto' in one critical section (fresh busy_since, no frozen activity clock), a
    claim during the review is grace-refused, and the review's completion clears it."""
    import types as _types
    from agent.background_review import _lease_auto_busy_begin, _lease_auto_busy_end

    key = "sess-bg-transition"
    session = _session_dict(key)
    session["agent"] = _types.SimpleNamespace()  # hook target
    server._sessions["live"] = session
    assert server._ensure_active_session_slot("live", session) is None
    turn_started = time.time()

    # Review spawns mid-turn: token registered, foreground turn keeps the 'user' mark.
    _lease_auto_busy_begin(session["agent"])
    entry = _entry_for(None, key)
    assert entry["busy"] is True and entry["busy_kind"] == "user", (
        "a live turn is never downgraded to auto")

    # The turn's finally: the mark CONVERTS (the old guard blocked both ends of this
    # handover, stranding 'user' with a frozen activity clock for the whole review).
    session["running"] = False
    server._lease_turn_settled(session)
    entry = _entry_for(None, key)
    assert entry["busy_kind"] == "auto" and entry["busy_detail"] == "bg_review"
    assert entry["busy_since"] >= turn_started, "the auto grace must measure from the settle"
    assert "activity_at" not in entry, "the frozen user-activity clock must not leak into auto"

    # A claim during the review is grace-refused (never stall-stolen at 61s).
    monkeypatch.setattr(server, "_LEASE_BUSY_RETRY_WINDOW_S", 0.0)  # no retry burn in-test
    probe = _session_dict(key)
    server._sessions["probe"] = probe
    result = server._ensure_active_session_slot("probe", probe)
    server._sessions.pop("probe", None)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY
    assert "background review" in str(result)

    # Review completes: token cleared, entry idle again.
    _lease_auto_busy_end(session["agent"])
    assert _entry_for(None, key).get("busy") is False


def test_tick_interrupt_failure_still_closes(gateway, monkeypatch, caplog):
    """An interrupt that raises during the displaced close must never skip the close
    (the ws-orphan reaper contract): the lease is gone either way, and a skipped close
    would re-fire every 1.5s tick forever."""
    key = "sess-tick-interrupt-fails"
    session = _session_dict(key, running=True)
    server._sessions["victim"] = session
    assert server._ensure_active_session_slot("victim", session) is None
    server._lease_turn_settled(session)

    thief = _session_dict(key)
    thief["profile_home"] = None
    server._sessions["thief"] = thief
    assert server._ensure_active_session_slot("thief", thief) is None

    def _boom(*_a, **_k):
        raise RuntimeError("interrupt exploded")

    monkeypatch.setattr(server, "_interrupt_session_turn", _boom)
    with caplog.at_level(logging.WARNING, logger="tui_gateway.server"):
        server._lease_maintenance_tick()
    assert "victim" not in server._sessions, "close must run despite the interrupt failure"
    assert any("closing anyway" in r.message for r in caplog.records)


def test_gateway_turn_busy_mark_and_activity_piggyback(gateway):
    """A gateway messaging turn claims with busy_kind='user' (its lease lives outside
    the tui_gateway session registry — nothing else marks or refreshes it): fresh
    activity refuses a steal, stale-past-stall steals. Plus the wiring canaries."""
    from hermes_cli.active_sessions import try_acquire_active_session

    key = "sess-gateway-turn"
    lease, refusal = try_acquire_active_session(
        session_id=key, surface="gateway:telegram", config={}, mark_busy=True,
        metadata={"live_session_id": "gateway-holder", "platform": "telegram"})
    assert refusal is None
    entry = _entry_for(None, key)
    assert entry["busy"] is True and entry["busy_kind"] == "user"

    phone = _session_dict(key)
    server._sessions["phone"] = phone
    result = server._ensure_active_session_slot("phone", phone)
    server._sessions.pop("phone", None)
    assert getattr(result, "reason", None) == AS.SESSION_BUSY, (
        "a streaming gateway turn must not be stolen while its activity is fresh")

    _mutate_entry(None, key, lambda e: e.update(activity_at=time.time() - (AS.HERMES_LEASE_STALL_ACTIVITY_S + 5)))
    phone = _session_dict(key)
    server._sessions["phone2"] = phone
    assert server._ensure_active_session_slot("phone2", phone) is None
    assert _entry_for(None, key)["pid"] == os.getpid()

    # Wiring canaries: the claim marks busy, and the turn runner piggybacks activity on
    # the streaming-delta and tool-progress callbacks (both directions verified above
    # live; these pin that the wiring cannot silently regress).
    repo_root = Path(__file__).resolve().parents[2]
    assert "mark_busy=True" in (repo_root / "gateway" / "run_busy.py").read_text()
    runner_src = (repo_root / "gateway" / "run_turn_runner.py").read_text()
    assert "touch_active_session_lease_activity" in runner_src
    assert "self.touch_active_session_lease_activity()" in runner_src


def test_followup_dispatch_false_releases_not_delivers(gateway, monkeypatch):
    """_run_prompt_submit's False (not admitted — displaced/closing) is a FAILURE, not
    a silent success: on_error runs (delivery bookkeeping must not mark work done),
    on_done never does, and no phantom message.start precedes the refusal."""
    emits: list = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emits.append(event))
    key = "sess-followup-dispatch"
    session = _session_dict(key)
    server._sessions["holder"] = session
    assert server._ensure_active_session_slot("holder", session) is None
    server._lease_turn_settled(session)

    thief = _session_dict(key)
    server._sessions["thief"] = thief
    assert server._ensure_active_session_slot("thief", thief) is None  # holder displaced

    calls: list[str] = []
    server._dispatch_followup_turn(
        "rid", "holder", session, "goal continuation", "goal continuation dispatch",
        on_done=lambda: calls.append("done"), on_error=lambda: calls.append("error"))
    assert calls == ["error"], "a refused dispatch must run on_error, never on_done"
    assert "message.start" not in emits, (
        "no turn bubble may be emitted before admission; the refusal emits an error frame")
    assert "error" in emits
    assert session.get("running") is False


def test_submit_early_refusal_unwinds_busy_mark(gateway):
    """Every prompt.submit refusal between the slot gate's busy mark and the turn start
    (here: 4004 confirm_truncate without a cut) must UNPUBLISH busy='user' — only a
    turn thread's finally ever clears it otherwise."""
    key = "sess-submit-early-refusal"
    session = _session_dict(key)
    server._sessions["live"] = session
    assert server._ensure_active_session_slot("live", session) is None
    assert _entry_for(None, key)["busy"] is True

    err, _fields = server._lock_in_submit_turn(
        "rid", "live", session, "hello", {"confirm_truncate": True}, False, None, None)
    assert err is not None and err["error"]["code"] == 4004
    entry = _entry_for(None, key)
    assert entry.get("busy") is False, "a refused submit must unwind the busy mark"
    assert session.get("running") is False


def test_transfer_resurrect_preserves_epoch(tmp_path):
    """A resurrected entry must carry the lease object's epoch: minting epoch=1 under a
    lease that holds >1 self-displaces on the very next revalidate."""
    home = tmp_path / "transfer-epoch"
    victim, _ = AS.try_acquire_active_session(
        session_id="sess-te", surface="desktop", config={}, registry_home=home,
        track_liveness=True, metadata={"live_session_id": "victim"})
    thief, refusal = AS.try_acquire_active_session(
        session_id="sess-te", surface="webui", config={}, registry_home=home,
        track_liveness=True, metadata={"live_session_id": "thief"})
    assert thief is not None and thief.epoch == 2

    # Simulate the thief's entry vanishing (prune race): the resurrect must re-mint at
    # the LEASE's epoch, not 1.
    state = AS._state_path(home)
    with AS._FileLock(AS._lock_path(home)):
        entries = [e for e in json.loads(state.read_text())["entries"]
                   if e.get("lease_id") != thief.lease_id]
        AS._write_entries(state, entries)
    assert AS.transfer_active_session(thief, session_id="sess-te-2") is True
    resurrected = _entry_for(home, "sess-te-2")
    assert resurrected["epoch"] == thief.epoch
    refreshed, refusal = AS.revalidate_active_session(thief)
    assert refreshed is thief and refusal is None, "the resurrected lease must not self-displace"


def test_admission_check_fail_closed_on_registry_error(gateway, monkeypatch):
    """A revalidate that RAISES (flock unavailable, ENOSPC) inside the running=True
    window surfaces as a fail-closed refusal — never an exception wedging the session."""
    key = "sess-admission-raise"
    session = _session_dict(key)
    server._sessions["live"] = session
    assert server._ensure_active_session_slot("live", session) is None

    def _raise(*_a, **_k):
        raise OSError("no file handles left")

    monkeypatch.setattr(AS, "revalidate_active_session", _raise)
    result = server._lease_admission_check("live", session)
    assert result is not None
    assert getattr(result, "reason", None) == AS.SESSION_COORDINATION_UNAVAILABLE


def test_skip_persist_covers_displaced_compute_host_turn(gateway, monkeypatch):
    """A displaced COMPUTE-HOST session (running=True, no local run thread — the
    interrupt deliberately leaves the flag for a child callback that never comes) must
    get the same skip-persist protection as a live local thread."""
    import types as _types

    def _make_session():
        persist: list[int] = []
        agent = _types.SimpleNamespace(
            session_id="sess-ch-persist",
            _session_messages=[{"role": "user", "content": "x"}],
            _persist_session=lambda snap: persist.append(1))
        sess = _session_dict("sess-ch-persist", source="webui")
        sess["agent"] = agent
        sess["running"] = True
        sess["_compute_host_active"] = True
        sess["_closing"] = True  # _pop_session_by_id already claimed it
        sess["_persist_probe"] = persist
        return sess

    import contextlib as _cl

    @_cl.contextmanager
    def _no_db(_session):
        yield None

    monkeypatch.setattr(server, "_session_db", _no_db)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)

    displaced = _make_session()
    server._teardown_popped_session(displaced, end_reason="lease_preempted")
    assert displaced["_persist_probe"] == [], (
        "a displaced compute-host turn must not persist over the new owner's row")

    control = _make_session()
    server._teardown_popped_session(control, end_reason="tui_close")
    assert control["_persist_probe"] == [1], "an ordinary close still persists"


def test_compute_host_relay_refreshes_parent_activity(gateway, monkeypatch):
    """Isolated compute-host turns stream in the child; the PARENT holds the lease and
    this relay sees every child frame — it must keep the parent's activity clock fresh
    (verified: the child claims no lease, so nothing else touches the parent entry)."""
    monkeypatch.setattr(server, "write_json", lambda message: True)
    key = "sess-ch-relay"
    session = _session_dict(key)
    server._sessions["ch-live"] = session
    assert server._ensure_active_session_slot("ch-live", session) is None
    _mutate_entry(None, key, lambda e: e.update(activity_at=time.time() - 9999))

    server._relay_compute_host_rpc({
        "type": "rpc", "params": {"type": "message.delta", "session_id": "ch-live", "text": "chunk"}})
    assert _entry_for(None, key)["activity_at"] >= time.time() - 2.0, (
        "a streamed child frame must refresh the parent lease's activity clock")


def test_review_fix_source_contracts():
    """Small wiring contracts from the review round: the retired write-only
    _lease_busy_mark_pending flag is gone, and the module-default watcher fallback is
    actually pinned (register's write-back runs before the server def exists)."""
    repo_root = Path(__file__).resolve().parents[2]
    methods_src = (repo_root / "tui_gateway" / "methods_prompt.py").read_text()
    assert "_lease_busy_mark_pending" not in methods_src
    server_src = (repo_root / "tui_gateway" / "server.py").read_text()
    assert "_session_reaper._lease_maintenance_tick = _lease_maintenance_tick" in server_src
