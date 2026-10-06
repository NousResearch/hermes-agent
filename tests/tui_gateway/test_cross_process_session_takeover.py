"""allow_session_takeover across backend PROCESSES that share one HERMES_HOME.

Two backends commonly serve one home: an always-on ``hermes serve`` plus the
Desktop's own child, or an SSH ``--isolated`` serve. With takeover enabled, the
second backend replaces the first one's registry entry, but nothing tells the
first process: its runtime still holds the lease object in memory, and
``_ensure_active_session_slot`` used to wave every later turn through on that
dict check alone. Both processes then wrote one stored session from two
snapshots, which is the concurrent writer the registry exists to prevent.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    active_session_lease_is_current,
    active_session_registry_snapshot,
    try_acquire_active_session,
)
from tui_gateway import server

_TAKEOVER_HOLDER_SCRIPT = """
import os
import time
from pathlib import Path

from hermes_cli.active_sessions import try_acquire_active_session

lease, message = try_acquire_active_session(
    session_id=os.environ["SESSION_ID"],
    surface="tui",
    config={"allow_session_takeover": True},
    metadata={"live_session_id": "other-backend"},
)
assert lease is not None and message is None, message
Path(os.environ["READY_FILE"]).write_text(lease.lease_id, encoding="utf-8")
deadline = time.monotonic() + 120
while not Path(os.environ["RELEASE_FILE"]).exists() and time.monotonic() < deadline:
    time.sleep(0.02)
lease.release()
"""


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _take_over_from_another_process(home: Path, session_id: str, tmp_path: Path):
    """Start a second backend process that takes ``session_id`` over and keeps it."""
    ready, release = tmp_path / "ready", tmp_path / "release"
    repo_root = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_TOKEN"))}
    env.update({
        "HERMES_HOME": str(home),
        "PYTHONPATH": os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p),
        "SESSION_ID": session_id, "READY_FILE": str(ready), "RELEASE_FILE": str(release),
    })
    child = subprocess.Popen([sys.executable, "-c", _TAKEOVER_HOLDER_SCRIPT], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 60
    while not ready.exists():
        if child.poll() is not None:
            out, err = child.communicate()
            pytest.fail(f"takeover process exited early\nstdout: {out}\nstderr: {err}")
        if time.monotonic() >= deadline:
            pytest.fail("timed out waiting for the takeover process")
        time.sleep(0.02)

    def stop():
        release.touch()
        try:
            child.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.communicate()

    return ready.read_text(encoding="utf-8"), stop


def _session(key: str) -> dict:
    return {"session_key": key, "source": "desktop", "history": [{"role": "user", "content": "stale"}],
            "history_lock": threading.Lock(), "history_version": 3}


def _leases_for(session_id: str) -> list[str]:
    return [e["lease_id"] for e in active_session_registry_snapshot() if e["session_id"] == session_id]


def test_lease_currency_reports_a_lease_another_owner_replaced(home):
    config = {"allow_session_takeover": True}
    mine, _ = try_acquire_active_session(
        session_id="S", surface="desktop", config=config, metadata={"live_session_id": "a"})
    assert active_session_lease_is_current(mine) is True
    try_acquire_active_session(
        session_id="S", surface="tui", config=config, metadata={"live_session_id": "b"})
    assert active_session_lease_is_current(mine) is False


def test_a_taken_over_runtime_reloads_and_reclaims_instead_of_writing_beside_the_new_owner(
        home, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"allow_session_takeover": True})
    durable = [{"role": "user", "content": "from the phone"}, {"role": "assistant", "content": "ok"}]
    monkeypatch.setattr(server, "_load_durable_truncation_history", lambda session, *a, **k: list(durable))
    session = _session("chat-1")
    assert server._ensure_active_session_slot("live-a", session) is None
    first = session["active_session_lease"].lease_id

    other, stop = _take_over_from_another_process(home, "chat-1", tmp_path)
    try:
        assert _leases_for("chat-1") == [other]
        # The next turn here must notice the handover instead of passing the dict check.
        assert server._ensure_active_session_slot("live-a", session) is None
        held = session["active_session_lease"].lease_id
        assert held != first
        assert _leases_for("chat-1") == [held], "exactly one owner, and it is the runtime that sent last"
        assert session["history"] == durable, "the turn must continue from what the other backend stored"
        assert session["history_version"] == 4
    finally:
        stop()


def test_without_takeover_here_the_handed_over_runtime_stops_writing(home, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_load_durable_truncation_history", lambda session, *a, **k: [])
    session = _session("chat-2")
    assert server._ensure_active_session_slot("live-a", session) is None

    other, stop = _take_over_from_another_process(home, "chat-2", tmp_path)
    try:
        refusal = server._ensure_active_session_slot("live-a", session)
        assert getattr(refusal, "reason", None) == SESSION_NOT_OWNED
        assert session.get("active_session_lease") is None
        assert _leases_for("chat-2") == [other]
    finally:
        stop()


def test_an_unreadable_transcript_refuses_the_turn_rather_than_replaying_a_stale_snapshot(
        home, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"allow_session_takeover": True})
    monkeypatch.setattr(server, "_load_durable_truncation_history", lambda session, *a, **k: None)
    session = _session("chat-3")
    assert server._ensure_active_session_slot("live-a", session) is None

    _other, stop = _take_over_from_another_process(home, "chat-3", tmp_path)
    try:
        assert server._ensure_active_session_slot("live-a", session) is not None
        assert session.get("active_session_lease") is None
        assert session["history"] == [{"role": "user", "content": "stale"}]
        assert _leases_for("chat-3") == [], "the refused runtime must not keep the lease it reclaimed"
    finally:
        stop()


def test_an_owner_that_was_not_taken_over_keeps_its_lease_and_history(home, monkeypatch):
    monkeypatch.setattr(server, "_load_cfg", lambda: {"allow_session_takeover": True})
    monkeypatch.setattr(server, "_load_durable_truncation_history",
                        lambda session, *a, **k: pytest.fail("no handover, no reload"))
    session = _session("chat-4")
    assert server._ensure_active_session_slot("live-a", session) is None
    lease = session["active_session_lease"]
    assert server._ensure_active_session_slot("live-a", session) is None
    assert session["active_session_lease"] is lease
    assert session["history_version"] == 3
