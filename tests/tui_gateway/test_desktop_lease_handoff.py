"""Same-Desktop cross-process lease handoff (Desktop ``SESSION_NOT_OWNED`` after a reconnect).

The Desktop dials a NEW ``hermes serve --isolated`` backend per connection, so after a gateway
restart / renderer reload / SSH reconnect the chat is leased by a DIFFERENT pid of the SAME
Desktop app instance. ``_take_over_detached_runtime_lease`` can only inspect this process's
``_sessions``, so the reconnect was refused ``SESSION_NOT_OWNED`` until the old backend exited
(never idle-reaped by design) — the user's own chat fenced against the user.

These tests pin the replacement: a runtime may take over a lease held by another runtime of the
SAME Desktop instance that has published it has NO live client for the chat, and every other
refusal (another Desktop instance, a foreign pid, another surface, a live client, an unreadable
marker) still stands.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import active_sessions
from hermes_cli.active_sessions import (
    SESSION_NOT_OWNED,
    ActiveSessionLease,
    _state_path,
    _write_entries,
    active_session_registry_snapshot,
)
from tui_gateway import server


INSTANCE_A = "a" * 32
INSTANCE_B = "b" * 32


@pytest.fixture(autouse=True)
def _no_ambient_desktop_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither the module global nor the env may leak a Desktop identity into a test."""
    monkeypatch.delenv("HERMES_DESKTOP_INSTANCE_ID", raising=False)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")


@pytest.fixture
def foreign_pid() -> int:
    """A REAL live pid that is not this process's (an unprovable-liveness pid would be pruned)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait(timeout=30)


def set_desktop_instance(monkeypatch: pytest.MonkeyPatch, instance: str) -> None:
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", instance)


def holder_entry(
    key: str, *, pid: int, surface: str = "desktop", instance: str = "", attached: object = None,
    lease_id: str = "holder-lease-1",
) -> dict:
    """A lease entry exactly as a sibling Desktop backend would have written it."""
    metadata: dict = {"live_session_id": f"live-{pid}", "bot_live_delivery_consumer": True}
    if instance:
        metadata["desktop_instance_id"] = instance
    if attached is not None:
        metadata["client_attached"] = attached
    return {
        "lease_id": lease_id,
        "session_id": key,
        "surface": surface,
        "pid": pid,
        "process_start_time": None,
        "started_at": 1_000.0,
        "updated_at": 1_000.0,
        "track_liveness": True,
        "metadata": metadata,
    }


def write_ledger(home: Path, entries: list[dict]) -> None:
    _write_entries(_state_path(home), entries)


def read_ledger(home: Path) -> list[dict]:
    return json.loads(_state_path(home).read_text(encoding="utf-8"))["entries"]


def desktop_session(home: Path, key: str) -> dict:
    return {
        "session_key": key,
        "source": "desktop",
        "profile_home": str(home),
        "history_lock": threading.Lock(),
        "running": False,
    }


def write_turn_marker(home: Path, key: str, pid: int) -> None:
    """The durable turn marker a mid-turn backend leaves (``<home>/desktop/interrupted_turns.json``)."""
    path = home / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: {
        "attempts": 0, "prompt": "a turn that is still running", "started_at": 9999.0,
        "auto_continue": False, "writer_pid": pid,
    }}), encoding="utf-8")


def test_second_runtime_of_the_same_desktop_takes_the_chat_over(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    key = "chat-1"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    assert server._ensure_active_session_slot("sid-new", session) is None

    entries = active_session_registry_snapshot(registry_home=home)
    assert [entry["session_id"] for entry in entries] == [key]
    entry = entries[0]
    assert entry["pid"] == os.getpid(), "the reconnecting backend owns the lease now"
    assert entry["metadata"]["desktop_instance_id"] == INSTANCE_A
    lease = session["active_session_lease"]
    assert lease is not None
    assert lease.lease_id == entry["lease_id"]
    assert lease.lease_id != "holder-lease-1", "a FRESH lease id supersedes the old token"

    # The superseded runtime still holds its old token and releases it on its own finalize; that
    # release must never drop the lease the new owner now holds (same lease id = that bug).
    stale = ActiveSessionLease(
        lease_id="holder-lease-1", session_id=key, surface="desktop", track_liveness=True,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )
    stale.release()
    assert [e["lease_id"] for e in read_ledger(home)] == [lease.lease_id]


def test_another_desktop_instance_still_gets_session_not_owned(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    key = "chat-2"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_B, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    refusal = server._ensure_active_session_slot("sid-new", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert "active_session_lease" not in session
    assert [e["lease_id"] for e in read_ledger(home)] == ["holder-lease-1"]


def test_foreign_live_pid_without_a_desktop_identity_still_gets_session_not_owned(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    key = "chat-3"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="desktop", attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    refusal = server._ensure_active_session_slot("sid-new", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert [e["pid"] for e in read_ledger(home)] == [foreign_pid]


def test_a_cli_or_tui_holder_of_the_same_id_still_gets_session_not_owned(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a *Desktop* lease is readable as a superseded one; a TUI/CLI owner is a real one."""
    home = tmp_path / "home"
    key = "chat-4"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="tui", instance=INSTANCE_A, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    refusal = server._ensure_active_session_slot("sid-new", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert [e["pid"] for e in read_ledger(home)] == [foreign_pid]


@pytest.mark.parametrize("attached", [True, None])
def test_no_takeover_while_the_holder_still_has_a_live_client(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch, attached: object
) -> None:
    """``True`` (a client is attached) and None (the holder never said) both refuse."""
    home = tmp_path / "home"
    key = "chat-5"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=attached)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    refusal = server._ensure_active_session_slot("sid-new", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert [e["lease_id"] for e in read_ledger(home)] == ["holder-lease-1"]


def test_a_non_desktop_runtime_never_supersedes_anything(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No identity (TUI/CLI/gateway) means no handoff, even with a perfectly eligible holder."""
    home = tmp_path / "home"
    key = "chat-6"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=False)])

    refusal = server._ensure_active_session_slot("sid-new", desktop_session(home, key))

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED


def test_no_takeover_while_the_holder_is_mid_turn(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder with no client but a turn in flight keeps its lease: the peer cannot be interrupted."""
    home = tmp_path / "home"
    key = "chat-9"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=False)])
    write_turn_marker(home, key, foreign_pid)
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    refusal = server._ensure_active_session_slot("sid-new", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert [e["pid"] for e in read_ledger(home)] == [foreign_pid]

    # Its turn concludes (the marker is cleared on any conclusion) -> the handoff is allowed.
    (home / "desktop" / "interrupted_turns.json").unlink()
    assert server._ensure_active_session_slot("sid-new", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]


def test_a_turn_marker_written_by_somebody_else_does_not_block_the_handoff(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    key = "chat-10"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=False)])
    write_turn_marker(home, key, os.getpid())  # this process's own, already-concluded turn marker
    set_desktop_instance(monkeypatch, INSTANCE_A)

    assert server._ensure_active_session_slot("sid-new", desktop_session(home, key)) is None


def test_the_supersede_predicate_refuses_every_other_holder_shape(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    set_desktop_instance(monkeypatch, INSTANCE_A)
    predicate = lambda holder: server._same_desktop_holder_may_be_superseded(holder, home=home)
    eligible = holder_entry("k", pid=foreign_pid, instance=INSTANCE_A, attached=False)
    assert predicate(eligible) is True

    assert predicate(holder_entry("k", pid=foreign_pid, instance=INSTANCE_B, attached=False)) is False
    assert predicate(holder_entry("k", pid=foreign_pid, surface="tui", instance=INSTANCE_A, attached=False)) is False
    assert predicate(holder_entry("k", pid=foreign_pid, instance=INSTANCE_A, attached=True)) is False
    assert predicate(holder_entry("k", pid=foreign_pid)) is False
    # Our own pid is the in-process takeover's decision, never this one's.
    assert predicate(holder_entry("k", pid=os.getpid(), instance=INSTANCE_A, attached=False)) is False
    # A holder whose pid is not even readable is never superseded.
    assert predicate({**eligible, "pid": "not-a-pid"}) is False
    # An in-flight turn on the holder refuses even with the client gone.
    write_turn_marker(home, "k", foreign_pid)
    assert predicate(eligible) is False


def test_lease_metadata_keeps_the_historical_shape_for_existing_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert server._lease_metadata("live-1") == {
        "live_session_id": "live-1", "bot_live_delivery_consumer": True,
    }
    set_desktop_instance(monkeypatch, INSTANCE_A)
    desktop = server._lease_metadata("live-2")
    assert desktop["live_session_id"] == "live-2"
    assert desktop["bot_live_delivery_consumer"] is True
    assert desktop["desktop_instance_id"] == INSTANCE_A
    assert desktop["client_attached"] is True  # a claim happens on a user turn

    # A claim from a non-Desktop runtime round-trips exactly the historical shape.
    tui_home = tmp_path / "tui-home"
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")
    tui_lease, message = server._claim_active_session_slot(
        "k", live_session_id="live-4", surface="tui", profile_home=tui_home)
    assert tui_lease is not None and message is None
    assert read_ledger(tui_home)[0]["metadata"] == {
        "live_session_id": "live-4", "bot_live_delivery_consumer": True,
    }
    tui_lease.release()


def test_the_holder_publishes_its_client_transport_state_on_transitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    set_desktop_instance(monkeypatch, INSTANCE_A)
    lease, message = server._claim_active_session_slot("chat-7", live_session_id="sid-owner",
                                                      profile_home=home)
    assert lease is not None and message is None
    session = desktop_session(home, "chat-7")
    session["active_session_lease"] = lease
    session["_lease_client_attached"] = True
    assert read_ledger(home)[0]["metadata"]["client_attached"] is True

    server._sync_lease_client_attachment(session, False)
    assert read_ledger(home)[0]["metadata"]["client_attached"] is False
    assert session["_lease_client_attached"] is False

    server._sync_lease_client_attachment(session, True)
    assert read_ledger(home)[0]["metadata"]["client_attached"] is True
    # The other metadata keys are untouched by the marker write.
    assert read_ledger(home)[0]["metadata"]["live_session_id"] == "sid-owner"
    lease.release()


def test_a_superseded_runtime_reclaims_instead_of_serving_under_its_stale_token(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The peer cannot be told; instead its OWN next turn re-checks the registry (fail closed)."""
    home = tmp_path / "home"
    key = "chat-8"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_A, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)
    # A token this process still believes it owns, already replaced in the registry by a peer.
    session["active_session_lease"] = ActiveSessionLease(
        lease_id="superseded-lease", session_id=key, surface="desktop", track_liveness=True,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )
    session["_lease_client_attached"] = True

    assert server._ensure_active_session_slot("sid-local", session) is None
    assert session["active_session_lease"].lease_id != "superseded-lease"


def test_the_desktop_instance_identity_is_read_off_the_ssh_lock_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import web_server

    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")
    web_server._apply_ssh_desktop_identity(tmp_path / INSTANCE_A / "backend.lock.json")
    assert active_sessions.desktop_instance_id() == INSTANCE_A
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")

    web_server._apply_ssh_desktop_identity(tmp_path / "not-an-ownership-id" / "backend.lock.json")
    assert active_sessions.desktop_instance_id() == "", "a malformed slot name never becomes an identity"
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")
    web_server._apply_ssh_desktop_identity(None)
    assert active_sessions.desktop_instance_id() == ""
