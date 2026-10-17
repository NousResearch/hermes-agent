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
import time
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_cli import active_sessions
from hermes_cli.active_sessions import (
    SESSION_COORDINATION_UNAVAILABLE,
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
    """No ambient Desktop identity may leak into a test."""
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
    metadata: dict = {"live_session_id": f"live-{pid}"}
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


def leased_session(
    home: Path, key: str, *, surface: str = "tui", lease_id: str = "tui-lease-1",
    enabled: bool = True, released: bool = False,
) -> dict:
    """A live session record that still CACHES the lease it claimed.

    Lease identity is (pid, live_session_id) in the registry, and the cached token is what the
    turn-admission fast path used to trust blindly: its ``lease_id`` is exactly what a superseding
    claim replaces. ``enabled=False`` is the inert borrowed token a compute-host child adopts
    (``_install_borrowed_lease``); ``released=True`` is a token whose row is already gone.
    """
    session = {
        "session_key": key,
        "source": surface,
        "profile_home": str(home),
        "history_lock": threading.Lock(),
        "running": False,
    }
    session["active_session_lease"] = ActiveSessionLease(
        lease_id=lease_id, session_id=key, surface=surface, enabled=enabled, released=released,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )
    return session


def pin_turn_isolation(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Pin ``dashboard.turn_isolation`` for ``_holder_turns_may_run_isolated``."""
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config",
                        lambda cfg=None: {"turn_isolation": enabled})


def write_turn_marker(home: Path, key: str, pid: int, *, age: float = 0.0) -> None:
    """The durable turn marker a running backend leaves (``<home>/desktop/interrupted_turns.json``).

    ``age`` days the progress stamp: 0 is a turn that emitted progress a moment ago, a large value is
    a HUNG turn — the one case a takeover may preempt.
    """
    path = home / "desktop" / "interrupted_turns.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.time() - age
    path.write_text(json.dumps({key: {
        "attempts": 0, "prompt": "a turn that is still running", "started_at": stamp,
        "progress_at": stamp, "auto_continue": False, "writer_pid": pid,
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


def test_another_desktop_instance_takes_the_chat_over_when_the_holder_is_idle(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device switch mints a DIFFERENT app instance; that must not fence the same person out."""
    home = tmp_path / "home"
    key = "chat-2"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_B, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    assert server._ensure_active_session_slot("sid-new", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]


def test_a_chat_handed_to_an_idle_client_is_taken_back_at_once(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a message sent just after the other client's turn ended must not stand down.

    The session still holds the token it lost to the sibling (its lease id is gone from the registry,
    so the cached-lease guard reads it as superseded). The holder has no turn marker, so the registry
    proves it idle and the takeover must succeed at once instead of "another client owns the chat now".
    """
    home = tmp_path / "home"
    key = "chat-stand-down"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_B, attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)
    session["active_session_lease"] = ActiveSessionLease(
        lease_id="our-revoked-lease", session_id=key, surface="desktop", track_liveness=True,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )

    assert server._ensure_active_session_slot("sid-stand-down", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]


def test_a_holder_still_busy_keeps_the_chat_and_we_stand_down(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Counterpart: a holder whose turn is still progressing is not taken over; we stand down."""
    home = tmp_path / "home"
    key = "chat-stand-down-busy"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, instance=INSTANCE_B, attached=False)])
    write_turn_marker(home, key, foreign_pid, age=0.0)
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)
    session["active_session_lease"] = ActiveSessionLease(
        lease_id="our-revoked-lease", session_id=key, surface="desktop", track_liveness=True,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )

    refusal = server._ensure_active_session_slot("sid-stand-down-busy", session)
    assert refusal is not None
    assert [e["pid"] for e in read_ledger(home)] == [foreign_pid]


def test_a_holder_without_any_desktop_identity_is_taken_over_too(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The registry is per HERMES_HOME, so an identity-less holder is still the same owner."""
    home = tmp_path / "home"
    key = "chat-3"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="desktop", attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    assert server._ensure_active_session_slot("sid-new", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]
def test_a_tui_or_cli_holder_is_taken_over_as_well(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI/TUI session is just another interactive client of this owner."""
    home = tmp_path / "home"
    key = "chat-4"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="tui", attached=False)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    assert server._ensure_active_session_slot("sid-new", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]
@pytest.mark.parametrize("attached", [True, None])
def test_a_live_client_on_the_holder_does_not_block_the_handoff(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch, attached: object
) -> None:
    """The chat is still OPEN on the old computer — that is the case this feature exists for."""
    home = tmp_path / "home"
    key = "chat-5"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=attached)])
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)

    assert server._ensure_active_session_slot("sid-new", session) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]
def test_a_runtime_without_a_desktop_identity_can_take_the_chat_over(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claimant may be the dashboard or a CLI: the owner is the HOME, not the app instance."""
    home = tmp_path / "home"
    key = "chat-6"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=False)])

    assert server._ensure_active_session_slot("sid-new", desktop_session(home, key)) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]
def test_a_turn_that_is_still_progressing_keeps_the_chat(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A holder that keeps emitting progress keeps its lease: the peer's turn is not interrupted."""
    home = tmp_path / "home"
    key = "chat-9"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=False)])
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


def test_a_hung_turn_hands_the_chat_over(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case "I walked away and that backend hung": no progress for the threshold -> take over."""
    home = tmp_path / "home"
    key = "chat-9b"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=True)])
    write_turn_marker(home, key, foreign_pid, age=4 * 3600)  # 4h of silence, threshold is 900s
    set_desktop_instance(monkeypatch, INSTANCE_A)

    assert server._ensure_active_session_slot("sid-new", desktop_session(home, key)) is None
    assert [e["pid"] for e in read_ledger(home)] == [os.getpid()]
def test_a_fresh_marker_written_by_anybody_blocks_the_handoff(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whoever wrote it, fresh progress on THIS chat means somebody is working in it."""
    home = tmp_path / "home"
    key = "chat-10"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=False)])
    write_turn_marker(home, key, os.getpid())
    set_desktop_instance(monkeypatch, INSTANCE_A)

    refusal = server._ensure_active_session_slot("sid-new", desktop_session(home, key))
    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED

    # ... and a marker that stopped progressing is exactly the hung case again.
    write_turn_marker(home, key, os.getpid(), age=4 * 3600)
    assert server._ensure_active_session_slot("sid-new", desktop_session(home, key)) is None
def test_the_predicate_refuses_everything_it_cannot_prove_idle(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    set_desktop_instance(monkeypatch, INSTANCE_A)
    predicate = lambda holder: server._holder_may_be_superseded(holder, home=home)
    eligible = holder_entry("k", pid=foreign_pid, attached=True)  # a client may still be attached
    assert predicate(eligible) is True

    # A gateway/bot delivery is never stolen by a window.
    assert predicate(holder_entry("k", pid=foreign_pid, surface="gateway")) is False
    # Our own pid is the in-process takeover's decision, never this one's.
    assert predicate(holder_entry("k", pid=os.getpid())) is False
    # A holder whose pid is not even readable is never superseded.
    assert predicate({**eligible, "pid": "not-a-pid"}) is False
    # Fresh progress on the chat refuses; silence past the threshold allows.
    write_turn_marker(home, "k", foreign_pid)
    assert predicate(eligible) is False
    write_turn_marker(home, "k", foreign_pid, age=4 * 3600)
    assert predicate(eligible) is True
def test_lease_metadata_keeps_the_historical_shape_for_existing_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert server._lease_metadata("live-1") == {"live_session_id": "live-1"}
    set_desktop_instance(monkeypatch, INSTANCE_A)
    desktop = server._lease_metadata("live-2")
    assert desktop["live_session_id"] == "live-2"
    assert desktop["desktop_instance_id"] == INSTANCE_A
    assert desktop["client_attached"] is True  # a claim happens on a user turn

    # A claim from a non-Desktop runtime round-trips exactly the historical shape.
    tui_home = tmp_path / "tui-home"
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")
    tui_lease, message = server._claim_active_session_slot(
        "k", live_session_id="live-4", surface="tui", profile_home=tui_home)
    assert tui_lease is not None and message is None
    assert read_ledger(tui_home)[0]["metadata"] == {"live_session_id": "live-4"}
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


def test_a_superseded_runtime_stands_down_instead_of_reclaiming(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-claiming would start a tug-of-war with a peer whose turn is still running.

    The peer here is mid-turn (a fresh progress marker), so the registry does NOT prove it idle and
    the superseded runtime must stand down rather than reclaim. An IDLE peer is taken back at once
    instead; see ``test_a_chat_handed_to_an_idle_client_is_taken_back_at_once``.
    """
    home = tmp_path / "home"
    key = "chat-8"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, attached=False)])
    write_turn_marker(home, key, foreign_pid, age=0.0)
    set_desktop_instance(monkeypatch, INSTANCE_A)
    session = desktop_session(home, key)
    # A token this process still believes it owns, already replaced in the registry by a peer.
    session["active_session_lease"] = ActiveSessionLease(
        lease_id="superseded-lease", session_id=key, surface="desktop", track_liveness=True,
        state_path=_state_path(home), lock_path=home / "runtime" / "active_sessions.lock",
    )
    session["_lease_client_attached"] = True

    refusal = server._ensure_active_session_slot("sid-local", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert "active_session_lease" not in session
    # The peer's entry is untouched: no second writer, no fight.
    assert [e["lease_id"] for e in read_ledger(home)] == ["holder-lease-1"]
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


# ---------------------------------------------------------------------------------------------------
# D1: the supersede asymmetry. ``_holder_may_be_superseded`` lets a Desktop atomically take a provably
# idle TUI/CLI lease, but the turn-admission resolver only consulted the registry for a DESKTOP
# surface, so a superseded TUI kept serving turns under a lease id the registry had already replaced.
# Every enabled, unreleased cached token is now re-verified against the registry it was claimed in,
# whatever the surface; an unreadable registry fails CLOSED instead of reading "no proof" as "mine".
# ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["tui", "cli"])
def test_a_superseded_tui_or_cli_never_serves_its_stale_lease(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    """D1 regression: a Desktop took this chat over; the TUI's next prompt may not run under its token.

    The Desktop is mid-turn, so it is not idle and the TUI must stand down with the same refusal a
    Desktop gets — not take the ``return None`` fast path that only looked at Desktop surfaces.
    (RED before the fix: the respawned TUI served a turn beside the live Desktop, two writers.)
    """
    home = tmp_path / "home"
    key = f"chat-d1-{surface}"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="desktop", instance=INSTANCE_B,
                                     attached=True, lease_id="desktop-lease-2")])
    write_turn_marker(home, key, foreign_pid, age=0.0)
    session = leased_session(home, key, surface=surface)

    refusal = server._ensure_active_session_slot("sid-tui-late", session)

    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert "active_session_lease" not in session
    assert [e["lease_id"] for e in read_ledger(home)] == ["desktop-lease-2"], "the holder's row is untouched"


def test_a_superseded_tui_takes_the_chat_back_once_the_desktop_is_idle(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of D1: an IDLE holder hands the chat back, with a FRESH lease id.

    Same rule as the Desktop handback: the registry must prove the current holder is idle. Serving
    the old token instead would be a second writer; refusing forever would fence the user out of
    the window they are typing in.
    """
    home = tmp_path / "home"
    key = "chat-d1-idle"
    write_ledger(home, [holder_entry(key, pid=foreign_pid, surface="desktop", instance=INSTANCE_B,
                                     attached=True, lease_id="desktop-lease-2")])
    pin_turn_isolation(monkeypatch, False)  # no marker: the Desktop's turn concluded
    session = leased_session(home, key, surface="tui")

    assert server._ensure_active_session_slot("sid-tui-late", session) is None

    lease = session["active_session_lease"]
    assert lease.lease_id != "tui-lease-1", "a fresh claim, never the superseded token"
    entry, = read_ledger(home)
    assert entry["lease_id"] == lease.lease_id and entry["pid"] == os.getpid()
    lease.release()


def test_a_tui_whose_registry_row_vanished_reclaims_instead_of_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached token whose row is GONE (reaped, dropped) is not a licence to run lease-less."""
    home = tmp_path / "home"
    key = "chat-d1-row-gone"
    write_ledger(home, [holder_entry("another-chat", pid=os.getpid(), attached=True,
                                     lease_id="another-lease")])
    session = leased_session(home, key, surface="tui")

    assert server._ensure_active_session_slot("sid-tui-late", session) is None

    rows = {entry["session_id"]: entry for entry in read_ledger(home)}
    lease = session["active_session_lease"]
    assert lease.lease_id != "tui-lease-1"
    assert rows[key]["lease_id"] == lease.lease_id and rows[key]["pid"] == os.getpid()
    assert rows["another-chat"]["lease_id"] == "another-lease", "another owner's row is never rewritten"
    lease.release()


@pytest.mark.parametrize("surface", ["desktop", "tui"])
def test_an_unreadable_registry_refuses_a_cached_lease_instead_of_serving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    """"Cannot read the registry" is not proof the cached token is still ours: refuse, touch nothing.

    For a Desktop this is a deliberate tightening of the old "unprovable ownership keeps today's
    behaviour" — the same fail-closed reading ``try_acquire_active_session`` already gives a missing
    proof, and never a licence to run a second writer.
    """
    home = tmp_path / "home"
    state = _state_path(home)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{ this is not the registry", encoding="utf-8")
    before = state.read_bytes()
    if surface == "desktop":
        set_desktop_instance(monkeypatch, INSTANCE_A)
    session = leased_session(home, "chat-d1-unreadable", surface=surface)
    held = session["active_session_lease"]

    refusal = server._ensure_active_session_slot("sid-tui-late", session)

    assert refusal is not None
    assert getattr(refusal, "reason", None) == SESSION_COORDINATION_UNAVAILABLE
    assert state.read_bytes() == before, "a refusal must not rewrite a registry it could not read"
    assert session["active_session_lease"] is held, "neither dropped nor re-claimed"


@pytest.mark.parametrize("surface", ["desktop", "tui"])
def test_a_current_cached_lease_keeps_serving_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    """Invariant 2: the registry still holds this exact lease id -> the turn runs, nothing is rewritten."""
    home = tmp_path / "home"
    key = "chat-still-current"
    if surface == "desktop":
        set_desktop_instance(monkeypatch, INSTANCE_A)
    session = leased_session(home, key, surface=surface)
    held = session["active_session_lease"]
    write_ledger(home, [holder_entry(key, pid=os.getpid(), surface=surface, attached=True,
                                     lease_id=held.lease_id)])
    before = read_ledger(home)

    assert server._ensure_active_session_slot("sid-current", session) is None
    assert session["active_session_lease"] is held
    assert read_ledger(home) == before


@pytest.mark.parametrize("surface", ["desktop", "tui"])
def test_an_inert_borrowed_lease_keeps_its_parent_owned_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str,
) -> None:
    """Invariant 3: the compute-host child's ``enabled=False`` token has no row of its own to verify.

    The PARENT owns the registry slot; the child must neither claim a second one nor be fenced out
    by the parent's own entry — and it must not even read the registry (no lock/state file appears).
    """
    home = tmp_path / "home"
    session = leased_session(home, "chat-borrowed", surface=surface,
                             lease_id="borrowed:parent-lease", enabled=False)
    held = session["active_session_lease"]

    assert server._ensure_active_session_slot("sid-child", session) is None

    assert session["active_session_lease"] is held
    assert not (home / "runtime").exists(), "the inert path stays out of the registry entirely"


def test_a_released_token_keeps_its_existing_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Invariant 3: ``release()`` already dropped the row and every releaser pops the token.

    A lingering released token therefore keeps the historical no-op reading — it is not re-claimed
    here (that is the claim path's job) and the guard does not read the registry for it.
    """
    home = tmp_path / "home"
    session = leased_session(home, "chat-released", surface="tui", released=True)
    held = session["active_session_lease"]

    assert server._ensure_active_session_slot("sid-released", session) is None
    assert session["active_session_lease"] is held
    assert not (home / "runtime").exists()
