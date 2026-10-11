"""The Desktop lease takeover must not read a MISSING durable turn signal as "holder idle".

``session_lifecycle._holder_turn_in_flight`` gates the same-Desktop cross-process handoff on the
durable turn marker (``tui_gateway.turn_marker``). That marker used to be written only for turns
carrying non-empty prompt TEXT, so a holder whose turn was still running could leave NO marker and
be superseded by a sibling backend of the same Desktop instance — two writers on one stored session.
The other two holes were an UNREADABLE marker file (read as "idle") and an isolated compute-host
turn (``dashboard.turn_isolation``), whose marker is written by the CHILD pid.

These tests pin the hardened contract:

* every turn that can run records a marker, blank prompt included, and the crash auto-continue
  contract (``read_turn_marker``) still never continues a blank marker;
* "unreadable marker file" is distinguishable from "no marker" and fails CLOSED;
* an isolated profile (compute-host turns) refuses the handoff when no marker names the holder;
* the verified behaviour that a concluded/non-holder marker does not block a handoff stands.
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
    _state_path,
    _write_entries,
)
from tui_gateway import prompt_turn, server
from tui_gateway.turn_marker import (
    MarkerUnreadable,
    read_turn_marker,
    read_turn_marker_entry,
    record_turn_start,
    turn_in_flight,
)


INSTANCE_A = "a" * 32


@pytest.fixture(autouse=True)
def _no_ambient_desktop_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_DESKTOP_INSTANCE_ID", raising=False)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", "")


@pytest.fixture
def foreign_pid() -> int:
    """A REAL live pid that is not this process's (the lease holder's)."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child.pid
    finally:
        child.kill()
        child.wait(timeout=30)


def _holder_entry(key: str, pid: int, *, attached: bool | None = False) -> dict:
    metadata: dict = {
        "live_session_id": f"live-{pid}",
        "bot_live_delivery_consumer": True,
        "desktop_instance_id": INSTANCE_A,
    }
    if attached is not None:
        metadata["client_attached"] = attached
    return {
        "lease_id": "holder-lease-1", "session_id": key, "surface": "desktop", "pid": pid,
        "process_start_time": None, "started_at": 1_000.0, "updated_at": 1_000.0,
        "track_liveness": True, "metadata": metadata,
    }


def _desktop_session(home: Path, key: str) -> dict:
    return {"session_key": key, "source": "desktop", "profile_home": str(home),
            "history_lock": threading.Lock(), "running": False}


def _marker_path(home: Path) -> Path:
    return home / "desktop" / "interrupted_turns.json"


def _write_marker(home: Path, key: str, *, writer_pid: int, prompt: str) -> Path:
    """Exactly the entry ``record_turn_start`` leaves, for a marker written by a chosen pid."""
    path = _marker_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({key: {
        "attempts": 0, "prompt": prompt, "started_at": 1_000.0, "auto_continue": False,
        "writer_pid": writer_pid,
    }}), encoding="utf-8")
    return path


def _isolate(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Pin ``dashboard.turn_isolation`` for ``_holder_turns_may_run_isolated``."""
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config",
                        lambda cfg=None: {"turn_isolation": enabled})


def test_a_blank_prompt_turn_still_records_a_marker_for_its_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Premise fix: a text-less turn records a marker, and it is still never auto-continued."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    # ``_session_home``/``record_turn_start`` are bound into prompt_turn's globals by
    # ``server.register`` at runtime; bind them the same way for a bare module import.
    monkeypatch.setattr(prompt_turn, "_session_home", lambda s: Path(s["profile_home"]), raising=False)
    monkeypatch.setattr(prompt_turn, "record_turn_start", record_turn_start, raising=False)
    session = {"session_key": "chat-guard-1", "profile_home": str(home), "history_lock": threading.Lock()}

    prompt_turn._record_turn_marker(session, "")  # what _run_prompt_submit passes for a text-less send

    entry = read_turn_marker_entry(home, "chat-guard-1")
    assert isinstance(entry, dict), "a blank-text turn must leave cross-process proof it is running"
    assert entry["prompt"] == ""
    assert entry["writer_pid"] == os.getpid()
    assert turn_in_flight(home, "chat-guard-1", writer_pid=os.getpid()) is True

    # The crash auto-continue contract is unchanged: nothing to re-submit -> None.
    assert read_turn_marker(home, "chat-guard-1") is None

    # A non-string prompt (structured content instead of text) is recorded as blank, not refused.
    prompt_turn._record_turn_marker(session, {"type": "image"})
    assert read_turn_marker_entry(home, "chat-guard-1")["prompt"] == ""


def test_a_running_blank_text_turn_keeps_the_holder_fenced(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) holder with client_attached: False + a running blank-text turn -> SESSION_NOT_OWNED."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-2", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])
    _write_marker(home, "chat-guard-2", writer_pid=foreign_pid, prompt="")  # the blank-text turn
    _isolate(monkeypatch, False)
    session = _desktop_session(home, "chat-guard-2")

    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is False
    refusal = server._ensure_active_session_slot("sid-guard", session)
    assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    assert [entry["pid"] for entry in json.loads(_state_path(home).read_text())["entries"]] == [foreign_pid]

    # Its turn concludes (the marker is cleared on any conclusion) -> the handoff is allowed again.
    _marker_path(home).unlink()
    assert server._ensure_active_session_slot("sid-guard", session) is None


def test_an_unreadable_marker_file_fails_closed(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) the file exists but cannot be read -> refuse, never read it as "no turn in flight"."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-3", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])
    marker = _write_marker(home, "chat-guard-3", writer_pid=foreign_pid, prompt="a turn that runs")
    marker.chmod(0o000)
    try:
        # Distinguishable from "no marker" for a fail-closed caller...
        with pytest.raises(MarkerUnreadable):
            read_turn_marker_entry(home, "chat-guard-3")
        # ...while the best-effort auto-continue reader stays best-effort.
        assert read_turn_marker(home, "chat-guard-3") is None

        assert server._same_desktop_holder_may_be_superseded(holder, home=home) is False
        refusal = server._ensure_active_session_slot("sid-guard", _desktop_session(home, "chat-guard-3"))
        assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
        assert [e["pid"] for e in json.loads(_state_path(home).read_text())["entries"]] == [foreign_pid]
    finally:
        marker.chmod(0o600)


def test_an_isolated_turn_of_the_holder_refuses_the_handoff(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) with turn isolation the marker names the compute-host CHILD, not the holder."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-4", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])
    _isolate(monkeypatch, True)
    session = _desktop_session(home, "chat-guard-4")

    # The child is a live process that is not the holder and not us.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        _write_marker(home, "chat-guard-4", writer_pid=child.pid, prompt="isolated turn")
        assert server._same_desktop_holder_may_be_superseded(holder, home=home) is False
        refusal = server._ensure_active_session_slot("sid-guard", session)
        assert refusal is not None and refusal.reason == SESSION_NOT_OWNED
    finally:
        child.kill()
        child.wait(timeout=30)


def test_an_isolated_profile_refuses_when_no_marker_is_readable(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) with isolation on, "no marker" cannot prove the holder is idle -> refuse."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-5", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])

    _isolate(monkeypatch, True)
    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is False

    # Isolation off (the default, and this machine): the same holder with no marker hands over.
    _isolate(monkeypatch, False)
    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is True


def test_a_holder_dead_writer_still_hands_the_chat_over(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Control: the idle path is untouched — a concluded/dead turn still allows the handoff."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-6", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])
    _isolate(monkeypatch, False)

    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is True
    assert server._ensure_active_session_slot("sid-guard", _desktop_session(home, "chat-guard-6")) is None

    # A marker naming the holder whose writer is provably gone is not a turn in flight.
    _write_marker(home, "chat-guard-6", writer_pid=8_000_000_000, prompt="crashed turn")
    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is True


def test_a_marker_written_by_somebody_else_does_not_block_without_isolation(
    tmp_path: Path, foreign_pid: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified behaviour kept: an unattributable marker is not proof of THE HOLDER's turn."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    holder = _holder_entry("chat-guard-7", foreign_pid, attached=False)
    _write_entries(_state_path(home), [holder])
    _isolate(monkeypatch, False)
    _write_marker(home, "chat-guard-7", writer_pid=os.getpid(), prompt="this process's own marker")

    assert server._same_desktop_holder_may_be_superseded(holder, home=home) is True


def test_a_reattached_client_wins_over_the_stale_detach_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detach publish re-checks for a live transport, so a resume in the window is not fenced."""
    home = tmp_path / "home"
    home.mkdir(parents=True)
    monkeypatch.setattr(active_sessions, "_DESKTOP_INSTANCE_ID", INSTANCE_A)
    lease, message = server._claim_active_session_slot(
        "chat-guard-8", live_session_id="sid-owner", profile_home=home)
    assert lease is not None and message is None
    session = _desktop_session(home, "chat-guard-8")
    session["active_session_lease"] = lease
    session["_lease_client_attached"] = True

    # A session.resume rebinds a live client between the detach and the registry write.
    monkeypatch.setattr(server, "_session_has_live_transport", lambda s, **kw: True)
    server._sync_lease_client_attachment(session, False)
    assert json.loads(_state_path(home).read_text())["entries"][0]["metadata"]["client_attached"] is True
    assert session["_lease_client_attached"] is True

    # With no live transport the publish is honoured as before.
    monkeypatch.setattr(server, "_session_has_live_transport", lambda s, **kw: False)
    server._sync_lease_client_attachment(session, False)
    assert json.loads(_state_path(home).read_text())["entries"][0]["metadata"]["client_attached"] is False
    assert session["_lease_client_attached"] is False
    lease.release()
