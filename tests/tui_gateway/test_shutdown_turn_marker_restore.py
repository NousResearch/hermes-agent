"""A graceful shutdown must leave the crash markers of the turns it interrupts.

The turn marker contract (turn_marker.py) is "written at turn start, cleared on any
conclusion" — only a process death leaves one behind. A SIGTERM/atexit shutdown
interrupted every running turn and let the turn's own ``finally`` retire the marker,
so an operational restart (config change, image upgrade, node drain) left
``session.resume`` no crash evidence and the in-flight work silently dropped
(#135286). ``_stop_turns_before_exit`` now snapshots each running turn's marker
before the interrupt and restores it after the settle window: a shutdown interrupt
counts as a process death for marker purposes, and the existing freshness /
attempts / writer-liveness guards bound the recovery.
"""

from __future__ import annotations

import threading

import pytest

from tui_gateway import server
from tui_gateway.turn_marker import (
    clear_turn_marker,
    read_turn_marker,
    record_turn_start,
)


@pytest.fixture()
def marker_home(monkeypatch, tmp_path):
    """Point the server's marker storage at a temp HERMES_HOME."""
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    return tmp_path


@pytest.fixture()
def live_session(marker_home):
    """Register one running session; always deregister on exit."""
    session = {
        "session_key": "shutdown-restore-key",
        "running": True,
        "history_lock": threading.Lock(),
        "_active_turn_marker_key": "shutdown-restore-key",
    }
    with server._sessions_lock:
        server._sessions["sess-shutdown-restore"] = session
    try:
        yield session
    finally:
        with server._sessions_lock:
            server._sessions.pop("sess-shutdown-restore", None)


def _patch_shutdown_turn_paths(monkeypatch, home, *, kill_raises=False):
    """Stub the interrupt (retiring the marker, as the turn thread's finally does) and the
    foreground-kill sweep."""
    interrupted: list[str] = []

    def _fake_interrupt(sid, session):
        key = str(
            session.get("_active_turn_marker_key") or session.get("session_key") or ""
        )
        clear_turn_marker(home, key)
        interrupted.append(sid)

    def _fake_kill(now=False):
        if kill_raises:
            raise RuntimeError("kill sweep failed")

    monkeypatch.setattr(server, "_interrupt_session_turn", _fake_interrupt)
    monkeypatch.setattr(
        "tools.environments.base.kill_live_foreground_processes", _fake_kill
    )
    return interrupted


def test_shutdown_interrupt_restores_the_turn_marker(
    live_session, marker_home, monkeypatch
):
    """The settle-window retire (turn finally) must not outlive the shutdown: the snapshot lands
    afterwards, with the original prompt and attempt count intact."""
    record_turn_start(
        marker_home,
        "shutdown-restore-key",
        "restart the gateway, then check the board",
        attempts=1,
    )
    interrupted = _patch_shutdown_turn_paths(monkeypatch, marker_home)

    server._stop_turns_before_exit(budget_s=0.05)

    assert interrupted == ["sess-shutdown-restore"]
    marker = read_turn_marker(marker_home, "shutdown-restore-key")
    assert marker is not None, "a shutdown interrupt must leave crash evidence behind"
    assert marker["prompt"] == "restart the gateway, then check the board"
    assert marker["attempts"] == 1  # crash-loop breaker count survives the restart


def test_marker_restore_survives_kill_sweep_failure(
    live_session, marker_home, monkeypatch
):
    """The kill/rejoin path failing still means the process is on its way out — the restore runs
    regardless (try/finally), and the failure itself propagates to the suppressing caller."""
    record_turn_start(
        marker_home, "shutdown-restore-key", "mid-flight work", attempts=0
    )
    _patch_shutdown_turn_paths(monkeypatch, marker_home, kill_raises=True)

    with pytest.raises(RuntimeError):
        server._stop_turns_before_exit(budget_s=0.05)

    assert read_turn_marker(marker_home, "shutdown-restore-key") is not None


def test_restored_marker_refreshes_started_at_and_keeps_flags(
    live_session, marker_home, monkeypatch
):
    """``started_at`` refreshes at restore time so the freshness window spans the restart itself;
    ``auto_continue=False`` (mailbox-owned imported turn) and the attempt count round-trip."""
    record_turn_start(
        marker_home,
        "shutdown-restore-key",
        "imported turn",
        attempts=2,
        auto_continue=False,
    )
    original = read_turn_marker(marker_home, "shutdown-restore-key")
    _patch_shutdown_turn_paths(monkeypatch, marker_home)

    server._stop_turns_before_exit(budget_s=0.05)

    marker = read_turn_marker(marker_home, "shutdown-restore-key")
    assert marker is not None
    assert marker["attempts"] == 2
    assert marker["auto_continue"] is False
    assert marker["started_at"] >= original["started_at"]


def test_bot_room_turns_are_excluded(marker_home, monkeypatch):
    """Hosted bot-room turns are recovered by their durable task/lease state machine; a restored
    marker there would never be acted on, only linger until it ages out."""
    session = {
        "session_key": "bot-room-key",
        "running": True,
        "source": "bot_room",
        "history_lock": threading.Lock(),
    }
    with server._sessions_lock:
        server._sessions["sess-bot-room"] = session
    try:
        record_turn_start(marker_home, "bot-room-key", "hosted room work")
        interrupted = _patch_shutdown_turn_paths(monkeypatch, marker_home)

        server._stop_turns_before_exit(budget_s=0.05)

        assert interrupted == ["sess-bot-room"]
        assert read_turn_marker(marker_home, "bot-room-key") is None
    finally:
        with server._sessions_lock:
            server._sessions.pop("sess-bot-room", None)


def test_markerless_running_turn_writes_nothing(live_session, marker_home, monkeypatch):
    """A running turn without a marker (empty prompt, pre-marker build) must not gain one."""
    _patch_shutdown_turn_paths(monkeypatch, marker_home)

    server._stop_turns_before_exit(budget_s=0.05)

    assert not (marker_home / "desktop" / "interrupted_turns.json").exists()
