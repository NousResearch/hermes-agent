"""A detached Desktop/TUI session waiting on a background process's completion notice must survive the reapers.

The agent starts ``terminal(background=True, notify_on_complete=True)`` and ends its turn on "I'll report back".
The completion reaches the model only through that session's notification poller, and closing the session kills
the process (``agent_close``). The LRU cap evicted such sessions seconds after the turn ended, so the process died
with exit -15, no completion was ever delivered, and the promised follow-up silently never happened.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def server(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending_ws_reaps.clear()


def _drain(registry):
    while not registry.completion_queue.empty():
        registry.completion_queue.get_nowait()


@pytest.fixture()
def registry():
    from tools.process_registry import process_registry

    saved = dict(process_registry._running), dict(process_registry._finished)
    process_registry._running.clear()
    process_registry._finished.clear()
    _drain(process_registry)
    yield process_registry
    _drain(process_registry)
    process_registry._running.clear()
    process_registry._finished.clear()
    process_registry._running.update(saved[0])
    process_registry._finished.update(saved[1])


class _Timer:
    created: list = []

    def __init__(self, delay, callback):
        self.delay, self.callback = delay, callback
        _Timer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        pass


@pytest.fixture()
def detached(server, monkeypatch):
    _Timer.created = []
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_has_active_delegations", lambda *a: False)
    popped = []
    monkeypatch.setattr(server, "_teardown_popped_session", lambda s, **kw: popped.append(s))
    sid = "sid-detached-proc"
    session = {
        "session_key": "desktop-proc-session",
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "last_active": 0.0,
        "created_at": 0.0,
    }
    server._sessions[sid] = session
    return sid, session, popped


def _spawn(registry, owner: str, **kw):
    """A background process as the TUI turn spawns it: owner_task_id is the turn's task id (the session key)."""
    from tools.process_registry import ProcessSession

    proc = ProcessSession(id=f"proc_{len(registry._running)}", command="hermes -p reviewer chat",
                          session_key=owner, owner_task_id=owner, started_at=time.time(), **kw)
    registry._running[proc.id] = proc
    return proc


@pytest.mark.parametrize("notify", [{"notify_on_complete": True}, {"watch_patterns": ["VERDICT"]}])
def test_cap_and_ttl_reapers_spare_session_with_pending_process_notice(server, registry, detached, notify):
    sid, session, _ = detached
    assert server._session_is_reapable(sid, session)  # idle and clientless: evictable today

    _spawn(registry, session["session_key"], **notify)

    assert not server._session_is_reapable(sid, session)
    assert not server._session_is_evictable(sid, session, time.time() + 10 * 86400)


def test_orphan_reap_rearms_while_process_will_notify(server, registry, detached):
    sid, session, popped = detached
    _spawn(registry, session["session_key"], notify_on_complete=True)

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert server._sessions.get(sid) is session and not popped
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]


def test_exited_process_keeps_session_until_its_notice_is_dequeued(server, registry, detached):
    """The exit is recorded before the completion is queued, and the notice becomes a turn only once the poller
    dequeues it: a reaper firing anywhere in that hand-off must not close the session and lose the follow-up."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.exited = True  # mark_exited ran; _move_to_finished (publish) has not yet

    assert not server._session_is_reapable(sid, session)

    registry._move_to_finished(proc)  # published: queued, not yet dequeued
    assert not registry.completion_queue.empty()
    assert not server._session_is_reapable(sid, session)

    registry.completion_queue.get_nowait()  # the poller took it
    assert server._session_is_reapable(sid, session)


def test_queued_notice_survives_registry_pruning(server, registry, detached):
    """Retained history is pruned by start time and a process cap, on every spawn: a long job's queued completion
    must keep protecting its owner after its finished entry is gone."""
    from tools.process_registry import FINISHED_TTL_SECONDS

    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.started_at = time.time() - FINISHED_TTL_SECONDS - 1
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    with registry._lock:
        registry._prune_if_needed()

    assert proc.id not in registry._finished
    assert not server._session_is_reapable(sid, session)


def test_session_released_once_notice_is_delivered(server, registry, detached):
    sid, session, popped = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    registry.completion_queue.get_nowait()

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert sid not in server._sessions and popped == [session]


def test_notice_held_after_stop_does_not_pin_session(server, registry, detached):
    """After the user's Stop the poller requeues notices until their next prompt; that wait must not be forever."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True)
    proc.mark_exited(0)
    registry._move_to_finished(proc)
    session["_turn_cancel_requested"] = True

    assert server._session_is_reapable(sid, session)


def test_recovered_process_whose_pid_died_releases_session(server, registry, detached, monkeypatch):
    """A checkpoint-recovered process has no reader thread to observe its exit; a dead PID must not pin its owner."""
    sid, session, _ = detached
    _spawn(registry, session["session_key"], notify_on_complete=True, detached=True, pid=12345)
    monkeypatch.setattr(registry, "_detached_host_fate", lambda *_: "gone")

    assert server._session_is_reapable(sid, session)


@pytest.mark.parametrize("unit", ["", "hermes-proc.scope"])
def test_uncollected_recovered_exit_never_pins_session(server, registry, detached, unit):
    """A recovered entry closed without a collected exit (exited_at stays 0, a scope keeps it retained) queues no
    completion, so it must not hold its owner however long ago it ended."""
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], notify_on_complete=True, detached=True, pid=12345,
                  systemd_unit=unit)
    registry._prune_uncollected_detached(proc)

    assert proc.exited and not proc.exited_at
    assert server._session_is_reapable(sid, session)


def test_silent_background_process_does_not_pin_session(server, registry, detached):
    """A process nobody asked to hear from (a dev server) must not make its session immortal."""
    sid, session, _ = detached
    _spawn(registry, session["session_key"])

    assert server._session_is_reapable(sid, session)


def test_another_sessions_process_does_not_pin_session(server, registry, detached):
    sid, session, _ = detached
    _spawn(registry, "some-other-session", notify_on_complete=True)

    assert server._session_is_reapable(sid, session)


def test_process_started_before_compression_still_pins_session(server, registry, detached):
    """Compression rotates session_key to the continuation id; the process keeps its original owner id. The agent
    still kills it on close (its _process_owner_task_ids keep the old id), so it must still pin the session."""
    from types import SimpleNamespace

    sid, session, _ = detached
    original = session["session_key"]
    _spawn(registry, original, notify_on_complete=True)
    session["session_key"] = "desktop-proc-session-continuation"
    session["agent"] = SimpleNamespace(session_id=session["session_key"],
                                       _process_owner_task_ids={original, session["session_key"]})

    assert not server._session_is_reapable(sid, session)


def test_disabled_watch_does_not_pin_session(server, registry, detached):
    sid, session, _ = detached
    proc = _spawn(registry, session["session_key"], watch_patterns=["VERDICT"])
    proc._watch_disabled = True

    assert server._session_is_reapable(sid, session)
