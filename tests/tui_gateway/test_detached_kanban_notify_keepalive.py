"""A detached Desktop/TUI session waiting on a subscribed Kanban card must survive the reapers.

``kanban_create`` auto-subscribes the calling Desktop/TUI session (``platform="tui"``, ``chat_id=session_key``) and the
agent ends its turn on "I'll report back when it finishes". The completion reaches the model only through that
session's notification poller. The WS-orphan reaper popped the session 20s after the window closed, so the card's
``completed`` event sat unclaimed until the user happened to reopen the chat hours later.
"""

from __future__ import annotations

import importlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn

SESSION_KEY = "desktop-kanban-session"


@pytest.fixture()
def server():
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending_ws_reaps.clear()


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
    sid = "sid-detached-kanban"
    session = {
        "session_key": SESSION_KEY,
        "transport": server._detached_ws_transport,
        "running": False,
        "history_lock": threading.Lock(),
        "last_active": 0.0,
        "created_at": 0.0,
    }
    server._sessions[sid] = session
    return sid, session, popped


def _subscribed_task(chat_id: str = SESSION_KEY) -> str:
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="report back", assignee="worker")
        kbn.add_notify_sub(conn, task_id=tid, platform="tui", chat_id=chat_id)
        return tid
    finally:
        conn.close()


def _complete(tid: str) -> None:
    conn = kbc.connect()
    try:
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()


def test_orphan_reap_rearms_while_subscribed_card_is_in_flight(server, detached):
    sid, session, popped = detached
    _subscribed_task()

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert server._sessions.get(sid) is session and not popped
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]


def test_cap_and_ttl_reapers_spare_session_awaiting_card(server, detached):
    sid, session, _ = detached
    assert server._session_is_reapable(sid, session)

    _subscribed_task()

    assert not server._session_is_reapable(sid, session)


def test_session_released_once_card_settles(server, detached):
    sid, session, popped = detached
    tid = _subscribed_task()
    _complete(tid)

    server._schedule_ws_orphan_reap(sid)
    _Timer.created[-1].callback()

    assert server._sessions.get(sid) is None and popped == [session]


def test_another_sessions_card_does_not_pin(server, detached):
    sid, session, _ = detached
    _subscribed_task(chat_id="some-other-session")

    assert server._session_is_reapable(sid, session)
