"""A completion replayed at process start requires a real notification consumer (#138348)."""

import asyncio
import io
import queue
import sqlite3
import threading
import time
from unittest.mock import Mock

import pytest

from tools import async_delegation as ad
from tools.process_registry import process_registry
from tools.process_registry_notifications import format_process_notification
from tui_gateway import server
from tui_gateway.transport import StdioTransport, TeeTransport
from tui_gateway.ws import WSTransport


@pytest.fixture
def notification_env(monkeypatch, tmp_path):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    ad._reset_for_tests()
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(process_registry, "completion_queue", queue.Queue())
    monkeypatch.setattr(server, "_emit", Mock())
    # The execution boundary reports admission; no model or worker is started.
    submit = Mock(return_value=True)
    monkeypatch.setattr(server, "_run_prompt_submit", submit)
    loop = asyncio.new_event_loop()
    try:
        yield home, loop, submit
    finally:
        loop.close()
        ad._reset_for_tests()


def _session(notification_env, monkeypatch, kind):
    home, loop, _submit = notification_env
    stdio = TeeTransport(StdioTransport(io.StringIO, threading.Lock()),
                         StdioTransport(io.StringIO, threading.Lock()))  # dashboard-sidecar-wrapped
    monkeypatch.setattr(server, "_stdio_transport", stdio)
    monkeypatch.setattr(server, "_stdio_is_rpc_channel", kind == "stdio")
    transport = {"stdio": stdio, "websocket": WSTransport(None, loop),
                 "detached": server._detached_ws_transport}[kind]
    session = {"session_key": "chat", "transport": transport, "profile_home": str(home),
               "history_lock": threading.RLock(), "running": False, "_finalized": False}
    server._sessions["ui1"] = session
    return session


def _orphan_owner(home):
    """The owner died while this process runs: pid 0 never passes the liveness probe."""
    conn = sqlite3.connect(home / "state.db")
    with conn:
        conn.execute("UPDATE async_delegations SET owner_pid=0, updated_at=?", (time.time() - 3600,))
    conn.close()


def _completion(home, *, via):
    event = {
        "type": "async_delegation", "delegation_id": "completed-child",
        "session_key": "chat", "origin_ui_session_id": "ui1",
        "dispatched_at": time.time(), "completed_at": time.time(),
        "status": "completed", "summary": "child result",
        "results": [{"status": "completed", "summary": "child result"}],
    }
    ad._persist_dispatch(event)
    ad._persist_completion(event, {"summary": "child result"})
    offered = queue.Queue()
    if via == "startup":
        assert ad.restore_undelivered_completions(offered) == 1
    else:
        _orphan_owner(home)
        assert ad.sweep_orphaned_completions(offered) == 1
    return offered.get_nowait()


@pytest.mark.parametrize("kind,via,owns", [
    ("detached", "startup", False),  # the reported shape: no client, lease survived the restart
    ("websocket", "startup", True),  # live WS peer
    ("stdio", "startup", True),      # this process's real TUI stdio RPC channel
    ("detached", "sweep", True),     # in-process orphan sweep keeps its routing
])
def test_startup_replay_requires_a_live_consumer(notification_env, monkeypatch, kind, via, owns):
    home, _loop, submit = notification_env
    session = _session(notification_env, monkeypatch, kind)
    event = _completion(home, via=via)
    assert event["restored"] is True and bool(event.get("startup_replay")) is (via == "startup")

    server._notif_handle_ready(
        "ui1", session, [event], set(), process_registry, format_process_notification, None)

    assert submit.call_count == int(owns)
    row = ad.get_durable_delegation(event["delegation_id"])
    assert (row["delivery_state"], row["delivery_attempts"]) == (("delivered", 1) if owns else ("pending", 0))
    if not owns:
        # Refusal hands the offer back; the sweep's re-offer stays gated until a client attaches.
        _orphan_owner(home)
        offered = queue.Queue()
        assert ad.sweep_orphaned_completions(offered) == 1
        assert offered.get_nowait()["startup_replay"] is True
