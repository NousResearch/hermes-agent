"""A detached Bot Chat must not kill its own message_agent delivery runner.

message_agent hands each message to a background runner (``bot_mode_dm._spawn_delivery``) that
waits for the other bot's reply and reports it as a ``notify_on_complete`` completion. When a
phone locks, the app's WebSocket drops; ``ws_orphan_reap_grace_s`` later the reaper tore the
session down, and agent close SIGTERMed the runner along with the reply it was waiting for.
"""

import json
import threading
import time

import pytest

from agent.client_lifecycle import ClientLifecycleMixin
from tools import bot_mode_dm
from tools.process_registry import process_registry
from tui_gateway import server

TASK_ID = "bot-chat-sender-task"
SID = "detached-bot-chat"


class _Agent(ClientLifecycleMixin):
    """The part of ``AIAgent.close`` that kills processes the session owns."""

    def __init__(self):
        self.session_id = TASK_ID
        self._process_owner_task_ids = {TASK_ID}

    def close(self):
        self._close_task_resources(self.session_id)


def _spawn_runner():
    """message_agent's real spawn path, with a stand-in for the runner command."""
    result = json.loads(bot_mode_dm._spawn_delivery("sleep 30", "@gimli", task_id=TASK_ID, agent=None))
    assert result["status"] == "queued", result
    runner = process_registry.get(result["process_id"])
    assert runner is not None and not runner.exited
    return runner


def _finish(runner):
    process_registry.kill_process(runner.id)
    deadline = time.monotonic() + 10
    while not runner.exited and time.monotonic() < deadline:
        time.sleep(0.05)
    assert runner.exited


@pytest.fixture
def runner():
    proc = _spawn_runner()
    yield proc
    if not proc.exited:
        _finish(proc)


def test_detached_session_keeps_its_delivery_runner_past_the_grace(monkeypatch, runner):
    """The grace passes while the runner still waits for the reply: the session stays parked
    and the runner alive. Once the runner reports, the next reap check closes the session."""
    callbacks = []

    class _Timer:
        def __init__(self, _delay, callback):
            callbacks.append(callback)
            self.daemon = False

        def start(self):
            return None

    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0.01)
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_teardown_session",
                        lambda session, *, end_reason="tui_close": session["agent"].close())
    session = {"agent": _Agent(), "session_key": "", "running": False, "history": [],
               "history_lock": threading.Lock(), "transport": server._detached_ws_transport}
    server._sessions[SID] = session
    try:
        server._schedule_ws_orphan_reap(SID)
        callbacks.pop(0)()

        assert not runner.exited, "the reaper killed the message_agent runner"
        assert SID in server._sessions
        assert not server._session_is_lru_evictable(SID, session)

        _finish(runner)
        callbacks.pop(0)()

        assert SID not in server._sessions
    finally:
        server._sessions.pop(SID, None)
        server._pending_ws_reaps.pop(SID, None)


def test_closing_the_sender_agent_leaves_the_delivery_runner_running(runner):
    """Any other close of the sender (TTL or LRU reaper, /new, an explicit close) must not take
    the delivery with it either: the runner is the only thing waiting for the reply."""
    _Agent().close()

    assert not runner.exited
