"""Real-path desktop connection-operation lifecycle through the TUI gateway."""

import json
import os
import threading
import time
from contextlib import ExitStack, suppress

import pytest

from gateway.session_context import clear_session_vars, reset_session_vars, set_session_vars
from tools.connectors import live
from tools.connectors.contract import Actor, TargetState
from tools.connectors.operation import ConnectionOperation, Target
from tools.connectors.tool import manage_connections
from tui_gateway import server
from tui_gateway.transport import StdioTransport

SID = "connector-operation-e2e"


class PipeClient:
    """A real JSON-RPC pipe transport, following the gateway RPC test fixture idiom."""

    def __init__(self, stack):
        read_fd, write_fd = os.pipe()
        self.reader = stack.enter_context(os.fdopen(read_fd, "r", encoding="utf-8"))
        self.writer = os.fdopen(write_fd, "w", encoding="utf-8")
        stack.callback(self._cleanup)
        self.transport = StdioTransport(lambda: self.writer, threading.Lock())
        self.frames = []
        self._frames_lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.reader:
            with self._frames_lock:
                self.frames.append(json.loads(line))

    def _cleanup(self):
        if self._closed:
            return
        self._closed = True
        with suppress(BrokenPipeError):
            self.writer.close()
        self._reader.join(timeout=5)
        assert not self._reader.is_alive()

    def events(self, event_type, predicate=lambda _payload: True):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with self._frames_lock:
                events = [
                    frame["params"]["payload"]
                    for frame in self.frames
                    if frame.get("method") == "event"
                    and frame.get("params", {}).get("type") == event_type
                    and predicate(frame["params"].get("payload") or {})
                ]
            if events:
                return events
            time.sleep(0.01)
        with self._frames_lock:
            return [
                frame["params"]["payload"]
                for frame in self.frames
                if frame.get("method") == "event"
                and frame.get("params", {}).get("type") == event_type
                and predicate(frame["params"].get("payload") or {})
            ]



@pytest.fixture
def owned_session(monkeypatch, tmp_path):
    home = tmp_path / "rabbit-home"
    home.mkdir()
    monkeypatch.setenv("RABBIT_HOME", str(home))
    reset_session_vars()
    live.reset_for_tests()
    with ExitStack() as stack:
        owner, stranger = PipeClient(stack), PipeClient(stack)
        session = {
            "transport": owner.transport,
            "agent": None,
            "session_key": SID,
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "source": "desktop",
        }
        monkeypatch.setitem(server._sessions, SID, session)
        yield owner, stranger
    live.reset_for_tests()
    reset_session_vars()


def _rpc(client, method, **params):
    with client._frames_lock:
        before = len(client.frames)
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": method,
            "params": {"owner": {"type": "session", "session_id": SID}, **params},
        },
        client.transport,
    )
    if response is not None:
        return response
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with client._frames_lock:
            replies = [frame for frame in client.frames[before:] if frame.get("id") == 7]
        if replies:
            return replies[-1]
        time.sleep(0.01)
    raise AssertionError("no reply")


def test_connection_respond_ignores_outcome_claims_and_rejects_strangers(owned_session):
    owner, stranger = owned_session
    operation = ConnectionOperation([Target("gmail", "connector", "connect")], session_key=SID)
    live.open(operation)
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher)

    refused = _rpc(
        owner,
        "connection.respond",
        op_id=operation.op_id,
        result={"targets": [{"name": "gmail", "status": "connected"}]},
    )
    assert refused["error"]["code"] == 4002, refused
    assert operation.target("gmail").state == TargetState.initiated

    foreign = _rpc(
        stranger,
        "connection.respond",
        op_id=operation.op_id,
        result={"targets": [{"name": "gmail", "status": "skipped"}]},
    )
    assert foreign["error"]["code"] == 4001
