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


class FakeConnectorClient:
    """The only fake: the tool-gateway HTTP client boundary."""

    def __init__(self):
        self._connected = set()
        self._lock = threading.Lock()

    def list_connectors(self, **_):
        with self._lock:
            connected = set(self._connected)
        return [
            {"connector": name, "enabled": True, "connected": name in connected}
            for name in ("gmail", "notion")
        ]

    def connections(self, connectors, *, reinitiate=False, **_):
        return {
            "results": [
                {
                    "connector": name,
                    "status": "initiated",
                    "connect_url": f"https://connect.example/{name}",
                    "connection_id": f"ca_{name}",
                    "reinitiated": reinitiate,
                }
                for name in connectors
            ]
        }

    def account_status(self, connection_id, *, timeout=None):
        """The one route the watcher reads: the account the mint named."""
        name = connection_id.split("ca_", 1)[-1]
        with self._lock:
            active = name in self._connected
        return {
            "connectionId": connection_id,
            "connector": name,
            "status": "active" if active else "pending",
            "label": f"{name}_a",
            "active": active,
            "createdAt": "2026-09-14T10:00:00.000Z",
            "updatedAt": "2026-09-14T10:00:00.000Z",
        }

    def set_connected(self, name):
        with self._lock:
            self._connected.add(name)


@pytest.fixture
def owned_session(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
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
    response = server.dispatch(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": method,
            "params": {"owner": {"type": "session", "session_id": SID}, **params},
        },
        client.transport,
    )
    assert response is not None
    return response


def test_desktop_connect_settles_through_callback_response(owned_session, monkeypatch):
    """The actual tool thread waits for the gateway card outcome, not a fake callback."""
    owner, _ = owned_session
    client = FakeConnectorClient()
    monkeypatch.setattr("tools.connectors.managed.WATCH_TICK_SECONDS", 0.05)
    result = {}
    finished = threading.Event()

    def run_tool():
        # Bind the way the server binds a turn: the desktop surface arrives as the session's
        # `source`, not as a messaging platform.
        tokens = server._set_session_context(SID)
        try:
            # The agent's durable session_id is not the gateway session key (compaction rotates it
            # mid-turn); the operation must register under the key every RPC looks it up by.
            result["raw"] = manage_connections(
                {"action": "connect", "connectors": ["gmail", "notion"]},
                client_factory=lambda: client,
                session_id="20260914_rotated_agent_id",
                connection_callback=server._agent_cbs(SID)["connection_callback"],
            )
        except BaseException as exc:  # assertion below reports any real-path failure
            result["error"] = exc
        finally:
            clear_session_vars(tokens)
            finished.set()

    tool_thread = threading.Thread(target=run_tool, daemon=True)
    tool_thread.start()

    (request,) = owner.events("connection.request")
    assert request["op_id"]
    assert len(request["targets"]) == 2
    assert request["deadline_at"] > time.time()

    initiated = _rpc(owner, "connectors.operation.status", op_id=request["op_id"])["result"]
    by_name = {target["name"]: target for target in initiated["targets"]}
    assert {name: target["state"] for name, target in by_name.items()} == {
        "gmail": "initiated",
        "notion": "initiated",
    }
    assert all(target["connect_url"] == f"https://connect.example/{name}" for name, target in by_name.items())

    client.set_connected("gmail")
    assert owner.events(
        "connection.update",
        lambda payload: payload.get("target") == "gmail" and payload.get("to") == "connected",
    )

    response = _rpc(
        owner,
        "connection.respond",
        op_id=request["op_id"],
        result={"targets": [{"name": "notion", "status": "skipped"}]},
    )
    assert response["result"] == {"status": "ok", "settled": True}

    assert finished.wait(5), "connection.respond must release the callback bridge"
    tool_thread.join(timeout=5)
    assert not tool_thread.is_alive()
    assert "error" not in result

    settled = json.loads(result["raw"])
    assert settled["status"] == "settled"
    assert settled["settled_by"] == "all_resolved"
    assert {target["name"]: target["state"] for target in settled["targets"]} == {
        "gmail": "connected",
        "notion": "skipped",
    }
    assert "connect_url" not in json.dumps(settled)
    assert live.current(SID) is None

    # A settled operation leaves the registry: the frozen result travelled in the tool result and
    # in the last connection.update; the status RPC has nothing left to serve.
    closed = _rpc(owner, "connectors.operation.status", op_id=request["op_id"])
    assert closed["error"]["code"] == 4004
    final_update = owner.events("connection.update", lambda payload: payload.get("settled") is True)[-1]
    assert {target["name"]: target["state"] for target in final_update["targets"]} == {
        "gmail": "connected",
        "notion": "skipped",
    }


def test_cli_connect_returns_urls_without_emitting_a_card(owned_session):
    owner, _ = owned_session
    client = FakeConnectorClient()
    tokens = set_session_vars(platform="cli", session_key=SID, session_id=SID)
    try:
        result = json.loads(
            manage_connections(
                {"action": "connect", "connectors": ["gmail", "notion"]},
                client_factory=lambda: client,
                session_id=SID,
                connection_callback=server._agent_cbs(SID)["connection_callback"],
            )
        )
    finally:
        clear_session_vars(tokens)

    assert result["status"] == "initiated"
    assert all(target["connect_url"] for target in result["targets"])
    assert owner.events("connection.request") == []
    assert live.current(SID) is None


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
