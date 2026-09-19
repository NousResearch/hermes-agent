"""Account-owned connector operations through the JSON-RPC rail."""

from __future__ import annotations

import threading
import time

from tests.fakes.connectors_managed import FakeManagedClient
from tests.tui_gateway.conftest import ReplyTransport, reply
from tools.connectors import live


def test_a_client_still_sending_the_old_top_level_session_id_is_refused():
    answer = reply(ReplyTransport(), "connectors.list", {"session_id": "s"})

    assert answer["error"]["code"] == 4000
    assert "session_id" in answer["error"]["message"]


def test_account_connect_starts_a_watcher_broadcasts_updates_and_closes(monkeypatch):
    from tools.connectors import managed
    from tui_gateway import server

    client = FakeManagedClient()
    transport = ReplyTransport()
    live.reset_for_tests()
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr(managed, "WATCH_TICK_SECONDS", 0.01)
    monkeypatch.setattr(managed, "managed_client", lambda: client)
    monkeypatch.setattr(server, "_live_transports", {transport})
    monkeypatch.setattr(server, "_live_transports_lock", threading.Lock())
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_DEV_SERVER", raising=False)
    try:
        answer = reply(transport, "connectors.connect", {
            "owner": {"type": "account"}, "connectors": ["gmail"], "reconnect": False,
        })
        assert answer["result"]["op_id"]
        # No chat session binds the surface, yet the link must still come back to the desktop app.
        assert client.mint_args == {"return_to": "hermes-desktop", "op": answer["result"]["op_id"]}
        assert answer["result"]["targets"] == [{
            "name": "gmail", "kind": "connector", "action": "connect", "state": "initiated",
            "connect_url": "https://connect.example/gmail", "connection_id": "ca_gmail",
        }]
        operation = live.get_by_op_id(answer["result"]["op_id"])
        assert operation is not None and client.mints == 1
        client.connected.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and live.get_by_op_id(operation.op_id) is not None:
            transport.event.wait(0.05)
            transport.event.clear()
        updates = [frame["params"]["payload"] for frame in transport.frames
                   if frame.get("method") == "event" and frame["params"].get("type") == "connection.update"]
        assert any(update["owner"] == {"type": "account"} and update["op_id"] == operation.op_id
                   and update.get("to") == "connected" for update in updates)
        # The broadcast reaches every connected client, so the authorization link never rides it.
        assert all("connect_url" not in target for update in updates for target in update["targets"])
        assert live.get_by_op_id(operation.op_id) is None
    finally:
        live.reset_for_tests()


def test_account_wake_is_profile_local(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.connectors.operation import ConnectionOperation, Target
    from tui_gateway import server

    home = tmp_path / "home"
    other = tmp_path / "other"
    home.mkdir()
    other.mkdir()
    live.reset_for_tests()
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr(server, "_hermes_home", str(home))
    operation = ConnectionOperation([Target("gmail", "connector", "connect")], session_key="account:other")
    token = set_hermes_home_override(str(other))
    try:
        live.open(operation)
    finally:
        reset_hermes_home_override(token)
    try:
        answer = reply(ReplyTransport(), "connectors.operation.wake", {
            "owner": {"type": "account"}, "op_id": operation.op_id,
        })
        assert answer["error"]["data"]["reason"] == "UNKNOWN_OPERATION"
    finally:
        live.reset_for_tests()
