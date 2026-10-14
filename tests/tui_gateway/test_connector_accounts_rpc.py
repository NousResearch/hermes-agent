"""Named connector accounts over the real RPC dispatch.

- ``connectors.connect`` with an alias opens an operation for that one account and mints it
- ``connectors.accounts.rename`` answers ALIAS_TAKEN when another account holds the name
"""

import threading

import pytest

from tools.connectors import live
from tools.connectors.gateway.errors import IdempotencyConflict
from tui_gateway import server


@pytest.fixture(autouse=True)
def _gate(monkeypatch):
    live.reset_for_tests()
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    yield
    live.reset_for_tests()


class _Transport:
    """Collects the frames a long handler writes back instead of returning."""

    def __init__(self):
        self.frames = []
        self.arrived = threading.Event()

    def write(self, obj):
        self.frames.append(obj)
        if obj.get("id") == 1:
            self.arrived.set()
        return True

    def close(self):
        pass


def _rpc(method, transport=None, **params):
    transport = transport or _Transport()
    # A shared transport still holds the previous call's reply; read only this call's frames.
    transport.frames.clear()
    transport.arrived.clear()
    reply = server.dispatch({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, transport)
    if reply is not None:
        return reply
    assert transport.arrived.wait(5), "no reply"
    return next(frame for frame in transport.frames if frame.get("id") == 1)


def test_account_connect_with_an_alias_mints_that_account(monkeypatch):
    mints = []
    minted = threading.Event()

    class Client:
        def connections(self, names, *, reinitiate=False, alias=None, **_):
            mints.append((tuple(names), reinitiate, alias))
            minted.set()
            return {"results": [{"connector": n, "status": "initiated", "connection_id": "ca_1",
                                 "connect_url": "https://connect.example/1"} for n in names]}

        def account_status(self, connection_id, *, timeout=None):
            return None

    monkeypatch.setattr("tools.connectors.managed.managed_client", Client)
    reply = _rpc("connectors.connect", owner={"type": "account"}, connectors=["gmail"], alias="work")
    assert "result" in reply, reply
    assert minted.wait(2)
    assert mints == [(("gmail",), False, "work")]
    assert [(t["name"], t.get("alias")) for t in reply["result"]["targets"]] == [("gmail", "work")]


def test_a_session_retry_for_another_account_does_not_re_mint_the_open_one(monkeypatch):
    from tools.connectors.contract import Actor, TargetState
    from tools.connectors.operation import ConnectionOperation, Target

    transport = _Transport()
    session = dict(transport=transport, agent=None, session_key="alias-sid", history=[],
                   history_lock=threading.Lock(), history_version=0, running=False, attached_images=[],
                   source="desktop")
    monkeypatch.setitem(server._sessions, "alias-sid", session)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})
    operation = ConnectionOperation([Target("gmail", "connector", "connect", alias="home")], session_key="alias-sid")
    live.open(operation)
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/1")
    operation.transition("gmail", TargetState.failed, Actor.backend_watcher, detail="expired")
    mints = []

    class Client:
        def connections(self, names, **kwargs):
            mints.append((tuple(names), kwargs.get("alias")))
            return {"results": []}

    monkeypatch.setattr("tools.connectors.managed.managed_client", Client)
    reply = _rpc("connectors.connect", transport, owner={"type": "session", "session_id": "alias-sid"},
                 connectors=["gmail"], alias="work")
    assert reply["error"]["code"] == 4004
    live.close(operation)
    unnamed = ConnectionOperation([Target("gmail", "connector", "reconnect", repair_id="ca_home")], session_key="alias-sid")
    live.open(unnamed)
    unnamed.transition("gmail", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/2")
    unnamed.transition("gmail", TargetState.failed, Actor.backend_watcher, detail="expired")
    reply = _rpc("connectors.connect", transport, owner={"type": "session", "session_id": "alias-sid"},
                 connectors=["gmail"], reconnect=True, connection_id="ca_work")
    assert reply["error"]["code"] == 4004
    assert mints == []


def test_a_session_retry_with_the_open_accounts_id_reissues_it(monkeypatch):
    from tools.connectors.contract import Actor, TargetState
    from tools.connectors.operation import ConnectionOperation, Target

    transport = _Transport()
    session = dict(transport=transport, agent=None, session_key="retry-sid", history=[],
                   history_lock=threading.Lock(), history_version=0, running=False, attached_images=[],
                   source="desktop")
    monkeypatch.setitem(server._sessions, "retry-sid", session)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})
    operation = ConnectionOperation([Target("gmail", "connector", "reconnect", alias="work")], session_key="retry-sid")
    live.open(operation)
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/1",
                         connection_id="ca_work")
    operation.transition("gmail", TargetState.failed, Actor.backend_watcher, detail="expired")
    mints = []

    class Client:
        def connections(self, names, **kwargs):
            mints.append((tuple(names), kwargs.get("connection_id")))
            return {"results": [{"connector": n, "status": "initiated", "connect_url": "https://l/2",
                                 "connection_id": "ca_work2"} for n in names]}

    monkeypatch.setattr("tools.connectors.managed.managed_client", Client)
    reply = _rpc("connectors.connect", transport, owner={"type": "session", "session_id": "retry-sid"},
                 connectors=["gmail"], reconnect=True, connection_id="ca_work")
    assert "result" in reply, reply
    assert mints == [(("gmail",), "ca_work")]


def test_rename_to_a_name_another_account_holds_is_alias_taken(monkeypatch):
    def taken(self, connection_id, alias):
        raise IdempotencyConflict("alias taken", code="alias_taken", status=409)

    monkeypatch.setattr("tools.connectors.portal.client.PortalConnectorClient.rename_account", taken)
    reply = _rpc("connectors.accounts.rename", connection_id="ca_1", alias="home")
    assert reply["error"]["data"]["reason"] == "ALIAS_TAKEN"
