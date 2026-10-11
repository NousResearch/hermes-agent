"""Several accounts of one connector, each addressed by its name (alias).

Contracts:
- the account a call ran under reaches the model on the single-call and the batch path
- connect with an alias asks the gateway for that account; reconnect with an alias repairs that
  account by its connection id even when another account of the same connector is healthy
- rename resolves the name to the account; an unknown name lists the valid names
- an open operation for one account is never reused for another account of the same connector
"""

import json
from unittest.mock import patch

import pytest

from tools.connectors import live
from tools.connectors import operation as op
from tools.connectors.gateway.client import ConnectorClient
from tools.connectors.tool import manage_connections


@pytest.fixture(autouse=True)
def _clean_live():
    live.reset_for_tests()
    yield
    live.reset_for_tests()


class _Response:
    def __init__(self, body):
        self.status_code = 200
        self._body = body

    def json(self):
        return self._body


class _ExecuteTransport:
    """Answers every execute with one result per call, each naming the account it ran under."""

    def request(self, method, url, *, headers=None, json=None, timeout=None):
        results = [{"index": i, "connector": c["connector"], "tool": c["tool"], "data": {"ok": True},
                    "account": c["arguments"].get("connector_alias")} for i, c in enumerate(json["tools"])]
        return _Response({"results": results, "successCount": len(results), "errorCount": 0,
                          "totalCount": len(results)})


def test_the_account_a_call_ran_under_reaches_the_model_on_both_dispatch_paths(monkeypatch):
    import model_tools
    from tools.connectors.gateway import bridge, config
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(config, "connectors_available", lambda: True)
    monkeypatch.setattr(bridge, "connectors_available", lambda: True)
    invalidate_check_fn_cache()
    monkeypatch.setattr(bridge, "_default_client_factory", lambda: ConnectorClient(
        transport=_ExecuteTransport(), endpoint_resolver=lambda: "https://gateway.test",
        header_provider=lambda _url: {"Authorization": "Bearer t"}))
    kwargs = dict(enabled_toolsets=["connections"], session_id="s", skip_pre_tool_call_hook=True,
                  skip_tool_request_middleware=True, skip_tool_execution_middleware=True)

    single = json.loads(model_tools.handle_function_call(
        "connectors__gmail__FETCH_EMAILS", {"connector_alias": "work"}, **kwargs))
    assert single["account"] == "work"

    batch = json.loads(model_tools.handle_function_call("tool_call", {"calls": [
        {"name": "connectors__gmail__FETCH_EMAILS", "arguments": {"connector_alias": "home"}},
        {"name": "connectors__gmail__FETCH_EMAILS", "arguments": {}},
    ]}, **kwargs))
    assert [entry.get("account") for entry in batch["results"]] == ["home", None]


class _Gateway:
    """The connections and account routes, recording every mint."""

    def __init__(self, connected=()):
        self.connected = set(connected)
        self.mints = []

    def list_connectors(self, *, timeout=None):
        return [{"connector": "gmail", "enabled": True, "connected": "gmail" in self.connected}]

    def connections(self, connectors, *, reinitiate=False, alias=None, connection_id=None, return_to=None, op=None):
        mint = {"connectors": tuple(connectors), "reinitiate": reinitiate, "alias": alias}
        if connection_id:
            mint["connection_id"] = connection_id
        self.mints.append(mint)
        return {"results": [{"connector": c, "status": "initiated", "connection_id": f"ca_{c}_{len(self.mints)}",
                             "connect_url": f"https://connect.example/{c}"} for c in connectors]}

    def account_status(self, connection_id, *, timeout=None):
        return {"connectionId": connection_id, "connector": "gmail", "status": "active", "label": "a@example.com",
                "active": True, "createdAt": "2026-10-08T00:00:00Z", "updatedAt": "2026-10-08T00:00:00Z"}


def _account(alias, label, status="active"):
    return {"connectionId": f"ca_{alias or label}", "connector": "gmail", "status": status, "label": label,
            "alias": alias, "active": status == "active", "createdAt": "t", "updatedAt": "t"}


def _run(args, gateway, accounts=()):
    with patch("tools.connectors.managed.WATCH_TICK_SECONDS", 0.0), \
         patch("tools.connectors.managed.portal_accounts", return_value=list(accounts)), \
         patch("tools.connectors.gateway.client.session_platform", return_value="desktop"):
        return json.loads(manage_connections(args, client_factory=lambda: gateway,
                                             connection_callback=lambda _payload: None, session_id="s1"))


def test_connect_with_an_alias_asks_for_that_account_and_watches_the_one_it_mints():
    gateway = _Gateway()
    out = _run({"action": "connect", "connectors": [{"name": "gmail", "alias": "work"}]}, gateway)
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": False, "alias": "work"}]
    (target,) = out["targets"]
    assert (target["alias"], target["state"]) == ("work", "connected")


def test_reconnect_with_an_alias_repairs_it_even_when_another_account_is_healthy():
    gateway = _Gateway(connected={"gmail"})
    accounts = [_account("home", "me@example.com"), _account("work", "me@corp.example", status="expired")]
    _run({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "work"}]}, gateway, accounts)
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": True, "alias": None, "connection_id": "ca_work"}]


def test_reconnect_of_an_unnamed_account_by_its_label_repairs_that_account_by_id():
    gateway = _Gateway(connected={"gmail"})
    accounts = [_account("home", "me@example.com"), _account(None, "gmail_old-label", status="expired")]
    _run({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "gmail_old-label"}]}, gateway, accounts)
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": True, "alias": None,
                              "connection_id": "ca_gmail_old-label"}]


def test_reconnect_of_a_healthy_unnamed_account_by_label_mints_nothing_unless_forced():
    accounts = [_account("home", "me@example.com"), _account(None, "gmail_old-label")]
    gateway = _Gateway()
    _run({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "gmail_old-label"}]}, gateway, accounts)
    assert gateway.mints == []
    _run({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "gmail_old-label"}], "force": True},
         gateway, accounts)
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": True, "alias": None,
                              "connection_id": "ca_gmail_old-label"}]


def test_an_open_repair_of_one_account_is_not_reused_for_another():
    repair = op.ConnectionOperation([op.Target("gmail", "connector", "reconnect", repair_id="ca_a")], session_key="s1")
    live.open(repair)
    assert live.find_target("gmail", repair_id="ca_a") is repair
    assert live.find_target("gmail", repair_id="ca_b") is None


def test_a_pending_link_reaches_the_model_with_its_account_id_and_no_link():
    from tools.connectors.gateway.merge import PlannedCall, render_remote_entry

    planned = PlannedCall(position=0, name="connectors__gmail__FETCH_EMAILS", connector="gmail",
                          tool="FETCH_EMAILS", arguments={})
    entry = render_remote_entry(planned, {"data": None, "error": {
        "code": "CONNECTION_REQUIRED", "message": "Connect this app to continue.", "connector": "gmail",
        "connection_id": "ca_pending", "hint": "The user already has a sign-in link for this account."}})
    assert entry["error"]["connection_id"] == "ca_pending"
    assert "connect_url" not in entry["error"]
    assert "already has a sign-in link" in entry["error"]["hint"]


def test_reconnect_of_a_name_no_account_has_yet_sends_the_alias():
    gateway = _Gateway()
    _run({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "work"}]}, gateway,
         [_account("home", "me@example.com")])
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": True, "alias": "work"}]


def test_try_again_on_a_named_account_repairs_the_account_it_minted_not_the_name(monkeypatch):
    from tools.connectors import run
    from tools.connectors.contract import Actor, TargetState

    gateway = _Gateway()
    monkeypatch.setattr("tools.connectors.managed.managed_client", lambda: gateway)
    operation = op.ConnectionOperation([op.Target("gmail", "connector", "reconnect", alias="work")], session_key="s1")
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher, connection_id="ca_work")
    operation.transition("gmail", TargetState.failed, Actor.backend_watcher, detail="expired")
    with patch("tools.connectors.gateway.client.session_platform", return_value="desktop"):
        assert run.reissue(operation, ["gmail"]) is None
    assert gateway.mints == [{"connectors": ("gmail",), "reinitiate": True, "alias": None, "connection_id": "ca_work"}]


def test_reconnect_with_an_alias_mints_nothing_when_the_account_list_cannot_be_read():
    from tools.connectors.gateway.errors import ToolGatewayError

    gateway = _Gateway()
    with patch("tools.connectors.managed.WATCH_TICK_SECONDS", 0.0), \
         patch("tools.connectors.managed.portal_accounts", side_effect=ToolGatewayError("down", status=503)):
        out = json.loads(manage_connections({"action": "reconnect", "connectors": [{"name": "gmail", "alias": "work"}]},
                                            client_factory=lambda: gateway, connection_callback=lambda _p: None,
                                            session_id="s1"))
    assert gateway.mints == []
    assert "error" in out


def test_rename_sends_the_new_name_for_the_resolved_account():
    from tools.connectors.portal.client import PortalConnectorClient

    sent = []

    class Transport:
        def request(self, method, url, *, headers=None, json=None, timeout=None):
            sent.append((method, url, json))
            return _Response({"connectionId": "ca_work", "connector": "gmail", "status": "active", "label": "me@corp.example",
                              "alias": json["alias"], "active": True, "createdAt": "t", "updatedAt": "t"})

    client = PortalConnectorClient(transport=Transport(), endpoint_resolver=lambda: "https://portal.test",
                                   header_provider=lambda _url: {"Authorization": "Bearer t"})
    with patch("tools.connectors.portal.client.PortalConnectorClient", return_value=client):
        out = _run({"action": "rename", "connectors": [{"name": "gmail", "alias": "work", "to": "office"}]},
                   _Gateway(), [_account("home", "me@example.com"), _account("work", "me@corp.example")])
    assert sent == [("PATCH", "https://portal.test/api/v1/connectors/accounts/ca_work", {"alias": "office"})]
    assert out["renamed"] == {"connector": "gmail", "alias": "office", "label": "me@corp.example"}


def test_rename_of_an_unknown_name_lists_the_names_that_exist():
    accounts = [_account("home", "me@example.com"), _account(None, "me@corp.example")]
    with patch("tools.connectors.portal.client.PortalConnectorClient.rename_account") as rename:
        out = _run({"action": "rename", "connectors": [{"name": "gmail", "alias": "wrk", "to": "office"}]},
                   _Gateway(), accounts)
    rename.assert_not_called()
    assert "home" in out["error"] and "me@corp.example" in out["error"]


def test_an_open_operation_for_one_account_is_not_reused_for_another():
    plain = op.ConnectionOperation([op.Target("gmail", "connector", "connect")], session_key="s1")
    live.open(plain)
    assert live.find_target("gmail") is plain
    assert live.find_target("gmail", alias="work") is None
