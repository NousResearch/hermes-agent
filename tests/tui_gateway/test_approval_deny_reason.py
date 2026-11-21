"""A typed deny reason on a Desktop folder/site card reaches the agent on the legacy tui_gateway.

Desktop sends ``{choice: "deny", reason}`` either as the response to the ``approval`` server request or
as ``approval.respond`` params; both must hand ``reason`` to ``resolve_gateway_approval``, which puts it
in the BLOCKED message the model reads.
"""
import pytest

from tests.tui_gateway import test_protocol

rpc_server = pytest.fixture()(test_protocol.server.__wrapped__)


def test_approval_deny_reason_reaches_the_agent(rpc_server, monkeypatch):
    from tools import approval
    from tui_gateway import server_requests

    server = rpc_server
    server._sessions["ui-1"] = {"session_key": "agent-1", "history": []}
    calls = []
    monkeypatch.setattr(approval, "resolve_gateway_approval",
                        lambda key, choice, **kwargs: calls.append(kwargs.get("reason")) or 1)
    server.handle_request({"id": "r4", "method": "approval.respond", "params": {
        "session_id": "ui-1", "request_id": "req-1", "choice": "deny", "reason": "not that folder"}})

    sent = {}
    monkeypatch.setattr(server_requests, "send_async", lambda kind, sid, payload, cb: sent.setdefault("cb", cb) and None)
    monkeypatch.setattr(approval, "register_gateway_settle", lambda *a: True)
    server._emit_approval_request("ui-1", {"request_id": "req-2", "command": "x", "description": ""})
    sent["cb"]({"choice": "deny", "reason": "not that site"})
    sent["cb"]({"choice": "once"})
    assert calls == ["not that folder", "not that site", None]
