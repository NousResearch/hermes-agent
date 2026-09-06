"""Telemetry admission follows the existing WS-upgrade credential gate."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hermes_cli import web_server_chat
from tui_gateway import server
from tui_gateway.transport import bind_transport, reset_transport
from tui_gateway.ws import WSTransport


@pytest.mark.parametrize("host, token, allowed", [
    ("127.0.0.1", "fixture-token", True), ("::1", "fixture-token", True),
    ("192.0.2.1", "fixture-token", False), ("127.0.0.1", "wrong", False),
])
def test_upgrade_auth_marks_only_local_authenticated_telemetry(monkeypatch, host, token, allowed):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "fixture-token")
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    ws = SimpleNamespace(query_params={"token": token}, client=SimpleNamespace(host=host))
    reason, _ = web_server_chat._ws_auth_reason(ws)
    assert (reason is None) == (token == "fixture-token")
    assert getattr(ws, "_hermes_local_telemetry", False) is allowed


@pytest.mark.parametrize("authenticated, local, code", [(False, True, 4403), (True, False, 4403), (True, True, None)])
def test_ws_rpc_admission_is_backend_metadata(monkeypatch, tmp_path, authenticated, local, code):
    monkeypatch.setattr(server, "_hermes_home", str(tmp_path))
    monkeypatch.setattr(server, "_sessions", {})
    loop = asyncio.new_event_loop()
    try:
        transport = WSTransport(object(), loop, authenticated=authenticated, local_telemetry=local)
        binding = bind_transport(transport)
        try:
            response = server.handle_request({"id": 1, "method": "usage.active_work", "params": {}})
        finally:
            reset_transport(binding)
        if code:
            assert response["error"]["code"] == code
        else:
            assert response["result"]["scope"]["device_local"] is True
            assert response["result"]["scope"]["renderer_device_local"] == "unknown"
        assert "fixture-token" not in json.dumps(response)
    finally:
        loop.close()
