"""RFC 9207 issuer survives native callback capture and reaches the SDK unchanged."""

import asyncio
import http.client
import io
import threading
import time
from urllib.parse import urlencode

import pytest

from hermes_constants import get_hermes_home
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.client.auth.utils import validate_authorization_response_iss
from mcp.shared.auth import AuthorizationCodeResult, OAuthMetadata
from tools.mcp_dashboard_oauth import DashboardOAuthFlow, dashboard_oauth_flow
from tools.mcp_oauth import (
    _callback_outcome,
    _make_callback_handler,
    _make_callback_waiter,
    _paste_callback_reader,
    _start_callback_server,
)
from tui_gateway.mcp_oauth_sessions import _start_loopback_listener


def _deliver_callback(route, flow, params, monkeypatch):
    if route == "dashboard":
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from hermes_cli.web_routers import mcp

        app = FastAPI()
        app.include_router(mcp.router)
        monkeypatch.setitem(mcp._mcp_oauth_flows, flow.flow_id, flow)
        response = TestClient(app).get(
            f"/api/mcp/oauth/callback/{flow.server_name}", params=params,
        )
        assert response.status_code == 200
        return
    if route == "rpc":
        from tui_gateway import mcp_oauth_sessions, server

        monkeypatch.setitem(mcp_oauth_sessions._sessions, flow.flow_id, {
            "flow": flow, "server_name": flow.server_name,
            "hermes_home": flow.hermes_home, "created_at": time.time(),
        })
        response = server._methods["mcp.servers.oauth.callback"](1, {
            "name": flow.server_name, "session_id": flow.flow_id, **params,
        })
        assert response["result"]["ok"] is True
        return
    captured = None
    if route == "loopback":
        listener = _start_loopback_listener(flow)
    else:
        handler, captured = _make_callback_handler()
        if route == "cli-paste":
            monkeypatch.setattr("tools.mcp_oauth.sys.stdin", io.StringIO(
                "http://127.0.0.1/callback?" + urlencode(params) + "\n"))
            _paste_callback_reader(captured)
            return _callback_outcome(captured, None)
        assert route == "cli-loopback"
        listener = _start_callback_server(0, handler)
        threading.Thread(target=listener.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection("127.0.0.1", listener.server_port, timeout=5)
    try:
        conn.request("GET", "/callback?" + urlencode(params))
        response = conn.getresponse()
        response.read()
        assert response.status == 200
    finally:
        conn.close()
        listener.shutdown()
        listener.server_close()
    if captured is not None:
        return _callback_outcome(captured, None)


@pytest.mark.parametrize("route", ["loopback", "dashboard", "rpc", "cli-loopback", "cli-paste"])
@pytest.mark.parametrize(
    "issuer,supported,error",
    [
        ("https://issuer.example", True, None),
        ("https://other.example", True, "iss mismatch"),
        ("https://issuer.example/", True, "iss mismatch"),
        (" https://issuer.example ", True, "iss mismatch"),
        ("https://other.example", False, "iss mismatch"),
        ("", True, "iss mismatch"),
        ("", False, "iss mismatch"),
        (None, True, "missing iss"),
        (None, False, None),
    ],
)
def test_callback_preserves_issuer_for_sdk_validation(route, issuer, supported, error, monkeypatch):
    flow = DashboardOAuthFlow(
        flow_id="issuer-flow", server_name="issuer-test", profile=None,
        hermes_home=str(get_hermes_home()), redirect_uri="",
    )
    asyncio.run(flow.publish_authorization_url("https://issuer.example/authorize?state=test-state"))
    metadata = OAuthMetadata.model_validate({
        "issuer": "https://issuer.example",
        "authorization_endpoint": "https://issuer.example/authorize",
        "token_endpoint": "https://issuer.example/token",
        "response_types_supported": ["code"],
        "authorization_response_iss_parameter_supported": supported,
    })
    params = {"code": "test-code", "state": "test-state"}
    if issuer is not None:
        params["iss"] = issuer
    result = _deliver_callback(route, flow, params, monkeypatch)
    if result is None:
        with dashboard_oauth_flow(flow):
            result = asyncio.run(_make_callback_waiter(0)())
    assert isinstance(result, AuthorizationCodeResult)
    assert result.iss == issuer
    assert (result.code, result.state) == ("test-code", "test-state")
    if error:
        with pytest.raises(OAuthFlowError, match=error):
            validate_authorization_response_iss(result.iss, metadata)
    else:
        validate_authorization_response_iss(result.iss, metadata)


@pytest.mark.parametrize("issuer", [False, True, 0, 1, 0.0, 1.5, [], ["issuer"], {}, {"iss": "issuer"}])
def test_rpc_rejects_non_string_issuer_without_consuming_callback(issuer, monkeypatch):
    from tui_gateway import mcp_oauth_sessions, server

    flow = DashboardOAuthFlow(
        flow_id="issuer-type-flow", server_name="issuer-test", profile=None,
        hermes_home=str(get_hermes_home()), redirect_uri="",
    )
    asyncio.run(flow.publish_authorization_url("https://issuer.example/authorize?state=test-state"))
    monkeypatch.setitem(mcp_oauth_sessions._sessions, flow.flow_id, {
        "flow": flow, "server_name": flow.server_name,
        "hermes_home": flow.hermes_home, "created_at": time.time(),
    })
    callback = server._methods["mcp.servers.oauth.callback"]
    params = {
        "name": flow.server_name, "session_id": flow.flow_id,
        "code": "test-code", "state": "test-state", "iss": issuer,
    }
    response = callback(1, params)
    assert response.get("error", {}).get("code") == 4063
    assert "iss" in response["error"]["message"]

    # Rejected input must not consume the callback; explicit null is accepted.
    params["iss"] = None
    assert callback(2, params)["result"]["ok"] is True
    with dashboard_oauth_flow(flow):
        result = asyncio.run(_make_callback_waiter(0)())
    assert isinstance(result, AuthorizationCodeResult)
    assert (result.code, result.state, result.iss) == ("test-code", "test-state", None)
