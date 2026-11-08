"""The Webapp's Authenticate RPC uses the serving listener's public callback."""

import asyncio
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from hermes_constants import get_hermes_home
from tests.tools.test_connectors_mcp_chat_oauth import _listener
from tests.tui_gateway.test_mcp_oauth_client_callback import _fake_worker_publishes_url
from tools.connectors import mcp_oauth
from tui_gateway import mcp_oauth_sessions as sessions

SERVER = "reports !'()* / team"
PUBLIC_BASE = "https://agent.example/hermes"
CFG = {"url": "https://mcp.example/mcp", "auth": "oauth"}


@pytest.fixture
def public_oauth(monkeypatch):
    from hermes_cli import mcp_config

    sessions._sessions.clear()
    _fake_worker_publishes_url(monkeypatch)
    monkeypatch.setattr(mcp_config, "_get_mcp_servers", lambda: {SERVER: CFG})
    bound = []

    def loopback_receiver(flow):
        bound.append(flow)
        return SimpleNamespace(server_address=("127.0.0.1", 8412), shutdown=lambda: None, server_close=lambda: None)

    monkeypatch.setattr(mcp_oauth, "_start_loopback_receiver", loopback_receiver)
    yield bound
    for rec in sessions._sessions.values():
        rec["flow"].mark_error("test cleanup", cancelled=True)
        assert rec["flow"]._worker_done.wait(5)
    sessions._sessions.clear()


def _start_rpc(**params):
    from tui_gateway import server
    from tui_gateway.contracts import registry

    contract = registry.METHODS["mcp.servers.oauth.start"]
    validated, problem = registry.validate_params(contract, {"name": SERVER, **params})
    assert problem is None
    return server._methods["mcp.servers.oauth.start"](1, validated)


def test_start_flow_public_base_skips_loopback(public_oauth):
    result = sessions.start_flow(str(get_hermes_home()), SERVER, CFG, public_base=PUBLIC_BASE)
    rec = sessions._sessions[result["session_id"]]
    assert rec["httpd"] is None
    assert rec["flow"].redirect_uri == f"{PUBLIC_BASE}/api/mcp/oauth/callback/{quote(SERVER, safe='')}"
    assert public_oauth == []


def test_public_rpc_flow_completes_through_webapp_callback(public_oauth, monkeypatch):
    from hermes_cli.web_routers.mcp import mcp_oauth_callback

    _listener(monkeypatch)
    out = _start_rpc(public_callback=PUBLIC_BASE)
    assert "error" not in out
    result = out["result"]
    assert result["ok"] is True
    flow = sessions._sessions[result["session_id"]]["flow"]
    assert flow.redirect_uri == f"{PUBLIC_BASE}/api/mcp/oauth/callback/{quote(SERVER, safe='')}"
    assert public_oauth == []
    rejected = asyncio.run(mcp_oauth_callback(SERVER, code="fixture-code", state="wrong"))
    assert rejected.status_code == 404
    response = asyncio.run(mcp_oauth_callback(
        SERVER, code="fixture-code", state="teststate123", iss="https://idp.example"))
    assert response.status_code == 200
    assert flow._worker_done.wait(5)
    assert flow._callback == ("fixture-code", "teststate123", "https://idp.example")
    assert sessions.poll_flow(result["session_id"], SERVER)["status"] == "approved"
    replay = asyncio.run(mcp_oauth_callback(SERVER, code="fixture-code", state="teststate123"))
    assert replay.status_code == 404


@pytest.mark.parametrize("state,public_url,page_base", [
    ("off", PUBLIC_BASE, PUBLIC_BASE), ("starting", PUBLIC_BASE, PUBLIC_BASE), ("serving", None, PUBLIC_BASE),
    ("serving", PUBLIC_BASE, "http://192.168.1.20:9120"), ("serving", PUBLIC_BASE, "https://agent.example"),
])
def test_public_rpc_keeps_the_loopback_listener_for_other_pages(public_oauth, monkeypatch, state, public_url, page_base):
    """No public URL (LAN mode), or a page served under another name or path: the browser may be on
    the gateway machine, as on base."""
    _listener(monkeypatch, state=state, public_url=public_url)
    out = _start_rpc(public_callback=page_base)
    assert out["result"]["ok"] is True
    flow = sessions._sessions[out["result"]["session_id"]]["flow"]
    assert flow.redirect_uri == "http://127.0.0.1:8412/callback"
    assert public_oauth == [flow]


@pytest.mark.parametrize("public_url,page_base", [
    (PUBLIC_BASE, "HTTPS://Agent.Example:443/hermes/"),
    ("https://bücher.example/hermes", "https://xn--bcher-kva.example/hermes"),
    ("https://faß.example/hermes", "https://xn--fa-hia.example/hermes"),
    ("https://☃.example/hermes", "https://xn--n3h.example/hermes"),  # a symbol IDNA 2008 rejects
])
def test_public_rpc_matches_the_public_url_case_port_slash_and_idn(public_oauth, monkeypatch, public_url, page_base):
    _listener(monkeypatch, public_url=public_url)
    out = _start_rpc(public_callback=page_base)
    flow = sessions._sessions[out["result"]["session_id"]]["flow"]
    assert flow.redirect_uri == f"{public_url}/api/mcp/oauth/callback/{quote(SERVER, safe='')}"
    assert public_oauth == []


@pytest.mark.parametrize("public_url,page_base", [
    (PUBLIC_BASE, "https://agent.example:0/hermes"),  # port 0 is not the default port
    ("https://faß.example/hermes", "https://fass.example/hermes"),  # IDNA 2003 would map ß to ss
])
def test_public_rpc_keeps_loopback_for_a_near_miss(public_oauth, monkeypatch, public_url, page_base):
    _listener(monkeypatch, public_url=public_url)
    out = _start_rpc(public_callback=page_base)
    assert sessions._sessions[out["result"]["session_id"]]["flow"].redirect_uri == "http://127.0.0.1:8412/callback"


def test_public_rpc_refuses_client_redirect(public_oauth, monkeypatch):
    _listener(monkeypatch)
    out = _start_rpc(public_callback=PUBLIC_BASE, client_redirect_uri="http://127.0.0.1:8412/callback")
    assert out["error"]["code"] == 4001
    assert "client_redirect_uri" in out["error"]["message"]
    assert not sessions._sessions
    assert public_oauth == []
