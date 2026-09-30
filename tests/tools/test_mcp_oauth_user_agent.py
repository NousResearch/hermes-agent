"""Tests for the per-server ``oauth.user_agent`` on MCP OAuth token requests.

Some authorization servers and WAFs reject httpx's default User-Agent on the
token endpoint (#75576). The header is opt-in, per-server, and applied ONLY to
the two token-endpoint requests (authorization-code exchange and refresh) —
never to MCP traffic or discovery. With no ``oauth.user_agent`` configured the
shared ``Hermes-Agent/<version>`` default is stamped instead: those requests
are hand-built and sent with ``client.send()``, which never merges the client's
default headers, so an unset UA used to mean NO ``User-Agent`` on the wire at
all and a WAF-fronted authorization server answered 403 (#115329).

The tests drive the REAL provider classes' request builders end to end: the
``httpx.Request`` the SDK would send is what gets inspected, not a mocked
constructor call. The device-flow test additionally observes the headers on a
real socket, because that path (``tools.mcp_oauth_device``) sends its own token
request instead of yielding it into the SDK's auth flow.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK required for OAuth support",
)

from tools.mcp_oauth import (  # noqa: E402 — after the SDK availability gate
    build_oauth_auth,
    token_request_user_agent,
)
from tools.mcp_oauth_provider import DEFAULT_AUTH_REQUEST_USER_AGENT  # noqa: E402


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


@pytest.fixture(autouse=True)
def clean_port_state():
    import tools.mcp_oauth as mod

    mod._assigned_cimd_ports.clear()
    yield
    mod._assigned_cimd_ports.clear()
    for port in list(mod._reserved_sockets):
        sock = mod._reserved_sockets.pop(port, None)
        if sock is not None:
            sock.close()


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_configured_user_agent_is_returned():
    assert token_request_user_agent({"user_agent": "My-MCP-Client/1.0"}) == "My-MCP-Client/1.0"


@pytest.mark.parametrize("cfg", [
    pytest.param({}, id="absent"),
    pytest.param({"user_agent": None}, id="null"),
    pytest.param({"user_agent": ""}, id="empty"),
    pytest.param({"user_agent": "   "}, id="whitespace-only"),
    pytest.param({"user_agent": 7}, id="non-string"),
])
def test_unset_user_agent_values_are_treated_as_absent(cfg):
    assert token_request_user_agent(cfg) is None


def test_user_agent_is_stripped():
    assert token_request_user_agent({"user_agent": "  UA/2 "}) == "UA/2"


# ---------------------------------------------------------------------------
# The requests the SDK actually sends
# ---------------------------------------------------------------------------


def _ready_for_token_requests(provider):
    """Give the provider the minimum context both builders require."""
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    provider.context.oauth_metadata = SimpleNamespace(
        token_endpoint="https://idp.example.com/oauth/token"
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "client-1",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })
    provider.context.current_tokens = OAuthToken.model_validate({
        "access_token": "at",
        "token_type": "Bearer",
        "refresh_token": "rt",
    })


def _build_provider_via(builder, monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    return builder("srv", "https://mcp.example.com/mcp", cfg)


def _manager_builder(server_name, server_url, cfg):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()
    return MCPOAuthManager().get_or_build_provider(server_name, server_url, cfg)


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_token_requests_carry_the_configured_user_agent(
    builder, tmp_path, monkeypatch
):
    """Both token-endpoint requests, on both provider construction paths."""
    provider = _build_provider_via(
        builder, monkeypatch, tmp_path, {"user_agent": "My-MCP-Client/1.0"}
    )
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    assert exchange.headers["User-Agent"] == "My-MCP-Client/1.0"
    assert refresh.headers["User-Agent"] == "My-MCP-Client/1.0"


@pytest.mark.parametrize("builder", [
    pytest.param(build_oauth_auth, id="build_oauth_auth"),
    pytest.param(_manager_builder, id="oauth_manager"),
])
def test_unconfigured_user_agent_falls_back_to_the_hermes_default(
    builder, tmp_path, monkeypatch
):
    """No config → the shared ``Hermes-Agent/<version>`` default, never a header-less request.

    A bare ``httpx.Request`` carries no User-Agent and ``client.send()`` never merges the
    client's default headers, so an unset ``oauth.user_agent`` used to put these POSTs on
    the wire with nothing but host/content-type/content-length (#115329).
    """
    provider = _build_provider_via(builder, monkeypatch, tmp_path, {})
    _ready_for_token_requests(provider)

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )
    refresh = asyncio.run(provider._refresh_token())

    assert exchange.headers.get("User-Agent") == DEFAULT_AUTH_REQUEST_USER_AGENT
    assert refresh.headers.get("User-Agent") == DEFAULT_AUTH_REQUEST_USER_AGENT


def test_user_agent_does_not_disturb_token_auth_preparation(tmp_path, monkeypatch):
    """The stamp runs after prepare_token_auth — a confidential client's
    Authorization header must survive alongside the custom User-Agent."""
    provider = _build_provider_via(
        build_oauth_auth, monkeypatch, tmp_path,
        {"user_agent": "UA/1", "client_id": "pre", "client_secret": "shh",
         "token_endpoint_auth_method": "client_secret_basic"},
    )
    _ready_for_token_requests(provider)
    from mcp.shared.auth import OAuthClientInformationFull

    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "pre",
        "client_secret": "shh",
        "token_endpoint_auth_method": "client_secret_basic",
        "redirect_uris": ["http://127.0.0.1:33333/callback"],
    })

    exchange = asyncio.run(
        provider._exchange_token_authorization_code("code", "verifier")
    )

    assert exchange.headers["User-Agent"] == "UA/1"
    assert exchange.headers.get("Authorization", "").startswith("Basic ")


# ---------------------------------------------------------------------------
# Device flow: the token poll is sent by Hermes itself, so watch the socket
# ---------------------------------------------------------------------------


@pytest.fixture
def device_authorization_server():
    """A local RFC 8628 authorization server that records the headers it receives."""
    seen: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}))
            base = f"http://127.0.0.1:{self.server.server_port}"
            if self.path == "/device":
                return self._reply(200, {"device_code": "fixture-device-code", "user_code": "TEST-CODE",
                                         "verification_uri": f"{base}/verify", "interval": 0.01,
                                         "expires_in": 30})
            if self.path == "/token":
                if sum(path == "/token" for path, _ in seen) == 1:
                    return self._reply(400, {"error": "authorization_pending"})
                return self._reply(200, {"access_token": "at", "refresh_token": "rt",
                                         "token_type": "Bearer", "expires_in": 3600})
            self._reply(404, {})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_device_flow_token_poll_carries_a_user_agent_on_the_wire(
    device_authorization_server, tmp_path, monkeypatch
):
    """Every request of an unconfigured device login reaches the server with a User-Agent.

    ``tools.mcp_oauth_device._authorize`` builds its token poll by hand and sends it with
    ``client.send()`` — the one token request that never passes through the SDK auth flow's
    default-UA stamp, so it left the socket header-less (#115329).
    """
    from mcp.shared.auth import OAuthClientInformationFull

    from tools.mcp_oauth import _build_client_metadata
    from tools.mcp_oauth_device import _authorize
    from tools.mcp_oauth_manager import HermesMCPOAuthProvider
    from tools.mcp_oauth_provider import prepare_oauth_config
    from tools.mcp_tool import sdk_httpx

    base, seen = device_authorization_server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Exactly how `login_device` builds it for an unconfigured (`oauth.user_agent` absent) server.
    cfg, storage = prepare_oauth_config("srv", f"{base}/mcp", {})
    cfg["_resolved_port"] = cfg.get("redirect_port", 8420)
    provider = HermesMCPOAuthProvider(
        server_url=f"{base}/mcp", server_name="srv", storage=storage,
        client_metadata=_build_client_metadata(cfg),
        token_user_agent=cfg.get("user_agent"),
    )
    provider.context.oauth_metadata = SimpleNamespace(
        issuer=base, token_endpoint=f"{base}/token", device_authorization_endpoint=f"{base}/device")
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "fixture-client",
        "token_endpoint_auth_method": "none",
        "redirect_uris": [f"http://127.0.0.1:33333/callback"],
    })

    httpx = sdk_httpx()

    async def run():
        async with httpx.AsyncClient(timeout=5.0) as client:
            return await _authorize(client, provider, {"timeout": 5})

    assert asyncio.run(run()).access_token == "at"

    assert [path for path, _ in seen] == ["/device", "/token", "/token"]
    agents = [headers.get("user-agent") for _, headers in seen]
    assert all(agents), seen  # nothing leaves Hermes header-less
    polls = [headers.get("user-agent") for path, headers in seen if path == "/token"]
    assert polls == [DEFAULT_AUTH_REQUEST_USER_AGENT] * len(polls)
