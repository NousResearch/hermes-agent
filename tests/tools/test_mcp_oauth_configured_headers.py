"""A server's configured ``headers`` reach its OAuth requests, on the MCP server's own origin only.

Regression for #155708. An auth proxy in front of an MCP host (Cloudflare Access service tokens, an
API-gateway key) redirects every request that lacks its headers. The MCP SDK builds its discovery,
registration and token requests as bare requests that never pick up the HTTP client's default
headers, so ``hermes mcp login`` against such a host died on the proxy's 302 before a browser ever
opened. The fixture plays that host: a request without the service-token header gets a 302 to the
proxy's login page, everything else is a same-origin MCP server + authorization server.
"""

import asyncio
import contextlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required for OAuth support")

ACCESS_HEADERS = {"CF-Access-Client-Id": "fixture-id.access", "CF-Access-Client-Secret": "fixture-secret"}


@pytest.fixture
def access_fronted_server():
    """MCP host behind an auth proxy; records (method, path, carried the service token, bearer)."""
    seen: list[tuple[str, str, bool, bool]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, status, payload=None, headers=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            if payload is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            path = self.path.split("?")[0]
            has_token = self.headers.get("CF-Access-Client-Id") == ACCESS_HEADERS["CF-Access-Client-Id"]
            bearer = (self.headers.get("Authorization") or "").startswith("Bearer ")
            seen.append((self.command, path, has_token, bearer))
            base = f"http://127.0.0.1:{self.server.server_port}"
            if not has_token:  # the proxy: cross-origin login page (an unreachable loopback port)
                return self._reply(302, headers={"Location": "http://127.0.0.1:1/cdn-cgi/access/login"})
            if path == "/mcp":
                if bearer:  # authorized: the OAuth leg under test is over
                    return self._reply(503, {"error": "fixture stops here"})
                return self._reply(401, {"error": "unauthorized"}, {
                    "WWW-Authenticate": f'Bearer resource_metadata="{base}/.well-known/oauth-protected-resource/mcp"'})
            if path.startswith("/.well-known/oauth-protected-resource"):
                return self._reply(200, {"resource": f"{base}/mcp", "authorization_servers": [base]})
            if path.startswith("/.well-known/oauth-authorization-server"):
                return self._reply(200, {
                    "issuer": base, "authorization_endpoint": f"{base}/authorize",
                    "token_endpoint": f"{base}/token", "registration_endpoint": f"{base}/register",
                    "device_authorization_endpoint": f"{base}/device",
                    "grant_types_supported": ["authorization_code", "refresh_token",
                                              "urn:ietf:params:oauth:grant-type:device_code"],
                    "response_types_supported": ["code"], "code_challenge_methods_supported": ["S256"]})
            if path == "/register":
                return self._reply(201, {**json.loads(raw), "client_id": "fixture-client"})
            if path == "/device":
                return self._reply(200, {"device_code": "dc", "user_code": "TEST-CODE",
                                         "verification_uri": f"{base}/verify", "interval": 0.01, "expires_in": 30})
            if path == "/token":
                return self._reply(200, {"access_token": "at", "refresh_token": "rt",
                                         "token_type": "Bearer", "expires_in": 3600})
            return self._reply(404, {"error": "not found"})

        do_GET = do_POST = do_DELETE = _handle

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_browser_login_carries_configured_headers_on_every_oauth_request(
    access_fronted_server, tmp_path, monkeypatch
):
    """The real connect path (``MCPServerTask._run_http`` → manager → provider → SDK auth flow) sends
    the configured headers on discovery, registration and the token exchange, so the proxy lets the
    whole OAuth leg through and a token is issued."""
    import tools.mcp_oauth as mo
    from mcp.client.auth import AuthorizationCodeResult
    from tools.mcp_oauth_manager import reset_manager_for_tests
    from tools.mcp_tool import MCPServerTask

    base, seen = access_fronted_server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    # Stand in for the browser: remember the authorize URL, "approve" with its state.
    authorize_urls: list[str] = []

    def redirect_handler(port, redirect_uri=None, redirect_host=None):
        async def _open(url):
            authorize_urls.append(url)
        return _open

    def callback_waiter(port, cimd_url=None, timeout=300.0):
        async def _wait():
            state = parse_qs(urlsplit(authorize_urls[-1]).query)["state"][0]
            return AuthorizationCodeResult(code="fixture-code", state=state)
        return _wait

    monkeypatch.setattr(mo, "_make_redirect_handler", redirect_handler)
    monkeypatch.setattr(mo, "_make_callback_waiter", callback_waiter)

    task = MCPServerTask("behind-access")
    task._auth_type = "oauth"
    task._sampling = None
    config = {"url": f"{base}/mcp", "auth": "oauth", "headers": dict(ACCESS_HEADERS), "connect_timeout": 10}

    async def drive():
        with mo.force_interactive_oauth(), contextlib.suppress(Exception):
            await asyncio.wait_for(task._run_http(config), timeout=20)

    asyncio.run(drive())

    paths = [path for _, path, _, _ in seen]
    assert "/register" in paths and "/token" in paths, seen
    assert all(has_token for _, _, has_token, _ in seen), seen  # no request met the proxy's 302
    assert any(path == "/mcp" and bearer for _, path, _, bearer in seen), seen  # a token was issued and used


@pytest.mark.asyncio
async def test_configured_headers_never_reach_a_separate_authorization_server(tmp_path, monkeypatch):
    """Headers go only to the MCP server's own origin: an authorization server elsewhere never receives
    the proxy credential, and a configured ``Authorization`` is never copied onto an OAuth request."""
    from mcp.shared.auth import OAuthClientMetadata
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    async def _noop(*_args):
        return None

    provider = _HERMES_PROVIDER_CLS(
        server_name="srv", server_url="https://mcp.example.com/mcp", storage=HermesTokenStorage("srv"),
        client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")]),
        redirect_handler=_noop, callback_handler=_noop)
    provider.set_forward_headers({**ACCESS_HEADERS, "Authorization": "Bearer static"})
    request = httpx.Request("POST", "https://mcp.example.com/mcp")
    flow = provider.async_auth_flow(request)

    assert await flow.__anext__() is request
    prm = await flow.asend(httpx.Response(401, request=request, headers={
        "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource"'}))
    assert prm.url.host == "mcp.example.com"
    assert prm.headers.get("CF-Access-Client-Secret") == "fixture-secret"
    assert "authorization" not in prm.headers

    asm = await flow.asend(httpx.Response(200, request=prm, json={
        "resource": "https://mcp.example.com/mcp", "authorization_servers": ["https://auth.example.net"]}))
    assert asm.url.host == "auth.example.net"
    assert "CF-Access-Client-Id" not in asm.headers and "CF-Access-Client-Secret" not in asm.headers
    assert asm.headers.get("user-agent")  # the default stamp still applies cross-origin

    await flow.aclose()


def test_device_login_carries_configured_headers(access_fronted_server, tmp_path, monkeypatch):
    """``hermes mcp login --flow device`` sends its own requests; they carry the headers too."""
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_device import login_device
    from tools.mcp_oauth_manager import reset_manager_for_tests

    base, seen = access_fronted_server
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()

    asyncio.run(login_device("behind-access", f"{base}/mcp", {"flow": "device"}, headers=dict(ACCESS_HEADERS)))

    assert [path for _, path, _, _ in seen][-2:] == ["/device", "/token"], seen
    assert all(has_token for _, _, has_token, _ in seen), seen
    assert HermesTokenStorage("behind-access").has_cached_tokens()
