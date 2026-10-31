"""MCP OAuth endpoint trust rules (https, no private targets behind a public server, bounded
responses) and sign-out on server removal (RFC 7009 revocation)."""

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("mcp.client.auth")

from tools import url_safety
from tools.mcp_oauth_endpoints import (
    OAUTH_MAX_RESPONSE_BYTES, OAuthEndpointRefused, check_oauth_endpoint, oauth_client_event_hooks)

_DNS = {"mcp.example.com": "93.184.216.34", "auth.example.com": "93.184.216.35", "intranet.example.com": "10.0.0.5"}


@pytest.fixture
def fake_dns(monkeypatch):
    real = url_safety._getaddrinfo

    def _resolve(host, port=None):
        if host in _DNS:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (_DNS[host], port or 0))]
        return real(host, port)
    monkeypatch.setattr(url_safety, "_getaddrinfo", _resolve)
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


@pytest.mark.parametrize("endpoint, server, allowed", [
    ("https://auth.example.com/token", "https://mcp.example.com/mcp", True),
    ("http://auth.example.com/token", "https://mcp.example.com/mcp", False),          # plain http to a public host
    ("https://intranet.example.com/token", "https://mcp.example.com/mcp", False),     # public server -> private
    ("https://127.0.0.1:8443/token", "https://mcp.example.com/mcp", False),           # public server -> loopback
    ("https://169.254.169.254/token", "http://127.0.0.1:9000/mcp", False),            # metadata floor, always
    ("http://127.0.0.1:9001/token", "http://127.0.0.1:9000/mcp", True),               # loopback dev server
    ("http://intranet.example.com/token", "http://192.168.1.20:8123/mcp", True),      # LAN server, LAN endpoint
    ("http://auth.example.com/token", "http://127.0.0.1:9000/mcp", False),            # local server, public http
    ("javascript:alert(1)", "http://127.0.0.1:9000/mcp", False),
])
def test_advertised_endpoint_rules(fake_dns, endpoint, server, allowed):
    if allowed:
        check_oauth_endpoint(endpoint, server)
    else:
        with pytest.raises(OAuthEndpointRefused):
            check_oauth_endpoint(endpoint, server)


def test_sdk_refresh_to_unsafe_token_endpoint_is_refused_before_sending(fake_dns, tmp_path, monkeypatch):
    """Every request the SDK's flow makes passes the rules: a public server whose metadata names a plain-http
    token endpoint never receives the refresh token."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = tmp_path / "mcp-tokens"
    tokens.mkdir()
    (tokens / "srv.json").write_text(json.dumps({
        "access_token": "old", "refresh_token": "refresh-secret", "token_type": "Bearer", "expires_in": 0,
        "expires_at": time.time() - 60, "hermes_issuer": "https://auth.example.com"}), encoding="utf-8")
    (tokens / "srv.client.json").write_text(json.dumps({"client_id": "cid", "redirect_uris": ["http://127.0.0.1:1/cb"]}), encoding="utf-8")
    (tokens / "srv.meta.json").write_text(json.dumps({
        "issuer": "https://auth.example.com", "authorization_endpoint": "https://auth.example.com/authorize",
        "token_endpoint": "http://auth.example.com/token", "response_types_supported": ["code"]}), encoding="utf-8")
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    from tools.mcp_tool import sdk_httpx
    reset_manager_for_tests()
    provider = get_manager().get_or_build_provider("srv", "https://mcp.example.com/mcp", None)
    httpx = sdk_httpx()
    seen = []

    def _handler(request):
        seen.append(str(request.url))
        return httpx.Response(401)
    async def _drive():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler), auth=provider) as client:
            await client.post("https://mcp.example.com/mcp", json={})
    with pytest.raises(OAuthEndpointRefused, match="only sent over https"):
        asyncio.run(_drive())
    assert not any("auth.example.com" in url for url in seen)
    reset_manager_for_tests()


def test_declared_fake_ip_server_gets_no_local_network_trust(fake_dns, monkeypatch):
    """A public server reached through a declared fake-IP proxy address is still public: it may not
    point Hermes at a private address, nor over plain http."""
    import ipaddress
    monkeypatch.setitem(_DNS, "proxied.example.com", "198.18.1.125")
    monkeypatch.setattr(url_safety, "_global_fake_ip_ranges", lambda: (ipaddress.ip_network("198.18.0.0/15"),))
    for endpoint in ("http://intranet.example.com/token", "https://intranet.example.com/token",
                     "http://auth.example.com/token"):
        with pytest.raises(OAuthEndpointRefused):
            check_oauth_endpoint(endpoint, "https://proxied.example.com/mcp")
    check_oauth_endpoint("https://auth.example.com/token", "https://proxied.example.com/mcp")


def _expired_session(tmp_path, monkeypatch, token_endpoint="https://auth.example.com/token"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = tmp_path / "mcp-tokens"
    tokens.mkdir()
    (tokens / "srv.json").write_text(json.dumps({
        "access_token": "old", "refresh_token": "refresh-secret", "token_type": "Bearer", "expires_in": 0,
        "expires_at": time.time() - 60, "hermes_issuer": "https://auth.example.com"}), encoding="utf-8")
    (tokens / "srv.client.json").write_text(json.dumps({"client_id": "cid", "redirect_uris": ["http://127.0.0.1:1/cb"]}), encoding="utf-8")
    (tokens / "srv.meta.json").write_text(json.dumps({
        "issuer": "https://auth.example.com", "authorization_endpoint": "https://auth.example.com/authorize",
        "token_endpoint": token_endpoint, "response_types_supported": ["code"]}), encoding="utf-8")
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    return get_manager().get_or_build_provider("srv", "https://mcp.example.com/mcp", None)


def _drive_through_mcp_transport(provider, handler):
    """Send one MCP request through the transport stack every Hermes MCP HTTP client uses."""
    from tools.mcp_tool import sdk_httpx
    from tools.mcp_tool_errors import _make_mcp_body_cap_transport
    httpx = sdk_httpx()

    async def _drive():
        transport = _make_mcp_body_cap_transport(httpx, httpx.MockTransport(handler))
        async with httpx.AsyncClient(transport=transport, auth=provider, follow_redirects=True) as client:
            await client.post("https://mcp.example.com/mcp", json={})
    return httpx, _drive


def test_token_endpoint_redirect_to_unsafe_endpoint_is_refused(fake_dns, tmp_path, monkeypatch):
    """httpx follows a token endpoint's 307 on its own, outside the auth flow: the hop still passes the
    endpoint rules, so the refresh token never reaches a plain-http private address."""
    provider = _expired_session(tmp_path, monkeypatch)
    seen = []

    def _handler(request):
        seen.append(str(request.url))
        if request.url.host == "auth.example.com":
            return httpx.Response(307, headers={"location": "http://intranet.example.com/token"})
        return httpx.Response(401)
    httpx, drive = _drive_through_mcp_transport(provider, _handler)
    with pytest.raises(OAuthEndpointRefused):
        asyncio.run(drive())
    assert "https://auth.example.com/token" in seen
    assert not any("intranet" in url for url in seen)


def test_token_error_body_is_capped_before_any_reader(fake_dns, tmp_path, monkeypatch):
    """A 400 from the token endpoint is read (dead-registration sniff) before the SDK sees it; the
    OAuth body bound applies to that read too."""
    provider = _expired_session(tmp_path, monkeypatch)
    pulled = []

    async def _huge():
        for _ in range(4):
            pulled.append(1)
            yield b"x" * (OAUTH_MAX_RESPONSE_BYTES // 2 + 1)

    def _handler(request):
        if request.url.host == "auth.example.com":
            return httpx.Response(400, content=_huge())
        return httpx.Response(401)
    httpx, drive = _drive_through_mcp_transport(provider, _handler)
    with pytest.raises(OAuthEndpointRefused, match="exceeds"):
        asyncio.run(drive())
    assert len(pulled) == 2


def test_oversized_oauth_response_is_refused(fake_dns):
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()

    async def _huge():
        for _ in range(3):
            yield b"x" * (OAUTH_MAX_RESPONSE_BYTES // 2 + 1)

    async def _fetch():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=_huge()))
        async with httpx.AsyncClient(transport=transport,
                                     event_hooks=oauth_client_event_hooks(httpx, "https://mcp.example.com/mcp")) as client:
            await client.get("https://auth.example.com/.well-known/oauth-authorization-server")
    with pytest.raises(OAuthEndpointRefused, match="exceeds"):
        asyncio.run(_fetch())


class _RevokeServer:
    """A loopback authorization server recording RFC 7009 calls; ``stall`` makes it never answer."""

    def __init__(self, stall=False):
        calls, gate = [], threading.Event()

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                calls.append((self.path, body))
                if stall:
                    gate.wait(10)
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
        self.calls, self._gate = calls, gate
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self._gate.set()
        self.httpd.shutdown()


def _seed_signed_in_server(home, base):
    (home / "config.yaml").write_text(f"mcp_servers:\n  srv:\n    url: {base}/mcp\n    auth: oauth\n", encoding="utf-8")
    tokens = home / "mcp-tokens"
    tokens.mkdir()
    (tokens / "srv.json").write_text(json.dumps({"access_token": "acc-1", "refresh_token": "ref-1", "token_type": "Bearer"}), encoding="utf-8")
    (tokens / "srv.client.json").write_text(json.dumps({
        "client_id": "cid", "client_secret": "shh", "token_endpoint_auth_method": "client_secret_post"}), encoding="utf-8")
    (tokens / "srv.meta.json").write_text(json.dumps({"issuer": base, "revocation_endpoint": f"{base}/revoke"}), encoding="utf-8")
    return tokens


@pytest.mark.parametrize("stall", [False, True], ids=["revokes", "stalled-server-does-not-block"])
def test_removing_server_revokes_then_forgets_tokens(tmp_path, monkeypatch, stall):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("tools.mcp_oauth_revoke.REVOKE_BUDGET_SECONDS", 1.0)
    server = _RevokeServer(stall=stall)
    try:
        tokens = _seed_signed_in_server(tmp_path, server.base)
        from hermes_cli.mcp_config import _remove_mcp_server
        outcomes = []
        started = time.monotonic()
        assert _remove_mcp_server("srv", report=outcomes.append)
        assert time.monotonic() - started < 5
        assert list(tokens.iterdir()) == []
        assert "srv" not in (tmp_path / "config.yaml").read_text(encoding="utf-8-sig")
        if stall:
            assert "incomplete" in outcomes[0]
        else:
            assert [path for path, _ in server.calls] == ["/revoke", "/revoke"]
            assert "token=ref-1&token_type_hint=refresh_token" in server.calls[0][1]
            assert "token=acc-1&token_type_hint=access_token" in server.calls[1][1]
            assert "client_secret=shh" in server.calls[0][1]
            assert "revoked 2 tokens" in outcomes[0]
    finally:
        server.close()
