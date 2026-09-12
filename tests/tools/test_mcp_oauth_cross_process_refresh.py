"""Two Hermes processes (gateway + desktop backend, or gateway + CLI) share one MCP tokens file.
An authorization server that ROTATES refresh tokens (single use, e.g. CoRecruit) rejects the
second process's refresh with 400 when both try around the same expiry; the SDK then cleared
tokens that the sibling had just validly written, and the server sat in "needs re-auth" until a
human ran ``hermes mcp login`` again.

These tests drive two provider instances against one storage the way two processes would, with a
fake token endpoint that rotates the refresh token and rejects a reused one.
"""
from __future__ import annotations

import pytest


async def _noop_redirect(_url: str) -> None:  # pragma: no cover
    return None


async def _noop_callback():  # pragma: no cover
    return ("code", None)


class _RotatingTokenServer:
    """Refresh-token grant with rotation: the current refresh token is valid exactly once."""

    def __init__(self, httpx, current_refresh: str):
        self.httpx = httpx
        self.valid_refresh = current_refresh
        self.issued = 0
        self.rejected = 0

    def respond(self, request):
        body = dict(pair.split("=", 1) for pair in request.content.decode().split("&"))
        if body.get("grant_type") != "refresh_token" or body.get("refresh_token") != self.valid_refresh:
            self.rejected += 1
            return self.httpx.Response(400, json={"error": "invalid_grant"}, request=request)
        self.issued += 1
        self.valid_refresh = f"refresh_{self.issued}"
        return self.httpx.Response(200, json={
            "access_token": f"access_{self.issued}", "token_type": "Bearer", "expires_in": 900,
            "refresh_token": self.valid_refresh, "scope": "admin",
        }, request=request)


async def _seed(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyHttpUrl, AnyUrl
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests
    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage("srv")
    # Expired access token + refresh token: every provider that loads this must refresh first.
    await storage.set_tokens(OAuthToken(access_token="access_0", token_type="Bearer", expires_in=0, refresh_token="refresh_0"))
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="test-client", redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"], token_endpoint_auth_method="none"))
    from mcp.shared.auth import OAuthMetadata
    storage.save_oauth_metadata(OAuthMetadata(
        issuer=AnyHttpUrl("https://as.example"), authorization_endpoint=AnyHttpUrl("https://as.example/authorize"),
        token_endpoint=AnyHttpUrl("https://as.example/token"), response_types_supported=["code"],
        grant_types_supported=["authorization_code", "refresh_token"]))

    def build():
        return _HERMES_PROVIDER_CLS(
            server_name="srv", server_url="https://example.com/mcp",
            client_metadata=OAuthClientMetadata(redirect_uris=[AnyUrl("http://127.0.0.1:12345/callback")], client_name="Hermes Agent"),
            storage=HermesTokenStorage("srv"), redirect_handler=_noop_redirect, callback_handler=_noop_callback)
    return storage, build


async def _run_flow(provider, httpx, server):
    """Drive one auth flow: answer token-endpoint requests with the rotating server, answer the
    resource request with 200. Returns the Bearer token that was sent to the resource."""
    req = httpx.Request("POST", "https://example.com/mcp")
    flow = provider.async_auth_flow(req)
    outgoing = await flow.__anext__()
    sent_bearer = None
    while True:
        if outgoing.url.host == "as.example":
            response = server.respond(outgoing)
        else:
            sent_bearer = outgoing.headers.get("Authorization")
            response = httpx.Response(200, request=outgoing)
        try:
            outgoing = await flow.asend(response)
        except StopAsyncIteration:
            return sent_bearer


@pytest.mark.asyncio
async def test_second_process_adopts_siblings_rotated_tokens_instead_of_clearing(tmp_path, monkeypatch):
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    storage, build = await _seed(tmp_path, monkeypatch)
    server = _RotatingTokenServer(httpx, "refresh_0")

    gateway, desktop = build(), build()
    # Both processes load the same expired pair into memory (each holds refresh_0).
    await gateway._initialize()
    await desktop._initialize()

    # Process A refreshes: the server rotates to refresh_1 and A writes it to disk.
    assert await _run_flow(gateway, httpx, server) == "Bearer access_1"
    assert server.issued == 1

    # Process B still holds refresh_0 in memory. Before the fix it POSTed refresh_0, got 400,
    # cleared tokens and wiped the file. It must instead adopt A's pair from disk.
    assert await _run_flow(desktop, httpx, server) == "Bearer access_1"
    assert server.rejected == 0, "stale refresh token must never be sent when disk holds a newer pair"
    assert server.issued == 1, "no second refresh needed: sibling's token is still valid"

    persisted = await storage.get_tokens()
    assert persisted is not None and persisted.refresh_token == "refresh_1", "tokens file must survive"


@pytest.mark.asyncio
async def test_rejected_refresh_prefers_disk_over_clearing(tmp_path, monkeypatch):
    """If a stale refresh does reach the server (lock timed out, guard skipped), the 400 must not
    destroy a newer pair on disk: adopt it and report success instead of forcing browser re-auth."""
    from mcp.shared.auth import OAuthToken
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    storage, build = await _seed(tmp_path, monkeypatch)
    provider = build()
    await provider._initialize()  # memory: refresh_0 (expired access)

    # Sibling process already rotated and wrote a fresh, valid pair to disk.
    await storage.set_tokens(OAuthToken(access_token="access_1", token_type="Bearer", expires_in=900, refresh_token="refresh_1"))

    rejected = httpx.Response(400, json={"error": "invalid_grant"}, request=httpx.Request("POST", "https://as.example/token"))
    assert await provider._handle_refresh_response(rejected) is True, "newer pair on disk means the refresh outcome is success"
    assert provider.context.current_tokens.access_token == "access_1"
    assert provider.context.is_token_valid()
    persisted = await storage.get_tokens()
    assert persisted is not None and persisted.refresh_token == "refresh_1", "400 must not wipe the sibling's tokens"


@pytest.mark.asyncio
async def test_rejected_refresh_with_no_newer_disk_pair_still_clears(tmp_path, monkeypatch):
    """The fallback only rescues a genuine sibling refresh; a dead grant still clears tokens."""
    from tools.mcp_tool import sdk_httpx
    httpx = sdk_httpx()
    storage, build = await _seed(tmp_path, monkeypatch)
    provider = build()
    await provider._initialize()
    rejected = httpx.Response(400, json={"error": "invalid_grant"}, request=httpx.Request("POST", "https://as.example/token"))
    assert await provider._handle_refresh_response(rejected) is False
    assert provider.context.current_tokens is None
