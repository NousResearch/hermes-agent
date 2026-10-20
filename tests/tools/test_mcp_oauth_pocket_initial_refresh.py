"""A rejected INITIAL (expired-token) refresh on a Pocket read fails closed before the SDK's interactive OAuth.

The SDK clears the tokens after a rejected refresh, sends the resource unauthenticated and runs full
discovery/authorization on its 401; for an allow-listed read that must stop at the rejection."""

import pytest

from tests.tools.test_mcp_oauth_pocket_recovery_bounds import (
    TOKEN_URL, _assert_idle, _assert_no_interactive, _exhausted_cls, _httpx, _provider, _rpc_request, _Server,
)

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")


def _rejecting(server, status, before_reject=None):
    async def handler(request, _orig=server.handler):
        if str(request.url) == TOKEN_URL and server._kind(request) == "refresh":
            server.log.append(("refresh", ""))
            if before_reject is not None:
                await before_reject()
            return _httpx().Response(status, json={"error": "invalid_grant"})
        return await _orig(request)
    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403])
async def test_rejected_initial_refresh_fails_closed_before_any_oauth(tmp_path, monkeypatch, status):
    provider, _, interactive = await _provider(tmp_path, monkeypatch, expired=True)
    server = _Server()
    server.resource_status = [401, 200]
    httpx = _httpx()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_rejecting(server, status)), auth=provider) as client:
        with pytest.raises(_exhausted_cls()):
            await client.send(_rpc_request())
    assert [k for k, _ in server.log] == ["refresh"]  # no unauthenticated resource send, no discovery
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)


@pytest.mark.asyncio
async def test_rejected_initial_refresh_adopts_a_peer_rotated_grant(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthToken
    provider, storage, interactive = await _provider(tmp_path, monkeypatch, expired=True)
    server = _Server()
    server.resource_status = [200]

    async def peer_rotates():
        await storage.set_tokens(OAuthToken(access_token="at-peer", token_type="Bearer", expires_in=3600,
                                            refresh_token="rt-peer"))
    httpx = _httpx()
    async with httpx.AsyncClient(transport=httpx.MockTransport(_rejecting(server, 400, peer_rotates)),
                                 auth=provider) as client:
        response = await client.send(_rpc_request())
    assert response.status_code == 200
    assert server.log == [("refresh", ""), ("resource", "Bearer at-peer")]
    _assert_no_interactive(server, interactive)
    _assert_idle(provider)
