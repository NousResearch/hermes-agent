"""Every OAuth request the MCP auth flow sends carries a timeout.

httpx applies a client's ``timeout`` only to the request handed to ``send()``; the requests an
``httpx.Auth`` flow yields (discovery, registration, token, refresh) go out with no timeout at all.
A token endpoint that accepts the TCP connection and never answers the TLS handshake therefore hung
the refresh, and every request queued behind the provider's lock, for as long as the caller's own
budget allowed. These tests use a real TCP peer that accepts and stays silent, and the real
``HermesMCPOAuthProvider`` driven by a real SDK httpx client.
"""

import socket
import threading
import time

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required for OAuth support")


@pytest.fixture
def silent_tls_peer():
    """A listener that accepts connections and never sends a byte (a stalled TLS handshake)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    held, stop = [], threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                held.append(srv.accept()[0])
            except OSError:
                continue

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield srv.getsockname()[1]
    stop.set()
    thread.join(2)
    for conn in held:
        conn.close()
    srv.close()


def _expired_provider(tmp_path, monkeypatch, token_endpoint):
    from unittest.mock import MagicMock

    from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stdin = MagicMock()
    stdin.isatty.return_value = True
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", stdin)
    reset_manager_for_tests()
    provider = MCPOAuthManager().get_or_build_provider("srv", "http://127.0.0.1:9/mcp", {})
    provider._initialized = True
    provider.context.oauth_metadata = OAuthMetadata(
        issuer="http://127.0.0.1:9", authorization_endpoint="http://127.0.0.1:9/authorize",
        token_endpoint=token_endpoint)
    provider.context.client_info = OAuthClientInformationFull.model_validate(
        {"client_id": "cid", "redirect_uris": ["http://127.0.0.1:1/cb"]})
    provider.context.current_tokens = OAuthToken(
        access_token="A1", token_type="Bearer", expires_in=1, refresh_token="R1")
    provider.context.token_expiry_time = time.time() - 10
    return provider


@pytest.mark.asyncio
async def test_refresh_against_a_silent_tls_peer_times_out(silent_tls_peer, tmp_path, monkeypatch):
    import anyio
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    provider = _expired_provider(tmp_path, monkeypatch, f"https://127.0.0.1:{silent_tls_peer}/token")

    started = time.monotonic()
    # The client's own timeout never reaches a request the auth flow yields, so before the fix the
    # refresh waited on the silent peer until this outer cap fired.
    with anyio.fail_after(40):
        async with httpx.AsyncClient(auth=provider, timeout=httpx.Timeout(300)) as client:
            with pytest.raises(httpx.TimeoutException):
                await client.get("http://127.0.0.1:9/mcp")
    assert time.monotonic() - started < 30
    # The provider is usable afterwards: the abandoned refresh released its fence.
    assert provider._hermes_fence is None

