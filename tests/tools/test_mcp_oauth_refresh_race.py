"""Processes sharing one MCP tokens file must not race each other's refresh.

The gateway, the dashboard and the desktop app backend each hold an MCP OAuth provider over the
same ``mcp-tokens/<server>.json``. At expiry all refreshed in the same second; the losers sent a
rotated-out refresh token (400, then 429 from the authorization server) and the server was parked.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK required")

try:
    import fcntl
except ImportError:  # Windows: the refresh lock is skipped, see test_refresh_lock_is_skipped_without_fcntl
    fcntl = None
posix_locks = pytest.mark.skipif(fcntl is None, reason="POSIX advisory file locks")


_REAL_HTTPX = []


def _metadata_network(monkeypatch, prm_resource=None):
    """Well-known discovery without the network: PRM for *prm_resource*, 404 otherwise."""
    import tools.mcp_tool as mcp_tool
    if not _REAL_HTTPX:
        _REAL_HTTPX.append(mcp_tool.sdk_httpx())
    real = _REAL_HTTPX[0]
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if prm_resource and "oauth-protected-resource" in str(request.url):
            return real.Response(200, json={"resource": prm_resource, "authorization_servers": ["https://idp.example.com"]})
        return real.Response(404)

    class Client(real.AsyncClient):
        def __init__(self, **kwargs):
            super().__init__(transport=real.MockTransport(handler), **kwargs)

    monkeypatch.setattr(mcp_tool, "sdk_httpx", lambda: SimpleNamespace(AsyncClient=Client, HTTPError=real.HTTPError))
    return seen


@pytest.fixture(autouse=True)
def _no_metadata_network(monkeypatch):
    _metadata_network(monkeypatch)


def _write_tokens(directory, access, refresh, expires_at):
    (directory / "srv.json").write_text(json.dumps({
        "access_token": access, "token_type": "Bearer", "expires_in": 3600,
        "refresh_token": refresh, "expires_at": expires_at}), encoding="utf-8")


def _provider(tmp_path, monkeypatch, *, expires_at):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    tokens = tmp_path / "mcp-tokens"
    tokens.mkdir(parents=True, exist_ok=True)
    _write_tokens(tokens, "OLD", "refresh-1", expires_at)
    (tokens / "srv.client.json").write_text('{"client_id": "client"}', encoding="utf-8")
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    provider = get_manager().get_or_build_provider("srv", "https://mcp.example.com", None)
    provider.context.oauth_metadata = SimpleNamespace(token_endpoint="https://idp.example.com/token")
    return provider, tokens


def _response(status):
    response = MagicMock()
    response.status_code = status
    response.request = SimpleNamespace(url="https://idp.example.com/token")
    return response


@pytest.mark.asyncio
async def test_each_process_refreshes_early_by_its_own_margin(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthToken
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() + 3600)
    before = time.time()
    await provider._store_tokens(OAuthToken(access_token="A", token_type="Bearer", expires_in=3600, refresh_token="r"))
    margin = before + 3600 - provider.context.token_expiry_time
    assert 0.05 * 3600 - 2 <= margin <= 0.15 * 3600 + 2


@pytest.mark.asyncio
async def test_failed_refresh_adopts_tokens_another_process_wrote(tmp_path, monkeypatch):
    provider, tokens = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()
    assert not provider.context.is_token_valid()
    _write_tokens(tokens, "NEW", "refresh-2", time.time() + 3600)  # the winner's refresh

    assert await provider._handle_refresh_response(_response(400)) is True
    assert provider.context.current_tokens.access_token == "NEW"
    assert provider.context.current_tokens.refresh_token == "refresh-2"
    assert provider.context.is_token_valid()


@pytest.mark.asyncio
async def test_failed_refresh_without_newer_tokens_still_clears(tmp_path, monkeypatch):
    import anyio
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()
    slept = []

    async def no_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    assert await provider._handle_refresh_response(_response(429)) is False
    assert provider.context.current_tokens is None
    assert len(slept) == 1  # re-read the file once after a short wait


@posix_locks
@pytest.mark.asyncio
async def test_waiter_uses_the_lock_holders_refresh_instead_of_refreshing(tmp_path, monkeypatch):
    provider, tokens = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    from tools.mcp_oauth_manager import get_manager
    await get_manager().invalidate_if_disk_changed("srv")  # the pre-flow disk watch has seen the old file
    other_process = open(tokens / "srv.json.refresh.lock", "a+")
    fcntl.flock(other_process.fileno(), fcntl.LOCK_EX)

    waiter = asyncio.create_task(provider._hermes_refresh_lock_if_due())
    await asyncio.sleep(0.3)
    assert not waiter.done()  # blocked on the other process's refresh
    _write_tokens(tokens, "NEW", "refresh-2", time.time() + 3600)
    fcntl.flock(other_process.fileno(), fcntl.LOCK_UN)
    other_process.close()

    assert await asyncio.wait_for(waiter, 5) is None  # nothing left to refresh
    assert provider.context.current_tokens.access_token == "NEW"


@posix_locks
@pytest.mark.asyncio
async def test_due_refresh_holds_the_lock_until_released(tmp_path, monkeypatch):
    provider, tokens = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    lock = await provider._hermes_refresh_lock_if_due()
    assert lock is not None
    probe = open(tokens / "srv.json.refresh.lock", "a+")
    with pytest.raises(BlockingIOError):
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    provider._hermes_release_refresh_lock(lock)
    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe.close()


@pytest.mark.asyncio
async def test_refresh_lock_is_skipped_without_fcntl(tmp_path, monkeypatch):
    """Windows has no fcntl: a due refresh goes ahead without the cross-process lock."""
    provider, _tokens = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    monkeypatch.setitem(sys.modules, "fcntl", None)  # makes ``import fcntl`` raise ImportError
    assert await provider._hermes_refresh_lock_if_due() is None
    provider._hermes_release_refresh_lock(open(tmp_path / "unused.lock", "a+"))  # must not raise


def _refresh_body(request) -> str:
    from urllib.parse import unquote_plus
    return unquote_plus(request.content.decode())


@pytest.mark.asyncio
async def test_refresh_from_an_initialize_request_still_names_the_resource(tmp_path, monkeypatch):
    """A new session's ``initialize`` has no MCP-Protocol-Version header; its refresh must be built
    like any other (with RFC 8707 ``resource``), or Cloudflare rejects the same token with 400."""
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()
    provider.context.protected_resource_metadata = None
    provider.context.protocol_version = None  # what the SDK sets for a header-less request
    assert "resource=https://mcp.example.com" in _refresh_body(await provider._refresh_token())


@pytest.mark.asyncio
async def test_refresh_uses_the_last_negotiated_protocol_version(tmp_path, monkeypatch):
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()
    provider.context.protected_resource_metadata = None
    provider._hermes_protocol_version = "2025-03-26"  # a server that negotiated a pre-8707 version
    provider.context.protocol_version = None
    assert "resource=" not in _refresh_body(await provider._refresh_token())


@pytest.mark.asyncio
async def test_refresh_names_the_grants_resource_not_the_url_query(tmp_path, monkeypatch):
    """Cloudflare's server URL is /mcp?codemode=false but its grant is for /mcp (PRM resource). A
    cold-loaded provider has no PRM, so the SDK named the URL with its query and got 400."""
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()
    provider.context.server_url = "https://mcp.example.com/mcp?codemode=false"
    provider.context.protected_resource_metadata = None
    provider.context.protocol_version = "2025-06-18"
    seen = _metadata_network(monkeypatch, prm_resource="https://mcp.example.com/mcp")

    body = _refresh_body(await provider._refresh_token()) + "&"
    assert "resource=https://mcp.example.com/mcp&" in body
    assert "codemode" not in body
    assert any("oauth-protected-resource" in url for url in seen)

    count = sum("oauth-protected-resource" in url for url in seen)
    await provider._refresh_token()  # PRM is fetched once per provider, not per refresh
    assert sum("oauth-protected-resource" in url for url in seen) == count


def _refused(status, error):
    response = _response(status)

    async def aread():
        return json.dumps({"error": error, "error_description": "secret-free detail"}).encode()

    response.aread = aread
    return response


@pytest.mark.asyncio
async def test_early_refresh_refused_keeps_the_valid_token_and_waits_for_expiry(tmp_path, monkeypatch, caplog):
    """Cloudflare answers 400 to a refresh while the access token is still valid: keep the token,
    stop refreshing early for this server, and refresh at its real expiry."""
    import logging
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() + 1800)
    await provider._initialize()
    true_expiry = provider._hermes_true_expiry
    provider.context.token_expiry_time = time.time() - 1  # the early-refresh margin made it due

    with caplog.at_level(logging.INFO, logger="tools.mcp_oauth_manager"):
        assert await provider._handle_refresh_response(_refused(400, "invalid_grant")) is True
    assert provider.context.current_tokens.access_token == "OLD"
    assert provider.context.token_expiry_time == true_expiry
    assert provider._hermes_refresh_skew == 0.0
    assert "invalid_grant" in caplog.text and "secret-free detail" not in caplog.text


@pytest.mark.asyncio
async def test_refusal_after_expiry_still_clears_and_logs_the_error_code(tmp_path, monkeypatch, caplog):
    import anyio
    import logging
    provider, _ = _provider(tmp_path, monkeypatch, expires_at=time.time() - 5)
    await provider._initialize()

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(anyio, "sleep", no_sleep)
    with caplog.at_level(logging.WARNING, logger="tools.mcp_oauth_manager"):
        assert await provider._handle_refresh_response(_refused(400, "invalid_grant")) is False
    assert provider.context.current_tokens is None
    assert "invalid_grant" in caplog.text
