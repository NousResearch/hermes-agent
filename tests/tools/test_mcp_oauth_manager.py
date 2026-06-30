"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""
import json
import os
import time
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "mcp.client.auth.oauth2",
    reason="MCP SDK 1.26.0+ required for OAuth support",
)


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


def test_manager_is_singleton():
    """get_manager() returns the same instance across calls."""
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()
    m1 = get_manager()
    m2 = get_manager()
    assert m1 is m2


def test_manager_get_or_build_provider_caches(tmp_path, monkeypatch):
    """Calling get_or_build_provider twice with same name returns same provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is p2


def test_manager_get_or_build_rebuilds_on_url_change(tmp_path, monkeypatch):
    """Changing the URL discards the cached provider."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://a.example.com/mcp", None)
    p2 = mgr.get_or_build_provider("srv", "https://b.example.com/mcp", None)
    assert p1 is not p2


def test_manager_remove_evicts_cache(tmp_path, monkeypatch):
    """remove(name) evicts the provider from cache AND deletes disk files."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    # Pre-seed tokens on disk
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({
        "access_token": "TOK",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not None
    assert (token_dir / "srv.json").exists()

    mgr.remove("srv")

    assert not (token_dir / "srv.json").exists()
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert p1 is not p2


def test_hermes_provider_subclass_exists():
    """HermesMCPOAuthProvider is defined and subclasses OAuthClientProvider."""
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.client.auth.oauth2 import OAuthClientProvider

    assert _HERMES_PROVIDER_CLS is not None
    assert issubclass(_HERMES_PROVIDER_CLS, OAuthClientProvider)


@pytest.mark.asyncio
async def test_disk_watch_invalidates_on_mtime_change(tmp_path, monkeypatch):
    """When the tokens file mtime changes, provider._initialized flips False.

    This is the behaviour Claude Code ships as
    invalidateOAuthCacheIfDiskChanged (CC-1096 / GH#24317) and is the core
    fix for Cthulhu's external-cron refresh workflow.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests

    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    tokens_file.write_text(json.dumps({
        "access_token": "OLD",
        "token_type": "Bearer",
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # First call: records mtime (zero -> real) -> returns True
    changed1 = await mgr.invalidate_if_disk_changed("srv")
    assert changed1 is True

    # No file change -> False
    changed2 = await mgr.invalidate_if_disk_changed("srv")
    assert changed2 is False

    # Touch file with a newer mtime
    future_mtime = time.time() + 10
    os.utime(tokens_file, (future_mtime, future_mtime))

    changed3 = await mgr.invalidate_if_disk_changed("srv")
    assert changed3 is True
    # _initialized flipped — next async_auth_flow will re-read from disk
    assert provider._initialized is False


def test_manager_builds_hermes_provider_subclass(tmp_path, monkeypatch):
    """get_or_build_provider returns HermesMCPOAuthProvider, not plain OAuthClientProvider."""
    from tools.mcp_oauth_manager import (
        MCPOAuthManager, _HERMES_PROVIDER_CLS, reset_manager_for_tests,
    )
    reset_manager_for_tests()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert _HERMES_PROVIDER_CLS is not None
    assert isinstance(provider, _HERMES_PROVIDER_CLS)
    assert provider._hermes_server_name == "srv"


def test_manager_fails_fast_noninteractive_without_cached_tokens(tmp_path, monkeypatch):
    """A daemon without cached MCP OAuth tokens must not enter browser auth."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)
    from tools.mcp_oauth import OAuthNonInteractiveError
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()

    with pytest.raises(OAuthNonInteractiveError, match="non-interactive"):
        mgr.get_or_build_provider("linear", "https://mcp.linear.app/mcp", None)

    assert mgr._entries["linear"].provider is None


# ---------------------------------------------------------------------------
# invalid_client auto-heal (GH#36767) — _maybe_flag_poisoned_client
# ---------------------------------------------------------------------------

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock


def _fake_response(status, url, body):
    """A minimal stand-in for the httpx.Response the SDK feeds our bridge."""
    resp = MagicMock()
    resp.status_code = status
    resp.request = SimpleNamespace(url=url)

    async def _aread():
        return body

    resp.aread = _aread
    return resp


def _provider_with_token_endpoint(tmp_path, oauth_config, token_endpoint, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()
    # Provider construction fails fast in a non-interactive environment with no
    # cached tokens (mcp_oauth_manager.py guard). The hermetic test env has no
    # TTY, so present an interactive stdin to reach the code under test.
    _set_interactive_stdin(monkeypatch)
    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://mcp.example.com", oauth_config)
    provider.context.oauth_metadata = SimpleNamespace(token_endpoint=token_endpoint)
    provider._initialized = True
    return provider


def test_invalid_client_at_token_endpoint_poisons(tmp_path, monkeypatch):
    """400 invalid_client on the token endpoint deletes the dead client.json."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')
    (d / "srv.meta.json").write_text("{}")
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert not (d / "srv.client.json").exists()
    assert (d / "srv.client.json.bak").exists()
    assert provider._initialized is False
    assert provider.context.client_info is None


def test_invalid_client_at_other_endpoint_is_ignored(tmp_path, monkeypatch):
    """An invalid_client body from a non-token endpoint must not poison."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://mcp.example.com/messages", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_success_response_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        200, "https://idp.example.com/oauth/token", b'{"access_token":"x"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_preregistered_client_is_never_poisoned(tmp_path, monkeypatch):
    """A config-supplied client_id is never auto-deleted (re-reg can't help)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = _provider_with_token_endpoint(
        tmp_path, {"client_id": "from-config"}, "https://idp.example.com/oauth/token", monkeypatch
    )
    d = tmp_path / "mcp-tokens"
    # _maybe_preregister_client wrote client.json from config during build.
    assert (d / "srv.client.json").exists()
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


def test_invalid_client_metadata_does_not_trip(tmp_path, monkeypatch):
    """RFC 7591 `invalid_client_metadata` must NOT be mistaken for invalid_client."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "live"}')
    provider = _provider_with_token_endpoint(
        tmp_path, {}, "https://idp.example.com/oauth/token", monkeypatch
    )
    resp = _fake_response(
        400, "https://idp.example.com/oauth/token", b'{"error":"invalid_client_metadata"}'
    )

    asyncio.run(provider._maybe_flag_poisoned_client(resp))

    assert (d / "srv.client.json").exists()
    assert provider._initialized is True


class _FakeMeta:
    """Metadata stub usable by both detection and the post-flow persist hook."""

    def __init__(self, token_endpoint):
        self.token_endpoint = token_endpoint

    def model_dump(self, **kwargs):
        return {"token_endpoint": self.token_endpoint}


def test_bridge_forwards_requests_and_poisons_on_token_endpoint_400(
    tmp_path, monkeypatch
):
    """Drive the REAL async_auth_flow bridge to prove the inserted detection
    hook does not break the bidirectional asend() forwarding contract — the
    genuinely fragile part. A patched SDK base generator stands in for the
    real OAuth flow so we control exactly which response the bridge sees.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')

    forwarded = []

    async def fake_base_flow(self, request):
        # Mimic the SDK: yield the request, receive the response, then finish.
        forwarded.append(("out", request))
        response = yield request
        forwarded.append(("in", response))

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider.context.oauth_metadata = _FakeMeta(token_ep)

    sentinel_request = object()
    poison_resp = _fake_response(400, token_ep, b'{"error":"invalid_client"}')

    async def drive():
        gen = provider.async_auth_flow(sentinel_request)
        out0 = await gen.__anext__()
        assert out0 is sentinel_request  # request forwarded unchanged
        try:
            await gen.asend(poison_resp)
        except StopAsyncIteration:
            pass

    asyncio.run(drive())

    # The poison response reached the inner generator (forwarding intact)...
    assert ("in", poison_resp) in forwarded
    # ...and the detection hook fired.
    assert not (d / "srv.client.json").exists()
    assert provider._initialized is False
    assert provider.context.client_info is None



# ---------------------------------------------------------------------------
# Force-OAuth tests (#53870)
# ---------------------------------------------------------------------------


def test_set_force_oauth_adds_name_to_set():
    """set_force_oauth() adds the server name to _force_oauth_names."""
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    assert "blynk" not in mgr._force_oauth_names
    mgr.set_force_oauth("blynk")
    assert "blynk" in mgr._force_oauth_names


def test_build_provider_sets_force_oauth_flag(tmp_path, monkeypatch):
    """_build_provider sets force_oauth=True when name is in _force_oauth_names."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    mgr.set_force_oauth("srv")
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    assert provider._force_oauth is True
    # Name should be consumed (one-shot)
    assert "srv" not in mgr._force_oauth_names


def test_build_provider_without_force_oauth_flag(tmp_path, monkeypatch):
    """_build_provider sets force_oauth=False when name is NOT in _force_oauth_names."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    assert provider._force_oauth is False


@pytest.mark.asyncio
async def test_force_oauth_converts_200_to_401(tmp_path, monkeypatch):
    """When _force_oauth is set and no tokens exist, a 200 response is
    replaced with a synthetic 401 to trigger the SDK's OAuth flow.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')

    forwarded = []
    oAuth_triggered = []

    async def fake_base_flow(self, request):
        # Mimic the SDK: yield request, receive response.
        # If response is 401, record that OAuth was triggered.
        forwarded.append(("out", request))
        response = yield request
        forwarded.append(("in", response))
        if response.status_code == 401:
            oAuth_triggered.append(True)

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider._force_oauth = True
    provider.context.oauth_metadata = None  # avoid model_dump on SimpleNamespace
    # Ensure no valid tokens
    provider.context.current_tokens = None

    sentinel_request = object()
    ok_resp = _fake_response(200, "https://mcp.example.com", b'{"result":{}}')

    gen = provider.async_auth_flow(sentinel_request)
    out0 = await gen.__anext__()
    assert out0 is sentinel_request
    try:
        await gen.asend(ok_resp)
    except StopAsyncIteration:
        pass

    # The bridge should have converted the 200 to a 401
    assert len(forwarded) == 2
    assert forwarded[1][1].status_code == 401  # synthetic 401 reached inner
    assert oAuth_triggered  # OAuth flow was triggered
    # _force_oauth should be cleared (one-shot)
    assert provider._force_oauth is False


@pytest.mark.asyncio
async def test_force_oauth_does_not_override_real_401(tmp_path, monkeypatch):
    """When the server returns a real 401, _force_oauth does not interfere."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')

    forwarded = []

    async def fake_base_flow(self, request):
        forwarded.append(("out", request))
        response = yield request
        forwarded.append(("in", response))

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider._force_oauth = True
    provider.context.oauth_metadata = None  # avoid model_dump on SimpleNamespace
    provider.context.current_tokens = None

    sentinel_request = object()
    real_401 = _fake_response(401, "https://mcp.example.com", b'{"error":"unauthorized"}')

    gen = provider.async_auth_flow(sentinel_request)
    out0 = await gen.__anext__()
    assert out0 is sentinel_request
    try:
        await gen.asend(real_401)
    except StopAsyncIteration:
        pass

    # The real 401 should pass through unchanged
    assert forwarded[1][1].status_code == 401
    assert forwarded[1][1] is real_401  # same object, not synthetic
    # _force_oauth should still be cleared (one-shot)
    assert provider._force_oauth is False


@pytest.mark.asyncio
async def test_force_oauth_not_applied_when_tokens_valid(tmp_path, monkeypatch):
    """When valid tokens exist, _force_oauth does not convert 200 to 401."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token_ep = "https://idp.example.com/oauth/token"
    d = tmp_path / "mcp-tokens"
    d.mkdir(parents=True)
    (d / "srv.client.json").write_text('{"client_id": "dead"}')

    forwarded = []

    async def fake_base_flow(self, request):
        forwarded.append(("out", request))
        response = yield request
        forwarded.append(("in", response))

    from mcp.client.auth.oauth2 import OAuthClientProvider
    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", fake_base_flow)

    provider = _provider_with_token_endpoint(tmp_path, {}, token_ep, monkeypatch)
    provider._force_oauth = True
    provider.context.oauth_metadata = None  # avoid model_dump on SimpleNamespace
    # Simulate valid tokens by making is_token_valid() return True
    from mcp.shared.auth import OAuthToken
    provider.context.current_tokens = OAuthToken(
        access_token="valid", token_type="Bearer", expires_in=3600
    )
    provider.context.update_token_expiry(provider.context.current_tokens)

    sentinel_request = object()
    ok_resp = _fake_response(200, "https://mcp.example.com", b'{"result":{}}')

    gen = provider.async_auth_flow(sentinel_request)
    out0 = await gen.__anext__()
    assert out0 is sentinel_request
    try:
        await gen.asend(ok_resp)
    except StopAsyncIteration:
        pass

    # 200 should pass through unchanged (tokens are valid)
    assert forwarded[1][1].status_code == 200
    assert forwarded[1][1] is ok_resp

