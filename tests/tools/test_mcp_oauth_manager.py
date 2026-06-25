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


@pytest.mark.asyncio
async def test_hermes_provider_sends_dynamic_client_secret_when_method_missing(tmp_path, monkeypatch):
    """Supabase-style DCR client_secret without method still reaches token body."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata

    assert _HERMES_PROVIDER_CLS is not None
    metadata = OAuthClientMetadata.model_validate({
        "redirect_uris": ["http://127.0.0.1:58007/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })
    provider = _HERMES_PROVIDER_CLS(
        server_name="supabase",
        server_url="https://mcp.supabase.com/mcp",
        client_metadata=metadata,
        storage=HermesTokenStorage("supabase"),
        redirect_handler=None,
        callback_handler=None,
    )
    provider.context.client_info = OAuthClientInformationFull.model_validate({
        "client_id": "client-123",
        "client_secret": "secret-456",
        "redirect_uris": ["http://127.0.0.1:58007/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })

    request = await provider._exchange_token_authorization_code("auth-code", "verifier")
    body = request.content.decode()

    assert provider.context.client_info.token_endpoint_auth_method == "client_secret_post"
    assert "client_secret=secret-456" in body


@pytest.mark.asyncio
async def test_hermes_provider_accepts_supabase_201_token_response(tmp_path, monkeypatch):
    """Supabase token exchange returns 201 Created with valid token JSON."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import httpx
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.shared.auth import OAuthClientMetadata

    assert _HERMES_PROVIDER_CLS is not None
    metadata = OAuthClientMetadata.model_validate({
        "redirect_uris": ["http://127.0.0.1:58007/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })
    provider = _HERMES_PROVIDER_CLS(
        server_name="supabase",
        server_url="https://mcp.supabase.com/mcp",
        client_metadata=metadata,
        storage=HermesTokenStorage("supabase"),
        redirect_handler=None,
        callback_handler=None,
    )
    response = httpx.Response(
        201,
        json={
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 86400,
            "token_type": "Bearer",
        },
        request=httpx.Request("POST", "https://api.supabase.com/oauth/token"),
    )

    await provider._handle_token_response(response)

    assert provider.context.current_tokens is not None
    assert provider.context.current_tokens.access_token == "access-123"
    assert (tmp_path / "mcp-tokens" / "supabase.json").exists()


@pytest.mark.asyncio
async def test_hermes_provider_accepts_supabase_201_refresh_response(tmp_path, monkeypatch):
    """Supabase refresh returns 201 Created with valid token JSON."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import httpx
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS
    from mcp.shared.auth import OAuthClientMetadata, OAuthToken

    assert _HERMES_PROVIDER_CLS is not None
    metadata = OAuthClientMetadata.model_validate({
        "redirect_uris": ["http://127.0.0.1:58007/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    })
    provider = _HERMES_PROVIDER_CLS(
        server_name="supabase",
        server_url="https://mcp.supabase.com/mcp",
        client_metadata=metadata,
        storage=HermesTokenStorage("supabase"),
        redirect_handler=None,
        callback_handler=None,
    )
    provider.context.current_tokens = OAuthToken(
        access_token="old-access",
        token_type="Bearer",
        expires_in=0,
        refresh_token="old-refresh",
    )
    response = httpx.Response(
        201,
        json={
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 86400,
            "token_type": "Bearer",
        },
        request=httpx.Request("POST", "https://api.supabase.com/oauth/token"),
    )

    assert await provider._handle_refresh_response(response) is True

    assert provider.context.current_tokens is not None
    assert provider.context.current_tokens.access_token == "new-access"
    token_file = tmp_path / "mcp-tokens" / "supabase.json"
    assert token_file.exists()
    assert json.loads(token_file.read_text())["access_token"] == "new-access"
