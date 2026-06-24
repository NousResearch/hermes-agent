"""Tests for the MCP OAuth manager (tools/mcp_oauth_manager.py).

The manager consolidates the eight scattered MCP-OAuth call sites into a
single object with disk-mtime watch, dedup'd 401 handling, and a provider
cache. See `tools/mcp_oauth_manager.py` for design rationale.
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

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
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

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
    tokens_file.write_text(
        json.dumps({
            "access_token": "OLD",
            "token_type": "Bearer",
        })
    )

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
        MCPOAuthManager,
        _HERMES_PROVIDER_CLS,
        reset_manager_for_tests,
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


# invalidate_if_disk_changed — edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalidate_returns_false_for_unknown_server(tmp_path, monkeypatch):
    """No entry for the server → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    result = await mgr.invalidate_if_disk_changed("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_invalidate_returns_false_when_provider_is_none(tmp_path, monkeypatch):
    """Entry exists but provider is None (build failed) → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    # Trigger a failed build (non-interactive, no cached tokens)
    with pytest.raises(Exception):
        mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    # Entry exists but provider is None
    assert mgr._entries["srv"].provider is None
    result = await mgr.invalidate_if_disk_changed("srv")
    assert result is False


@pytest.mark.asyncio
async def test_invalidate_returns_false_on_file_not_found(tmp_path, monkeypatch):
    """Tokens file deleted from disk → returns False (no crash)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # Delete the tokens file
    (token_dir / "srv.json").unlink()

    result = await mgr.invalidate_if_disk_changed("srv")
    assert result is False


# ---------------------------------------------------------------------------
# handle_401 — thundering-herd deduplication
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_401_returns_false_for_unknown_server(tmp_path, monkeypatch):
    """No entry for the server → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    result = await mgr.handle_401("nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_returns_false_when_provider_is_none(tmp_path, monkeypatch):
    """Entry exists but provider is None → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch, is_tty=False)
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    with pytest.raises(Exception):
        mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    result = await mgr.handle_401("srv")
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_disk_changed_returns_true(tmp_path, monkeypatch):
    """When disk changed (external refresh), handle_401 returns True."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    tokens_file.write_text(
        json.dumps({
            "access_token": "OLD",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # First invalidate records the mtime
    await mgr.invalidate_if_disk_changed("srv")

    # Touch the file to simulate external refresh
    future_mtime = time.time() + 10
    os.utime(tokens_file, (future_mtime, future_mtime))

    result = await mgr.handle_401("srv", failed_access_token="OLD")
    assert result is True


@pytest.mark.asyncio
async def test_handle_401_no_disk_change_checks_can_refresh(tmp_path, monkeypatch):
    """No disk change → falls back to provider.context.can_refresh_token()."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Record mtime so disk won't appear changed
    await mgr.invalidate_if_disk_changed("srv")

    # Mock can_refresh_token on the provider's context
    ctx = MagicMock()
    ctx.can_refresh_token = MagicMock(return_value=True)
    provider.context = ctx

    result = await mgr.handle_401("srv", failed_access_token="TOK")
    assert result is True
    ctx.can_refresh_token.assert_called_once()


@pytest.mark.asyncio
async def test_handle_401_can_refresh_false_returns_false(tmp_path, monkeypatch):
    """No disk change + can_refresh_token()=False → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    await mgr.invalidate_if_disk_changed("srv")

    ctx = MagicMock()
    ctx.can_refresh_token = MagicMock(return_value=False)
    provider.context = ctx

    result = await mgr.handle_401("srv", failed_access_token="TOK")
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_no_context_returns_false(tmp_path, monkeypatch):
    """No context on provider → can_refresh defaults to False → returns False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    await mgr.invalidate_if_disk_changed("srv")

    # Remove context attribute
    provider.context = None

    result = await mgr.handle_401("srv", failed_access_token="TOK")
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_can_refresh_raises_returns_false(tmp_path, monkeypatch):
    """can_refresh_token() raises → caught, can_refresh stays False."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    await mgr.invalidate_if_disk_changed("srv")

    ctx = MagicMock()
    ctx.can_refresh_token = MagicMock(side_effect=RuntimeError("boom"))
    provider.context = ctx

    result = await mgr.handle_401("srv", failed_access_token="TOK")
    assert result is False


@pytest.mark.asyncio
async def test_handle_401_deduplicates_concurrent_calls(tmp_path, monkeypatch):
    """Two concurrent handle_401 with same token → only one recovery attempt."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    await mgr.invalidate_if_disk_changed("srv")

    ctx = MagicMock()
    ctx.can_refresh_token = MagicMock(return_value=True)
    provider.context = ctx

    # Launch two concurrent handle_401 calls with the same failed token
    import asyncio

    results = await asyncio.gather(
        mgr.handle_401("srv", failed_access_token="TOK"),
        mgr.handle_401("srv", failed_access_token="TOK"),
    )
    # Both should return True (same future)
    assert all(r is True for r in results)
    # can_refresh_token should only be called once (dedup'd)
    assert ctx.can_refresh_token.call_count == 1


@pytest.mark.asyncio
async def test_handle_401_default_key_when_no_token(tmp_path, monkeypatch):
    """failed_access_token=None → uses '<unknown>' as dedup key."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None
    await mgr.invalidate_if_disk_changed("srv")

    ctx = MagicMock()
    ctx.can_refresh_token = MagicMock(return_value=False)
    provider.context = ctx

    result = await mgr.handle_401("srv")  # no failed_access_token
    assert result is False


# ---------------------------------------------------------------------------
# _build_provider — SDK unavailable paths
# ---------------------------------------------------------------------------


def test_build_provider_returns_none_when_cls_is_none(tmp_path, monkeypatch):
    """When _HERMES_PROVIDER_CLS is None (SDK missing), returns None."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager
    import tools.mcp_oauth_manager as mod

    mgr = MCPOAuthManager()
    entry = mod._ProviderEntry(server_url="https://example.com/mcp", oauth_config=None)

    with patch.object(mod, "_HERMES_PROVIDER_CLS", None):
        result = mgr._build_provider("srv", entry)
    assert result is None


def test_build_provider_returns_none_when_oauth_not_available(tmp_path, monkeypatch):
    """When _OAUTH_AVAILABLE is False, returns None."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager
    import tools.mcp_oauth_manager as mod
    import tools.mcp_oauth as oauth_mod

    mgr = MCPOAuthManager()
    entry = mod._ProviderEntry(server_url="https://example.com/mcp", oauth_config=None)

    with patch.object(oauth_mod, "_OAUTH_AVAILABLE", False):
        result = mgr._build_provider("srv", entry)
    assert result is None


# ---------------------------------------------------------------------------
# remove — edge cases
# ---------------------------------------------------------------------------


def test_remove_unknown_server_is_noop(tmp_path, monkeypatch):
    """remove() on a server that was never registered → no crash."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager

    mgr = MCPOAuthManager()
    mgr.remove("never-registered")  # must not raise


# ---------------------------------------------------------------------------
# _persist_oauth_metadata_if_changed
# ---------------------------------------------------------------------------


def test_persist_metadata_no_metadata_is_noop(tmp_path, monkeypatch):
    """When oauth_metadata is None, _persist is a no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Set context.oauth_metadata to None
    ctx = MagicMock()
    ctx.oauth_metadata = None
    provider.context = ctx

    # Must not raise
    provider._persist_oauth_metadata_if_changed()


def test_persist_metadata_non_hermes_storage_is_noop(tmp_path, monkeypatch):
    """When storage is not HermesTokenStorage, _persist is a no-op."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Set context with metadata but non-HermesTokenStorage
    ctx = MagicMock()
    ctx.oauth_metadata = MagicMock(token_endpoint="https://as.example.com/token")
    ctx.storage = MagicMock()  # not HermesTokenStorage
    provider.context = ctx

    # Must not raise — isinstance check fails, so no save attempted
    provider._persist_oauth_metadata_if_changed()


def test_persist_metadata_saves_when_token_endpoint_differs(tmp_path, monkeypatch):
    """When token_endpoint differs from existing, metadata is saved."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager
    from tools.mcp_oauth import HermesTokenStorage

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    storage = HermesTokenStorage("srv")
    meta = MagicMock(token_endpoint="https://as.example.com/token")
    ctx = MagicMock()
    ctx.oauth_metadata = meta
    ctx.storage = storage
    provider.context = ctx

    # load_oauth_metadata returns None (no existing) → should save
    with patch.object(storage, "load_oauth_metadata", return_value=None):
        with patch.object(storage, "save_oauth_metadata") as save_mock:
            provider._persist_oauth_metadata_if_changed()
            save_mock.assert_called_once_with(meta)


def test_persist_metadata_skips_when_token_endpoint_same(tmp_path, monkeypatch):
    """When token_endpoint matches existing, no save."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager
    from tools.mcp_oauth import HermesTokenStorage

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    storage = HermesTokenStorage("srv")
    meta = MagicMock(token_endpoint="https://as.example.com/token")
    existing = MagicMock(token_endpoint="https://as.example.com/token")
    ctx = MagicMock()
    ctx.oauth_metadata = meta
    ctx.storage = storage
    provider.context = ctx

    with patch.object(storage, "load_oauth_metadata", return_value=existing):
        with patch.object(storage, "save_oauth_metadata") as save_mock:
            provider._persist_oauth_metadata_if_changed()
            save_mock.assert_not_called()


# ---------------------------------------------------------------------------
# async_auth_flow — bidirectional generator bridge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_auth_flow_bridges_bidirectional_generator(tmp_path, monkeypatch):
    """async_auth_flow forwards .asend() values into the inner generator."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import (
        MCPOAuthManager,
        get_manager,
        reset_manager_for_tests,
    )

    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = get_manager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Mock super().async_auth_flow to return a simple bidirectional generator
    # that yields a request and expects a response via .asend()
    sent_responses = []

    async def fake_super_flow(self, request):
        response = yield "REQUEST_1"
        sent_responses.append(response)
        yield "REQUEST_2"
        # StopAsyncIteration ends the flow

    # Patch OAuthClientProvider.async_auth_flow (the base class method)
    from mcp.client.auth.oauth2 import OAuthClientProvider

    with patch.object(OAuthClientProvider, "async_auth_flow", fake_super_flow):
        # Set context to None so _persist_oauth_metadata_if_changed is a no-op
        provider.context = MagicMock()
        provider.context.oauth_metadata = None

        gen = provider.async_auth_flow("fake-request")

        # First yield: get the outgoing request
        outgoing1 = await gen.__anext__()
        assert outgoing1 == "REQUEST_1"

        # Send a response back via .asend()
        outgoing2 = await gen.asend("RESPONSE_1")
        assert outgoing2 == "REQUEST_2"

        # End the flow (StopAsyncIteration)
        try:
            await gen.__anext__()
            assert False, "should have raised StopAsyncIteration"
        except StopAsyncIteration:
            pass

    assert sent_responses == ["RESPONSE_1"]
    reset_manager_for_tests()


@pytest.mark.asyncio
async def test_async_auth_flow_persists_metadata_on_completion(tmp_path, monkeypatch):
    """async_auth_flow calls _persist_oauth_metadata_if_changed on StopAsyncIteration."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import (
        MCPOAuthManager,
        get_manager,
        reset_manager_for_tests,
    )

    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(
        json.dumps({
            "access_token": "TOK",
            "token_type": "Bearer",
        })
    )

    mgr = get_manager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Mock super().async_auth_flow to immediately stop (empty generator)
    async def empty_flow(self, request):
        return
        yield  # make it an async generator

    from mcp.client.auth.oauth2 import OAuthClientProvider

    # Set up context with metadata so _persist is called
    from tools.mcp_oauth import HermesTokenStorage

    storage = HermesTokenStorage("srv")
    meta = MagicMock(token_endpoint="https://as.example.com/token")
    ctx = MagicMock()
    ctx.oauth_metadata = meta
    ctx.storage = storage
    provider.context = ctx

    persist_called = []
    with patch.object(OAuthClientProvider, "async_auth_flow", empty_flow):
        with patch.object(
            provider,
            "_persist_oauth_metadata_if_changed",
            lambda: persist_called.append(True),
        ):
            gen = provider.async_auth_flow("fake-request")
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass

    assert persist_called == [True]
    reset_manager_for_tests()
