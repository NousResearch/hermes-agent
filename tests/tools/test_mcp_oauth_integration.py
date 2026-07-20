"""End-to-end integration tests for the MCP OAuth consolidation.

Exercises the full chain — manager, provider subclass, disk watch, 401
dedup — with real file I/O and real imports (no transport mocks, no
subprocesses). These are the tests that would catch Cthulhu's original
BetterStack bug: an external process rewrites the tokens file on disk,
and the running Hermes session picks up the new tokens on the next auth
flow without requiring a restart.
"""
import asyncio
import json
import os
import time

import pytest


pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required")


def _set_interactive_stdin(monkeypatch, *, is_tty: bool = True) -> None:
    from unittest.mock import MagicMock

    mock_stdin = MagicMock()
    mock_stdin.isatty.return_value = is_tty
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", mock_stdin)


@pytest.mark.asyncio
async def test_external_refresh_picked_up_without_restart(tmp_path, monkeypatch):
    """Simulate Cthulhu's cron workflow end-to-end.

    1. A running Hermes session has OAuth tokens loaded in memory.
    2. An external process (cron) writes fresh tokens to disk.
    3. On the next auth flow, the manager's disk-watch invalidates the
       in-memory state so the SDK re-reads from storage.
    4. ``provider.context.current_tokens`` now reflects the new tokens
       with no process restart required.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    client_info_file = token_dir / "srv.client.json"

    # Pre-seed the baseline state: valid tokens the session loaded at startup.
    tokens_file.write_text(json.dumps({
        "access_token": "OLD_ACCESS",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "OLD_REFRESH",
    }))
    client_info_file.write_text(json.dumps({
        "client_id": "test-client",
        "redirect_uris": ["http://127.0.0.1:12345/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider(
        "srv", "https://example.com/mcp", None,
    )
    assert provider is not None

    # The SDK's _initialize reads tokens from storage into memory. This
    # is what happens on the first http request under normal operation.
    await provider._initialize()
    assert provider.context.current_tokens.access_token == "OLD_ACCESS"

    # Now record the baseline mtime in the manager (this happens
    # automatically via the HermesMCPOAuthProvider.async_auth_flow
    # pre-hook on the first real request, but we exercise it directly
    # here for test determinism).
    await mgr.invalidate_if_disk_changed("srv")

    # EXTERNAL PROCESS: cron rewrites the tokens file with fresh creds.
    # The old refresh_token has been consumed by this external exchange.
    future_mtime = time.time() + 1
    tokens_file.write_text(json.dumps({
        "access_token": "NEW_ACCESS",
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": "NEW_REFRESH",
    }))
    os.utime(tokens_file, (future_mtime, future_mtime))

    # The next auth flow should detect the mtime change and reload.
    changed = await mgr.invalidate_if_disk_changed("srv")
    assert changed, "manager must detect the disk mtime change"
    assert provider._initialized is False, "_initialized must flip so SDK re-reads storage"

    # Simulate the next async_auth_flow: _initialize runs because _initialized=False.
    await provider._initialize()
    assert provider.context.current_tokens.access_token == "NEW_ACCESS"
    assert provider.context.current_tokens.refresh_token == "NEW_REFRESH"


@pytest.mark.asyncio
async def test_handle_401_deduplicates_concurrent_callers(tmp_path, monkeypatch):
    """Ten concurrent 401 handlers for the same token should fire one recovery.

    Mirrors Claude Code's pending401Handlers dedup pattern — prevents N MCP
    tool calls hitting 401 simultaneously from all independently clearing
    caches and re-reading the keychain (which thrashes the storage and
    bogs down startup per CC-1096).
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({
        "access_token": "TOK",
        "token_type": "Bearer",
        "expires_in": 3600,
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider(
        "srv", "https://example.com/mcp", None,
    )
    assert provider is not None

    # Count how many times invalidate_if_disk_changed is called — proxy for
    # how many actual recovery attempts fire.
    call_count = 0
    real_invalidate = mgr.invalidate_if_disk_changed

    async def counting(name):
        nonlocal call_count
        call_count += 1
        return await real_invalidate(name)

    monkeypatch.setattr(mgr, "invalidate_if_disk_changed", counting)

    # Fire 10 concurrent handlers with the same failed token.
    results = await asyncio.gather(*(
        mgr.handle_401("srv", "SAME_FAILED_TOKEN") for _ in range(10)
    ))

    # All callers get the same result (the shared future's resolution).
    assert all(r == results[0] for r in results), "dedup must return identical result"
    # Exactly ONE recovery ran — the rest awaited the same pending future.
    assert call_count == 1, f"expected 1 recovery attempt, got {call_count}"


@pytest.mark.asyncio
async def test_handle_401_returns_false_when_no_provider(tmp_path, monkeypatch):
    """handle_401 for an unknown server returns False cleanly."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    result = await mgr.handle_401("nonexistent", "any_token")
    assert result is False


@pytest.mark.asyncio
async def test_invalidate_if_disk_changed_handles_missing_file(tmp_path, monkeypatch):
    """invalidate_if_disk_changed returns False when tokens file doesn't exist."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # No tokens file exists yet — this is the pre-auth state
    result = await mgr.invalidate_if_disk_changed("srv")
    assert result is False


@pytest.mark.asyncio
async def test_provider_is_reused_across_reconnects(tmp_path, monkeypatch):
    """The manager caches providers; multiple reconnects reuse the same instance.

    This is what makes the disk-watch stick across reconnects: tearing down
    the MCP session and rebuilding it (Task 5's _reconnect_event path) must
    not create a new provider, otherwise ``last_mtime_ns`` resets and the
    first post-reconnect auth flow would spuriously "detect" a change.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _set_interactive_stdin(monkeypatch)
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # Simulate a reconnect: _run_http calls get_or_build_provider again
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert p1 is p2, "manager must cache the provider across reconnects"


@pytest.mark.asyncio
async def test_idle_expired_token_self_heals_on_reconnect_without_disk_write(
    tmp_path, monkeypatch
):
    """An OAuth token that expires while the session sits idle must refresh on
    reconnect — with NO external disk write and NO tokens-file mtime change.

    Repro of the linear-MCP outage: a live streamable-HTTP OAuth session sits
    idle, its ~24h access token expires in memory, the connection drops, and
    the run loop's reconnect/parked-revival path rebuilds the transport reusing
    the SAME cached provider. Disk-watch (invalidate_if_disk_changed) can NOT
    save it — the tokens file was never rewritten, so its mtime is unchanged.
    Unless the reconnect path resets ``_initialized``, ``_initialize`` never
    re-runs, ``token_expiry_time`` is never re-seeded, ``is_token_valid()``
    keeps reporting the dead token valid, and the SDK's refresh branch never
    fires — the server parks forever on 401.

    ``reset_initialized`` is the reconnect-path hook that fixes this: it flips
    ``_initialized`` so the next ``_initialize`` re-reads the (unchanged-on-disk
    but now-expired) token, re-seeds expiry, and ``can_refresh_token()`` reports
    True so the SDK refreshes on the reconnect handshake.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    tokens_file = token_dir / "srv.json"
    (token_dir / "srv.client.json").write_text(json.dumps({
        "client_id": "test-client",
        "redirect_uris": ["http://127.0.0.1:12345/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }))

    # Token that is ALREADY EXPIRED on disk (absolute expires_at in the past),
    # but with a live refresh_token — exactly the idle-drop state.
    tokens_file.write_text(json.dumps({
        "access_token": "EXPIRED_ACCESS",
        "token_type": "Bearer",
        "expires_at": time.time() - 60,
        "refresh_token": "LIVE_REFRESH",
    }))

    mgr = MCPOAuthManager()
    provider = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)
    assert provider is not None

    # Record the baseline mtime first (first call flips zero -> real mtime and
    # resets _initialized as a side effect), then run _initialize so the
    # provider ends up in the steady "connected" state: _initialized True with
    # the (already-expired) token loaded — exactly the idle-past-expiry state.
    await mgr.invalidate_if_disk_changed("srv")
    await provider._initialize()
    assert provider._initialized is True

    # Disk-watch alone cannot help — the tokens file was never rewritten, so
    # its mtime is unchanged.
    assert await mgr.invalidate_if_disk_changed("srv") is False
    assert provider._initialized is True, "disk-watch must NOT have reset it"

    # The reconnect path resets _initialized so the SDK re-reads + refreshes.
    assert mgr.reset_initialized("srv") is True
    assert provider._initialized is False

    # Next auth flow re-runs _initialize, re-seeds expiry from the expired
    # on-disk token, and the SDK now knows it must refresh.
    await provider._initialize()
    assert provider.context.can_refresh_token() is True, (
        "after reconnect reset, the SDK must recognise the token as refreshable"
    )


@pytest.mark.asyncio
async def test_run_http_resets_provider_initialized_on_oauth_path(
    tmp_path, monkeypatch
):
    """_run_http must call reset_initialized after fetching the OAuth provider.

    This is the wiring that makes an idle-expired token self-heal on reconnect:
    every _run_http entry (each reconnect re-enters it) resets the cached
    provider's _initialized flag so the SDK re-reads storage, re-seeds expiry,
    and refreshes on the reconnect handshake. Without this call the cached
    provider keeps _initialized=True and the dead token is reused forever.

    We drive _run_http far enough to fetch the provider, then let the transport
    step fail (no real server) — reset_initialized must already have fired.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Pre-seed cached tokens so provider construction does not fail-fast in the
    # non-interactive test env (mcp_oauth_manager guard).
    token_dir = tmp_path / "mcp-tokens"
    token_dir.mkdir(parents=True)
    (token_dir / "srv.json").write_text(json.dumps({
        "access_token": "TOK",
        "token_type": "Bearer",
        "expires_at": time.time() - 60,
        "refresh_token": "R",
    }))

    from tools import mcp_tool
    from tools.mcp_oauth_manager import get_manager, reset_manager_for_tests
    reset_manager_for_tests()

    reset_calls = []
    mgr = get_manager()
    real_reset = mgr.reset_initialized

    def spy_reset(name, **kwargs):
        reset_calls.append(name)
        return real_reset(name, **kwargs)

    monkeypatch.setattr(mgr, "reset_initialized", spy_reset)

    task = mcp_tool.MCPServerTask("srv")
    task._auth_type = "oauth"

    config = {
        "url": "https://example.com/mcp",
        "oauth": None,
        # a short connect timeout so the transport step fails fast
        "connect_timeout": 0.1,
    }

    # The transport connection will fail (no server); we only care that the
    # provider was fetched and reset_initialized fired before that.
    with pytest.raises(Exception):
        await task._run_http(config)

    assert "srv" in reset_calls, (
        "_run_http must call reset_initialized('srv') on the OAuth path so a "
        "reconnect forces the SDK to re-read + refresh an idle-expired token"
    )
