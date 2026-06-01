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
    from tools.mcp_oauth_manager import MCPOAuthManager, reset_manager_for_tests
    reset_manager_for_tests()

    mgr = MCPOAuthManager()
    p1 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    # Simulate a reconnect: _run_http calls get_or_build_provider again
    p2 = mgr.get_or_build_provider("srv", "https://example.com/mcp", None)

    assert p1 is p2, "manager must cache the provider across reconnects"
