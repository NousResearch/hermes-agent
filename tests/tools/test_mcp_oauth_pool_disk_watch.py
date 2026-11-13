"""Shared-pool disk watching preserves content-version detection from main."""
import os
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_pinned_pool_rewrite_with_preserved_mtime_reloads(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    profile = tmp_path / "profile"
    monkeypatch.setenv("HERMES_HOME", str(profile))
    pool = tmp_path / "owner" / "mcp-tokens" / "team.json"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"access_token":"old"}', encoding="utf-8")
    provider = SimpleNamespace(context=SimpleNamespace(current_tokens=object()), _initialized=True)
    manager = MCPOAuthManager()
    manager._entries[manager._key("team")] = _ProviderEntry(
        server_url="https://example.com/mcp", oauth_config=None,
        pool_path=str(pool), provider=provider,
    )
    # First observation only seeds the baseline for a live provider.
    assert await manager.invalidate_if_disk_changed("team") is False
    assert provider._initialized is True
    before = pool.stat()
    pool.write_text('{"access_token":"new"}', encoding="utf-8")
    os.utime(pool, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert await manager.invalidate_if_disk_changed("team") is True
    assert provider._initialized is False
    provider._initialized = True
    assert await manager.invalidate_if_disk_changed("team") is False
    assert provider._initialized is True
    assert not (profile / "mcp-tokens").exists()


@pytest.mark.asyncio
async def test_temporarily_absent_shared_pool_keeps_live_grant(tmp_path, monkeypatch):
    from tools.mcp_oauth_manager import MCPOAuthManager, _ProviderEntry

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    pool = tmp_path / "owner" / "mcp-tokens" / "team.json"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"access_token":"old"}', encoding="utf-8")
    grant = object()
    provider = SimpleNamespace(context=SimpleNamespace(current_tokens=grant), _initialized=True)
    manager = MCPOAuthManager()
    manager._entries[manager._key("team")] = _ProviderEntry(
        server_url="https://example.com/mcp", oauth_config=None,
        pool_path=str(pool), provider=provider,
    )
    assert await manager.invalidate_if_disk_changed("team") is False
    pool.unlink()
    assert await manager.invalidate_if_disk_changed("team") is False
    assert provider._initialized is True
    assert provider.context.current_tokens is grant
    pool.write_text('{"access_token":"new"}', encoding="utf-8")
    assert await manager.invalidate_if_disk_changed("team") is True
    assert provider._initialized is False
