"""Root-exported MCP OAuth pools (``oauth.share_with_profiles``): resolver, storage pinning and the
manager's pool-aware cache/eviction.

Profiles hold their own ``mcp-tokens/`` by default. A root home may export ONE server's grant to
named profiles whose entry for the same name is byte-for-byte the same OAuth identity; the tests
here pin the fail-closed edges of that contract and the manager rules built on top of it.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

URL = "https://mcp.example.com/mcp"


def _write(home: Path, servers: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")


def _estate(tmp_path, *, root_entry: dict | None, profile_entry: dict | None, profile="worker"):
    root = tmp_path / ".hermes"
    prof = root / "profiles" / profile
    _write(root, {"team": root_entry} if root_entry is not None else {})
    _write(prof, {"team": profile_entry} if profile_entry is not None else {})
    return root, prof


def _token(access: str) -> MagicMock:
    tok = MagicMock()
    tok.model_dump.return_value = {"access_token": access, "token_type": "Bearer", "expires_in": 3600}
    return tok


def _seed(home: Path, name: str, access: str) -> Path:
    d = home / "mcp-tokens"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps({"access_token": access, "token_type": "Bearer", "expires_in": 3600}), encoding="utf-8")
    return p


EXPORT = {"url": URL, "auth": "oauth", "oauth": {"share_with_profiles": True}}
MATCH = {"url": URL, "auth": "oauth"}


# ---------------------------------------------------------------------------
# Resolver: when does a profile read the root pool?
# ---------------------------------------------------------------------------

class TestSharedPoolResolution:
    def test_matching_profile_reads_and_writes_the_root_pool(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage

        root, prof = _estate(tmp_path, root_entry=EXPORT, profile_entry=MATCH)
        monkeypatch.setenv("HERMES_HOME", str(prof))
        storage = HermesTokenStorage("team")
        asyncio.run(storage.set_tokens(_token("shared")))
        assert (root / "mcp-tokens" / "team.json").exists()
        assert not (prof / "mcp-tokens" / "team.json").exists()
        assert asyncio.run(storage.get_tokens()).access_token == "shared"

    def test_pool_is_pinned_at_construction(self, tmp_path, monkeypatch):
        """A live storage keeps its pool even when the profile config changes underneath it."""
        from tools.mcp_oauth import HermesTokenStorage

        root, prof = _estate(tmp_path, root_entry=EXPORT, profile_entry=MATCH)
        monkeypatch.setenv("HERMES_HOME", str(prof))
        storage = HermesTokenStorage("team")
        (prof / "config.yaml").unlink()
        assert storage._tokens_path() == root / "mcp-tokens" / "team.json"
        assert HermesTokenStorage("team")._tokens_path() == prof / "mcp-tokens" / "team.json"

    @pytest.mark.parametrize("root_entry, profile_entry, why", [
        ({"url": URL, "auth": "oauth"}, MATCH, "no export flag"),
        ({"url": URL, "auth": "oauth", "oauth": {"share_with_profiles": "yes"}}, MATCH, "flag must be boolean true"),
        (EXPORT, None, "profile has no entry of its own — nothing is inherited"),
        (EXPORT, {"url": "https://other.example.com/mcp", "auth": "oauth"}, "endpoint differs"),
        (EXPORT, {"url": URL, "auth": "oauth", "transport": "sse"}, "transport differs"),
        (EXPORT, {"url": URL, "auth": "oauth", "oauth": {"client_id": "mine"}}, "client settings differ"),
        (EXPORT, {"url": URL, "headers": {"Authorization": "Bearer x"}}, "profile is not OAuth"),
        ({**EXPORT, "url": "${MCP_URL}"}, {"url": "${MCP_URL}", "auth": "oauth"}, "templated endpoint (per-scope value)"),
        ({**EXPORT, "transport": "${T}"}, {"url": URL, "auth": "oauth", "transport": "${T}"}, "templated transport"),
        ({**EXPORT, "oauth": {"share_with_profiles": True, "client_id": "${CID}"}},
         {"url": URL, "auth": "oauth", "oauth": {"client_id": "${CID}"}}, "templated client identity"),
        ({**EXPORT, "url": "ftp://mcp.example.com/mcp"}, {"url": "ftp://mcp.example.com/mcp", "auth": "oauth"}, "non-HTTP URL"),
    ])
    def test_profile_keeps_local_pool_when_the_export_contract_is_not_met(
            self, tmp_path, monkeypatch, root_entry, profile_entry, why):
        from tools.mcp_oauth import HermesTokenStorage

        root, prof = _estate(tmp_path, root_entry=root_entry, profile_entry=profile_entry)
        monkeypatch.setenv("HERMES_HOME", str(prof))
        monkeypatch.setenv("MCP_URL", URL)
        assert HermesTokenStorage("team")._tokens_path() == prof / "mcp-tokens" / "team.json", why

    def test_root_and_non_profile_homes_never_consult_a_parent(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage

        root, _ = _estate(tmp_path, root_entry=EXPORT, profile_entry=MATCH)
        monkeypatch.setenv("HERMES_HOME", str(root))
        assert HermesTokenStorage("team")._tokens_path() == root / "mcp-tokens" / "team.json"
        # A dir under a folder called ``profiles`` whose parent carries no Hermes-home markers is not
        # a named profile (``hermes_constants.named_profile_home``), so nothing above it is consulted.
        stray = tmp_path / "elsewhere" / "profiles" / "x"
        _write(stray, {"team": MATCH})
        assert HermesTokenStorage("team", hermes_home=stray)._tokens_path() == stray / "mcp-tokens" / "team.json"

    def test_export_flag_is_not_part_of_the_identity(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import _shared_server_identity

        assert _shared_server_identity(EXPORT) == _shared_server_identity(MATCH)


# ---------------------------------------------------------------------------
# Storage: names, tombstone, snapshot/restore
# ---------------------------------------------------------------------------

class TestStorageFiles:
    def test_unsafe_server_names_get_distinct_files(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        a, b = HermesTokenStorage("a/b"), HermesTokenStorage("a_b")
        assert a._tokens_path() != b._tokens_path()
        assert b._tokens_path().name == "a_b.json", "plain names keep their historical path"
        assert a._tokens_path().name.startswith("~u~"), "encoded names live in a namespace no plain name can spell"
        assert "/" not in a._tokens_path().name

    def test_permanent_remove_tombstones_and_unblock_clears(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        _seed(tmp_path, "srv", "x")
        storage.remove(permanent=True)
        assert not storage.has_cached_tokens()
        assert storage.rebuild_blocked()
        storage.unblock_rebuild()
        assert not storage.rebuild_blocked()

    def test_snapshot_round_trips_cimd_marker_and_client_backup_is_cleared(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import HermesTokenStorage

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        _seed(tmp_path, "srv", "x")
        storage.mark_cimd_rejected()
        (tmp_path / "mcp-tokens" / "srv.client.json.bak").write_text("{}", encoding="utf-8")
        snap = storage.snapshot()
        assert "srv.cimd-off" in snap
        storage.remove()
        assert not storage.cimd_rejected()
        assert not (tmp_path / "mcp-tokens" / "srv.client.json.bak").exists()
        storage.restore(snap)
        assert storage.cimd_rejected()
        assert storage.has_cached_tokens()
        assert (tmp_path / "mcp-tokens" / "srv.json").stat().st_mode & 0o777 == 0o600

    def test_rollback_skips_only_when_a_newer_token_landed(self, tmp_path, monkeypatch):
        """Client/metadata left behind by a FAILED attempt must not block restoring the old grant."""
        from tools.mcp_oauth import HermesTokenStorage

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        storage = HermesTokenStorage("srv")
        _seed(tmp_path, "srv", "OLD")
        snap = storage.snapshot()
        storage.remove()
        (tmp_path / "mcp-tokens" / "srv.client.json").write_text('{"client_id":"half-registered"}', encoding="utf-8")
        storage.restore(snap, only_if_absent=True)
        assert json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text())["access_token"] == "OLD"
        # ...but a concurrent SUCCESS is never overwritten.
        _seed(tmp_path, "srv", "FRESH")
        storage.restore(snap, only_if_absent=True)
        assert json.loads((tmp_path / "mcp-tokens" / "srv.json").read_text())["access_token"] == "FRESH"


# ---------------------------------------------------------------------------
# Manager: pool-aware cache, eviction, disk watch
# ---------------------------------------------------------------------------

def _interactive(monkeypatch):
    stdin = MagicMock()
    stdin.isatty.return_value = True
    monkeypatch.setattr("tools.mcp_oauth.sys.stdin", stdin)


class TestManagerPools:
    def _shared(self, tmp_path, monkeypatch):
        from hermes_constants import set_hermes_home_override
        from tools.mcp_oauth_manager import MCPOAuthManager

        root = tmp_path / ".hermes"
        a, b = root / "profiles" / "a", root / "profiles" / "b"
        _write(root, {"team": EXPORT}); _write(a, {"team": MATCH}); _write(b, {"team": MATCH})
        _seed(root, "team", "ROOT")
        _interactive(monkeypatch)
        manager = MCPOAuthManager()
        providers = {}
        for home in (root, a, b):
            tok = set_hermes_home_override(home)
            try:
                providers[home] = manager.get_or_build_provider("team", URL, None)
            finally:
                from hermes_constants import reset_hermes_home_override
                reset_hermes_home_override(tok)
        return manager, root, a, b, providers

    def test_participants_are_distinct_providers_on_one_pool(self, tmp_path, monkeypatch):
        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        assert len({id(p) for p in providers.values()}) == 3, "provider identity stays per (home, name)"
        pools = {manager._entries[manager._key("team", h)].pool_path for h in (root, a, b)}
        assert pools == {str((root / "mcp-tokens" / "team.json").resolve())}
        for p in providers.values():
            assert asyncio.run(p.context.storage.get_tokens()).access_token == "ROOT"

    def test_participant_reauth_clears_the_pool_but_keeps_sibling_providers(self, tmp_path, monkeypatch):
        """``hermes mcp login`` from B: the shared grant is cleared for the fresh flow, but root's and
        A's providers stay cached — they adopt the new grant through the disk watch once it lands."""
        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        manager.remove("team", hermes_home=b)
        assert manager._key("team", b) not in manager._entries
        assert manager._key("team", root) in manager._entries
        assert manager._key("team", a) in manager._entries
        assert not (root / "mcp-tokens" / "team.json").exists()
        assert not getattr(providers[root], "_hermes_detached", False)

    def test_owner_revocation_fences_every_participant(self, tmp_path, monkeypatch):
        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        manager.remove("team", hermes_home=root, block_rebuild=True)
        assert not any(k[1] == "team" for k in manager._entries)
        assert not (root / "mcp-tokens" / "team.json").exists()
        for p in providers.values():
            assert p.context.current_tokens is None, "fenced providers must not keep serving in-memory tokens"
            assert p._hermes_detached is True

    @pytest.mark.asyncio
    async def test_a_fenced_provider_refuses_to_authenticate_requests(self, tmp_path, monkeypatch):
        from tools.mcp_oauth import OAuthStorageDetachedError

        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        manager.detach("team", hermes_home=a)
        assert (root / "mcp-tokens" / "team.json").exists(), "detach never touches the pool"
        flow = providers[a].async_auth_flow(MagicMock())
        with pytest.raises(OAuthStorageDetachedError):
            await flow.__anext__()
        assert not getattr(providers[b], "_hermes_detached", False)

    def test_remove_targets_the_pinned_pool_after_profile_config_changes(self, tmp_path, monkeypatch):
        """A's config drops the shared entry AFTER its provider was built on the root pool; removing
        A's OAuth state must hit the pool that provider actually used, not re-resolve a local one."""
        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        _write(a, {"team": {"url": "https://elsewhere.example/mcp", "auth": "oauth"}})
        _seed(a, "team", "LOCAL_A")
        manager.remove("team", hermes_home=a)
        assert not (root / "mcp-tokens" / "team.json").exists()
        assert (a / "mcp-tokens" / "team.json").exists(), "the local pool was never this provider's"
        assert manager._key("team", root) in manager._entries, "a re-auth never fences the other participants"

    def test_config_change_that_leaves_the_pool_keeps_the_root_grant(self, tmp_path, monkeypatch):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        manager, root, a, b, providers = self._shared(tmp_path, monkeypatch)
        _write(a, {"team": {"url": "https://elsewhere.example/mcp", "auth": "oauth"}})
        tok = set_hermes_home_override(a)
        try:
            rebuilt = manager.get_or_build_provider("team", "https://elsewhere.example/mcp", None)
        finally:
            reset_hermes_home_override(tok)
        assert rebuilt is not providers[a]
        assert (root / "mcp-tokens" / "team.json").exists(), "A leaving the pool is not a revocation"
        assert manager._entries[manager._key("team", a)].pool_path == str((a / "mcp-tokens" / "team.json").resolve())
        assert providers[a].context.current_tokens is None
        assert providers[a]._hermes_detached is True, "the old provider must not present the root grant"
        assert manager._key("team", root) in manager._entries

    def test_config_change_on_the_same_pool_wipes_it(self, tmp_path, monkeypatch):
        from tools.mcp_oauth_manager import MCPOAuthManager

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _interactive(monkeypatch)
        grant = _seed(tmp_path, "srv", "OLD")
        manager = MCPOAuthManager()
        first = manager.get_or_build_provider("srv", URL, None)
        second = manager.get_or_build_provider("srv", "https://new.example/mcp", None)
        assert second is not first
        assert not grant.exists(), "a new identity must not reload tokens minted for the old one"
        third = manager.get_or_build_provider("srv", "https://new.example/mcp", {"client_id": "c2"})
        assert third is not second, "an oauth-block change rebuilds too"

    def test_tombstone_blocks_rebuild_until_unblocked(self, tmp_path, monkeypatch):
        from tools.mcp_oauth_manager import MCPOAuthManager

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _interactive(monkeypatch)
        _seed(tmp_path, "srv", "x")
        manager = MCPOAuthManager()
        assert manager.get_or_build_provider("srv", URL, None) is not None
        manager.remove("srv", block_rebuild=True)
        assert manager.get_or_build_provider("srv", URL, None) is None
        manager.unblock("srv")
        assert manager.get_or_build_provider("srv", URL, None) is not None

    @pytest.mark.asyncio
    async def test_disk_watch_follows_the_shared_pool(self, tmp_path, monkeypatch):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.mcp_oauth_manager import MCPOAuthManager

        root, prof = _estate(tmp_path, root_entry=EXPORT, profile_entry=MATCH)
        grant = _seed(root, "team", "ROOT")
        _interactive(monkeypatch)
        manager = MCPOAuthManager()
        tok = set_hermes_home_override(prof)
        try:
            provider = manager.get_or_build_provider("team", URL, None)
        finally:
            reset_hermes_home_override(tok)
        assert await manager.invalidate_if_disk_changed("team", hermes_home=prof) is True  # first sight
        assert await manager.invalidate_if_disk_changed("team", hermes_home=prof) is False
        later = time.time() + 10
        os.utime(grant, (later, later))  # a sibling refreshed the shared pool
        assert await manager.invalidate_if_disk_changed("team", hermes_home=prof) is True
        assert provider._initialized is False
        grant.unlink()  # a sibling's re-auth is in flight: not a change, keep the in-memory grant
        assert await manager.invalidate_if_disk_changed("team", hermes_home=prof) is False


def test_config_cache_sees_a_same_size_rewrite_with_a_pinned_mtime(tmp_path):
    """The resolver's cache is keyed on ctime too: ``cp -p``/``os.utime`` rewrites that keep mtime,
    size and inode must still be re-read, or a withdrawn export would stay cached as shared."""
    import os

    from tools.mcp_oauth import _load_mcp_server_config

    home = tmp_path / "h"
    home.mkdir()
    cfg = home / "config.yaml"
    cfg.write_text("mcp_servers:\n  team: {url: 'https://a.example/mcp', auth: oauth}\n")
    st = cfg.stat()
    assert _load_mcp_server_config(home, "team")["url"] == "https://a.example/mcp"
    time.sleep(0.05)  # past the kernel's coarse timestamp tick, as any real edit is
    with open(cfg, "r+") as fh:  # in place: same inode, same size
        fh.write("mcp_servers:\n  team: {url: 'https://b.example/mcp', auth: oauth}\n")
    os.utime(cfg, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert cfg.stat().st_size == st.st_size and cfg.stat().st_ino == st.st_ino
    assert _load_mcp_server_config(home, "team")["url"] == "https://b.example/mcp"


def test_abandoning_the_auth_flow_mid_refresh_releases_the_fence_now(tmp_path, monkeypatch):
    """httpx cancels the flow while the refresh POST is in flight: the fence must be free
    immediately, or an owner's revocation (which takes it) would stall until GC."""
    import asyncio

    import httpx

    from tools.mcp_oauth import hold_refresh_fence
    from tools.mcp_oauth_manager import MCPOAuthManager

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    pool = tmp_path / "mcp-tokens" / "team.json"
    pool.parent.mkdir(parents=True)
    pool.write_text(json.dumps({"access_token": "OLD", "token_type": "Bearer", "expires_in": 1,
                                "refresh_token": "R", "expires_at": 1, "issuer": "https://auth.example.com"}))
    (pool.parent / "team.client.json").write_text(json.dumps({
        "client_id": "cid", "redirect_uris": ["http://127.0.0.1:1/callback"], "token_endpoint_auth_method": "none"}))
    (pool.parent / "team.meta.json").write_text(json.dumps({
        "issuer": "https://auth.example.com", "authorization_endpoint": "https://auth.example.com/a",
        "token_endpoint": "https://auth.example.com/token", "response_types_supported": ["code"]}))
    provider = MCPOAuthManager().get_or_build_provider("team", "https://mcp.example.com/mcp", None)

    async def abandon_mid_refresh():
        flow = provider.async_auth_flow(httpx.Request("GET", "https://mcp.example.com/mcp"))
        out = await flow.__anext__()
        assert str(out.url) == "https://auth.example.com/token", "precondition: the refresh POST"
        await flow.aclose()
        # Still inside the loop: no async-generator finalizer has had a chance to run.
        with hold_refresh_fence(pool, timeout=0.5):
            pass

    asyncio.run(abandon_mid_refresh())
