"""Owner withdrawal of a root-exported MCP OAuth pool reaches LIVE participants in OTHER processes.

The manager's in-memory detach only fences providers cached by the process that made the change.
Profiles run as separate processes (gateways, cron, CLI), so the authority a participant was
built with must be re-proved from disk before every use: the ``.removed`` tombstone next to the
pool and the two ``config.yaml`` files the pool was resolved from.

Each test runs a real peer interpreter holding profile B's provider, built once and kept live for
the whole test, and changes the root through the real ``hermes_cli.mcp_config`` helpers in this
process. The peer is never rebuilt or restarted: it must fail closed on its next request. A
transient token absence (a sibling re-authorizing) must not fence anyone, and a refresh that is
already in flight must not write a rotated grant back over an owner's revocation.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import hermes_yaml as yaml

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK 1.26.0+ required for OAuth support")

URL = "https://mcp.example.com/mcp"
SERVER = {"url": URL, "auth": "oauth"}
EXPORT = {**SERVER, "oauth": {"share_with_profiles": True}}
REPO_ROOT = Path(__file__).resolve().parents[2]

# The peer: profile B's runtime. It builds ONE provider at start-up and answers commands on stdin
# with one JSON line each, so the parent can interleave its own mutations between B's requests.
PEER = textwrap.dedent('''
    import asyncio, json, sys
    import httpx
    from tools.mcp_oauth_manager import get_manager

    URL = sys.argv[1]
    provider = get_manager().get_or_build_provider("team", URL, None)

    async def probe():
        flow = provider.async_auth_flow(httpx.Request("GET", URL))
        try:
            request = await flow.__anext__()
        except Exception as exc:  # the verdict IS the exception type
            return {"ok": False, "error": type(exc).__name__}
        finally:
            await flow.aclose()
        return {"ok": True, "authorization": request.headers.get("authorization")}

    async def refresh_then_persist():
        # One refresh generation, driven through the provider's own fenced hooks: take the fence
        # and re-prove authority, tell the parent, wait, then persist the rotated grant.
        from mcp.shared.auth import OAuthToken
        try:
            await provider._refresh_token()
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}
        emit({"fenced": True})
        sys.stdin.readline()
        rotated = OAuthToken(access_token="ROTATED_BY_B", token_type="Bearer", expires_in=3600,
                             refresh_token="ROTATED_REFRESH")
        response = httpx.Response(200, json=rotated.model_dump(mode="json", exclude_none=True))
        try:
            await provider._handle_refresh_response(response)
        except Exception as exc:
            return {"ok": False, "error": type(exc).__name__}
        return {"ok": True}

    def emit(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    emit({"ready": provider is not None})
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "probe":
            emit(asyncio.run(probe()))
        elif cmd == "refresh":
            emit(asyncio.run(refresh_then_persist()))
        elif cmd == "try_fence":
            from pathlib import Path
            from tools.mcp_oauth import hold_refresh_fence, RefreshFenceTimeout
            try:
                with hold_refresh_fence(Path(provider.context.storage.pool_path), timeout=0):
                    emit({"acquired": True})
            except RefreshFenceTimeout:
                emit({"acquired": False})
        elif cmd == "quit":
            break
''')


def _write_config(home: Path, servers: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")
    # The resolver caches parses on (mtime_ns, size, inode): make every rewrite visible even on
    # filesystems with coarse timestamps.
    st = (home / "config.yaml").stat()
    os.utime(home / "config.yaml", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


def _seed_pool(home: Path, access: str) -> Path:
    d = home / "mcp-tokens"
    d.mkdir(parents=True, exist_ok=True)
    (d / "team.client.json").write_text(json.dumps({
        "client_id": "cid", "redirect_uris": ["http://127.0.0.1:1/callback"],
        "token_endpoint_auth_method": "none"}), encoding="utf-8")
    # Metadata on disk so the peer never goes to the network for discovery.
    (d / "team.meta.json").write_text(json.dumps({
        "issuer": "https://auth.example.com", "authorization_endpoint": "https://auth.example.com/authorize",
        "token_endpoint": "https://auth.example.com/token", "response_types_supported": ["code"]}),
        encoding="utf-8")
    path = d / "team.json"
    path.write_text(json.dumps({
        "access_token": access, "token_type": "Bearer", "expires_in": 3600, "refresh_token": access + "_R",
        "expires_at": time.time() + 3600, "issuer": "https://auth.example.com"}), encoding="utf-8")
    return path


class Peer:
    """Profile B's live runtime in its own interpreter."""

    def __init__(self, home: Path):
        env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT)}
        self.proc = subprocess.Popen(
            [sys.executable, "-c", PEER, URL], cwd=str(REPO_ROOT), env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert self.read() == {"ready": True}, "peer failed to build its provider"

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError(f"peer exited: {self.proc.stderr.read()[-2000:]}")
        return json.loads(line)

    def send(self, cmd: str) -> dict:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        return self.read()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture
def estate(tmp_path, monkeypatch):
    """Root exports ``team``; profiles A and B carry the identical entry and share the root pool.
    Profile B runs as a live peer process. Yields ``(root, a, b, peer, activate)``; ``activate``
    points this process at a home the way a profile-scoped turn does."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.mcp_oauth_manager import reset_manager_for_tests

    root = tmp_path / ".hermes"
    a, b = root / "profiles" / "a", root / "profiles" / "b"
    _write_config(root, {"team": EXPORT})
    _write_config(a, {"team": dict(SERVER)})
    _write_config(b, {"team": dict(SERVER)})
    _seed_pool(root, "ROOT_GRANT")
    reset_manager_for_tests()
    state = {"token": None}

    def activate(home: Path):
        if state["token"] is not None:
            reset_hermes_home_override(state["token"])
        state["token"] = set_hermes_home_override(home)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home)
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: home / "config.yaml")
        monkeypatch.setattr("hermes_cli.config.get_env_path", lambda: home / ".env")
        monkeypatch.setattr("hermes_cli.mcp_config.get_hermes_home", lambda: home)

    peer = Peer(b)
    try:
        assert peer.send("probe") == {"ok": True, "authorization": "Bearer ROOT_GRANT"}
        yield root, a, b, peer, activate
    finally:
        peer.close()
        if state["token"] is not None:
            reset_hermes_home_override(state["token"])
        reset_manager_for_tests()


def _grant(home: Path) -> Path:
    return home / "mcp-tokens" / "team.json"


def test_export_turned_off_fences_the_live_peer_and_keeps_the_root_grant(estate):
    """(a) root ``share_with_profiles: true -> false``: storage identity is unchanged, so root keeps
    its grant, but the peer loses the right to present it without being restarted."""
    from hermes_cli.mcp_config import _save_mcp_server

    root, _a, _b, peer, activate = estate
    activate(root)
    assert _save_mcp_server("team", {**SERVER, "oauth": {"share_with_profiles": False}}) is True

    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}
    assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT", "export-off must not revoke root"
    assert peer.send("probe")["ok"] is False, "a fenced provider stays fenced"


def test_root_identity_change_fences_the_live_peer(estate):
    """(b) root re-points its entry: the old shared grant is revoked and the peer, still configured
    for the old identity, can neither present it nor resurrect it."""
    from hermes_cli.mcp_config import _save_mcp_server

    root, _a, _b, peer, activate = estate
    activate(root)
    assert _save_mcp_server("team", {"url": "https://mcp.example.com/v2", "auth": "oauth",
                                     "oauth": {"share_with_profiles": True}}) is True

    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}
    assert not _grant(root).exists(), "the peer must not write the old grant back"


def test_root_removal_fences_the_live_peer(estate):
    """(c) root removes the server: the tombstone fences the peer before it can use or re-mint."""
    from hermes_cli.mcp_config import _remove_mcp_server

    root, _a, _b, peer, activate = estate
    activate(root)
    assert _remove_mcp_server("team") is True

    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}
    assert not _grant(root).exists()
    assert (root / "mcp-tokens" / "team.removed").exists()


def test_a_sibling_reauthorizing_does_not_fence_the_live_peer(estate):
    """A transient token absence is not a revocation: A's re-login clears the shared pool while it
    runs, and B keeps serving until the new grant lands, then adopts it."""
    root, a, _b, peer, activate = estate
    from tools.mcp_oauth_manager import get_manager

    activate(a)
    get_manager().remove("team")  # the re-auth path: this home's entry + the pool, no participant fence
    assert not _grant(root).exists()
    assert peer.send("probe")["ok"] is True, "missing tokens during a sibling's re-auth must not fence"

    _seed_pool(root, "NEW_GRANT")
    assert peer.send("probe") == {"ok": True, "authorization": "Bearer NEW_GRANT"}


def test_revocation_waits_for_an_in_flight_refresh_and_is_not_resurrected(estate, monkeypatch):
    """The peer holds the refresh fence with authority proven, then root revokes. Revocation must
    wait for the fence, then delete what the peer persisted: the rotated grant never outlives it."""
    import threading

    from hermes_cli.mcp_config import _remove_mcp_server

    root, _a, _b, peer, activate = estate
    peer.proc.stdin.write("refresh\n")
    peer.proc.stdin.flush()
    assert peer.read() == {"fenced": True}

    activate(root)
    from tools import mcp_oauth
    advanced = threading.Event()
    advance = mcp_oauth.advance_pool_epoch

    def published(pool):
        advance(pool)
        advanced.set()

    monkeypatch.setattr(mcp_oauth, "advance_pool_epoch", published)
    result = {}
    revoker = threading.Thread(target=lambda: result.setdefault("ok", _remove_mcp_server("team")))
    revoker.start()
    assert advanced.wait(timeout=10)
    assert revoker.is_alive(), "revocation must wait for the in-flight refresh generation"

    peer.proc.stdin.write("\n")  # let the peer persist its rotated grant and release the fence
    peer.proc.stdin.flush()
    assert peer.read() == {"ok": True}
    revoker.join(timeout=30)
    assert result.get("ok") is True

    assert not _grant(root).exists(), "the rotated grant must not survive the owner's revocation"
    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}


def test_a_refresh_that_starts_after_withdrawal_fails_closed_under_the_fence(estate):
    """The peer's request passed its entry check just before root withdrew the export; its refresh
    then runs. Authority is re-proved under the refresh fence, so no refresh POST is built and
    nothing is written to the pool the peer no longer has a right to."""
    from hermes_cli.mcp_config import _save_mcp_server

    root, _a, _b, peer, activate = estate
    activate(root)
    assert _save_mcp_server("team", {**SERVER, "oauth": {"share_with_profiles": False}}) is True
    before = _grant(root).read_bytes()

    assert peer.send("refresh") == {"ok": False, "error": "OAuthStorageDetachedError"}
    assert _grant(root).read_bytes() == before, "a withdrawn participant must not rotate the owner's grant"


def test_a_tombstone_alone_fences_the_live_peer(estate):
    """Defense in depth: the owner's permanent-removal tombstone fences a peer even while every
    config.yaml still describes a matching export (e.g. the revocation reached disk before or
    without the config write)."""
    from tools.mcp_oauth_manager import get_manager

    root, _a, _b, peer, activate = estate
    activate(root)
    get_manager().remove("team", block_rebuild=True)

    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}


def test_a_local_pool_identity_change_fences_a_live_provider_in_another_process(estate):
    """Not only shared pools: when a home re-points its OWN entry, a provider another process built
    for the old identity must stop, even though its pool is still 'its own'. (Here root plays the
    owner of a pool nobody shares: export is off, so B is on its own local pool.)"""
    from hermes_cli.mcp_config import _save_mcp_server

    root, _a, b, peer, activate = estate
    activate(root)
    assert _save_mcp_server("team", {**SERVER, "oauth": {"share_with_profiles": False}}) is True
    peer.send("probe")  # B's shared provider fences; the peer is rebuilt below on its own pool
    peer.close()
    _seed_pool(b, "B_LOCAL")
    local = Peer(b)
    try:
        assert local.send("probe") == {"ok": True, "authorization": "Bearer B_LOCAL"}
        activate(b)
        assert _save_mcp_server("team", {"url": "https://other.example.com/mcp", "auth": "oauth"}) is True
        assert local.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}
        assert not _grant(b).exists(), "the old identity's grant must not be written back"
    finally:
        local.close()


def test_a_provider_that_sleeps_through_remove_and_readd_is_fenced(estate):
    """Root removes the export and re-adds it (same identity) while B is idle. The tombstone is gone
    by B's next request, but the epoch advanced: B's provider was built for the previous grant."""
    from hermes_cli.mcp_config import _remove_mcp_server, _save_mcp_server

    root, _a, _b, peer, activate = estate
    activate(root)
    assert _remove_mcp_server("team") is True
    assert _save_mcp_server("team", dict(EXPORT)) is True
    assert not (root / "mcp-tokens" / "team.removed").exists()
    _seed_pool(root, "READDED_GRANT")

    assert peer.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}


def test_an_unreadable_root_config_fails_closed_without_detaching(estate):
    """Unknown is not revoked: while root's config.yaml cannot be read, B's requests fail closed,
    and B serves again as soon as it reads — no reconnect needed."""
    root, _a, _b, peer, _activate = estate
    cfg = root / "config.yaml"
    cfg.chmod(0)
    try:
        if os.access(cfg, os.R_OK):
            pytest.skip("running as a user that bypasses file permissions")
        assert peer.send("probe") == {"ok": False, "error": "OAuthAuthorityUnavailableError"}
    finally:
        cfg.chmod(0o600)
    assert peer.send("probe") == {"ok": True, "authorization": "Bearer ROOT_GRANT"}


def test_a_hand_edited_identity_change_fences_a_live_provider(estate):
    """A hand edit of config.yaml goes through no Hermes code, so nothing advances the epoch, and a
    home's OWN pool is never re-resolved: the provider's own-entry fingerprint is the only thing
    that notices the identity it was built for is gone. The live provider here is the root's."""
    root, _a, _b, _peer, _activate = estate
    owner = Peer(root)
    try:
        assert owner.send("probe") == {"ok": True, "authorization": "Bearer ROOT_GRANT"}
        _write_config(root, {"team": {"url": "https://other.example.com/mcp", "auth": "oauth",
                                      "oauth": {"share_with_profiles": True}}})
        assert owner.send("probe") == {"ok": False, "error": "OAuthStorageDetachedError"}
    finally:
        owner.close()


@pytest.mark.parametrize("entrypoint", ["cli", "dashboard", "connector"])
def test_reauth_snapshot_and_clear_exclude_a_cross_process_refresh(estate, monkeypatch, entrypoint):
    """At the exact snapshot/clear seam another process must be unable to take the refresh fence.
    Its next successful rotation after the clear must also win over the failed attempt's rollback.
    """
    from hermes_cli import mcp_config
    from tools.mcp_oauth import HermesTokenStorage

    root, a, _b, peer, activate = estate
    activate(a)
    snapshot = HermesTokenStorage.snapshot
    observed = []

    def observe_snapshot(storage, **kwargs):
        backup = snapshot(storage, **kwargs)
        observed.append(peer.send("try_fence"))
        return backup

    def rotate_then_fail(*args, **kwargs):
        assert _grant(root).exists() is (entrypoint == "connector")
        assert peer.send("refresh") == {"fenced": True}
        peer.proc.stdin.write("\n")
        peer.proc.stdin.flush()
        assert peer.read() == {"ok": True}
        raise RuntimeError("the new login failed")

    monkeypatch.setattr(HermesTokenStorage, "snapshot", observe_snapshot)
    monkeypatch.setattr(mcp_config, "_probe_single_server", rotate_then_fail)
    if entrypoint == "cli":
        assert mcp_config._reauth_oauth_server("team", SERVER) is False
    elif entrypoint == "dashboard":
        from hermes_cli.web_server_mcp import _run_dashboard_mcp_oauth
        from tools.mcp_dashboard_oauth import DashboardOAuthFlow
        flow = DashboardOAuthFlow("race", "team", "a", str(a), "http://127.0.0.1/callback")
        _run_dashboard_mcp_oauth(flow, SERVER)
    else:
        from tools.connectors.mcp_oauth import probe_with_rollback
        with pytest.raises(RuntimeError, match="new login failed"):
            probe_with_rollback("team", SERVER, str(a), None, False)
    assert observed == ([] if entrypoint == "connector" else [{"acquired": False}])
    assert json.loads(_grant(root).read_text())["refresh_token"] == "ROTATED_REFRESH"


@pytest.mark.parametrize("boundary", ["config", "epoch"])
def test_fresh_process_cannot_load_old_bytes_while_owner_waits_for_refresh(estate, monkeypatch, boundary):
    """The new epoch may already be visible, but the old identity's bearer stays quarantined."""
    import threading
    from hermes_cli.mcp_config import _OAuthPoolTransition, _save_mcp_server
    from tools import mcp_oauth

    root, _a, _b, peer, activate = estate
    assert peer.send("refresh") == {"fenced": True}
    activate(root)
    published = threading.Event()
    resume = threading.Event()
    advance = mcp_oauth.advance_pool_epoch

    def publish(pool):
        advance(pool)
        if boundary == "epoch":
            published.set()

    monkeypatch.setattr(mcp_oauth, "advance_pool_epoch", publish)
    commit = _OAuthPoolTransition.commit

    def after_config(transition):
        if boundary == "config":
            published.set()
            assert resume.wait(timeout=15)
        return commit(transition)

    monkeypatch.setattr(_OAuthPoolTransition, "commit", after_config)
    result = {}
    new_url = "https://different.example.com/mcp"
    revoker = threading.Thread(target=lambda: result.setdefault(
        "ok", _save_mcp_server("team", {**EXPORT, "url": new_url})))
    revoker.start()
    try:
        assert published.wait(timeout=10)
        assert _grant(root).exists(), "the peer still owns the fence, so bytes cannot yet be deleted"
        script = '''
import asyncio, json, sys
from tools.mcp_oauth import HermesTokenStorage, OAuthStorageDetachedError
from tools.mcp_oauth_manager import get_manager
s = HermesTokenStorage("team")
try:
    tokens = asyncio.run(s.get_tokens())
    loaded = tokens is not None
except OAuthStorageDetachedError:
    loaded = False
provider = get_manager().get_or_build_provider("team", sys.argv[1], {"share_with_profiles": True})
print(json.dumps({"loaded": loaded, "cached": s.has_cached_tokens(), "built": provider is not None}))
'''
        probe = subprocess.run([sys.executable, "-c", script, new_url], cwd=REPO_ROOT,
                               env={**os.environ, "HERMES_HOME": str(root)},
                               capture_output=True, text=True, timeout=15, check=True)
        assert json.loads(probe.stdout) == {"loaded": False, "cached": False, "built": False}
    finally:
        resume.set()
        peer.proc.stdin.write("\n")
        peer.proc.stdin.flush()
        peer.read()
        revoker.join(timeout=15)
    assert result == {"ok": True}
    assert not _grant(root).exists()
    assert not (root / "mcp-tokens" / "team.quarantined").exists()


@pytest.mark.parametrize("failure", ["cancel", "initialize", "save"])
def test_connector_attempt_cannot_rollback_a_peer_rotation_after_mint(estate, monkeypatch, failure):
    """Even after the new attempt minted a different client/token pair, R1 remains coherent."""
    from types import SimpleNamespace
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
    from hermes_cli import mcp_config
    from tools.connectors.mcp_oauth import probe_with_rollback
    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_tool_loop import _ensure_mcp_loop, _run_on_mcp_loop

    root, a, _b, peer, activate = estate
    activate(a)
    flow = SimpleNamespace(cancelled=failure == "cancel")

    async def mint():
        storage = HermesTokenStorage("team", requested=SERVER)
        assert Path(storage.pool_path) == _grant(root)
        assert storage._tokens_path() != _grant(root)
        await storage.set_client_info(OAuthClientInformationFull(client_id="NEW_CLIENT"))
        await storage.set_tokens(OAuthToken(access_token="NEW_ATTEMPT", token_type="Bearer",
                                           refresh_token="NEW_REFRESH", expires_in=3600))

    def probe(*args, details, **kwargs):
        _ensure_mcp_loop()
        _run_on_mcp_loop(mint)
        assert peer.send("refresh") == {"fenced": True}
        peer.proc.stdin.write("\n")
        peer.proc.stdin.flush()
        assert peer.read() == {"ok": True}
        if failure == "initialize":
            raise RuntimeError("initialize failed")
        details["initialized"] = True
        return []

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    if failure == "save":
        def failed_save(config):
            raise OSError("config write failed")
        monkeypatch.setattr(mcp_config, "save_config", failed_save)
    with pytest.raises((RuntimeError, OSError)):
        probe_with_rollback("team", SERVER, str(a), flow, False)
    assert json.loads(_grant(root).read_text())["refresh_token"] == "ROTATED_REFRESH"
    assert json.loads((root / "mcp-tokens/team.client.json").read_text())["client_id"] == "cid"
    assert not list((root / "mcp-tokens").glob(".reauth-*"))
    assert not (root / "mcp-tokens/team.quarantined").exists()


def test_config_publication_is_already_quarantined_and_failed_save_restores_access(estate, monkeypatch):
    from hermes_cli import mcp_config
    from tools.mcp_oauth import HermesTokenStorage

    root, _a, _b, peer, activate = estate
    activate(root)
    original = _grant(root).read_bytes()

    def failed_save(config):
        storage = HermesTokenStorage("team", hermes_home=root)
        assert storage.rebuild_blocked()
        assert not storage.has_cached_tokens()
        assert peer.send("probe") == {"ok": False, "error": "OAuthAuthorityUnavailableError"}
        raise OSError("disk full")

    monkeypatch.setattr(mcp_config, "save_config", failed_save)
    with pytest.raises(OSError, match="disk full"):
        mcp_config._save_mcp_server("team", {**EXPORT, "url": "https://new.example.com"})
    assert _grant(root).read_bytes() == original
    assert peer.send("probe") == {"ok": True, "authorization": "Bearer ROOT_GRANT"}
    assert not (root / "mcp-tokens/team.quarantined").exists()


@pytest.mark.parametrize("change_identity", [False, True])
def test_connector_publishes_coherent_staged_grant_and_updates_live_peer(estate, monkeypatch, change_identity):
    import asyncio
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
    from hermes_cli import mcp_config
    from tools.connectors.mcp_oauth import probe_with_rollback
    from tools.mcp_oauth import HermesTokenStorage

    root, _a, _b, peer, activate = estate
    activate(root)
    cfg = {**EXPORT, "url": "https://new.example.com/mcp"} if change_identity else EXPORT

    def probe(*args, details, **kwargs):
        storage = HermesTokenStorage("team", requested=cfg)
        asyncio.run(storage.set_client_info(OAuthClientInformationFull(client_id="NEW_CLIENT")))
        asyncio.run(storage.set_tokens(OAuthToken(access_token="NEW_GRANT", token_type="Bearer", expires_in=3600)))
        storage._meta_path().write_bytes((root / "mcp-tokens/team.meta.json").read_bytes())
        assert json.loads(_grant(root).read_text())["access_token"] == "ROOT_GRANT"
        details["initialized"] = True
        return []

    monkeypatch.setattr(mcp_config, "_probe_single_server", probe)
    probe_with_rollback("team", cfg, str(root), None, False)
    assert json.loads(_grant(root).read_text())["access_token"] == "NEW_GRANT"
    assert json.loads((root / "mcp-tokens/team.client.json").read_text())["client_id"] == "NEW_CLIENT"
    assert not list((root / "mcp-tokens").glob(".reauth-*"))
    assert not (root / "mcp-tokens/team.quarantined").exists()
    assert peer.send("probe") == (
        {"ok": False, "error": "OAuthStorageDetachedError"} if change_identity
        else {"ok": True, "authorization": "Bearer NEW_GRANT"})
