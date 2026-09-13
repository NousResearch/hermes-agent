"""Two profiles that share an OAuth MCP server URL keep distinct identity state (#109422):

a profile NEVER adopts another profile's live connection when the two authenticate as
different users (different on-disk OAuth token files). Profiles with identical token
state, or non-OAuth routes, still share as before.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

OAUTH_CFG = {
    "url": "https://mcp.example/cal",
    "auth": "oauth",
    "enabled": True,
    "oauth": {"client_id": "shared-client", "scope": "calendar.events"},
}


def _tool():
    return SimpleNamespace(name="whoami", description="d",
                           inputSchema={"type": "object", "properties": {}}, annotations=None)


def _server(name, cfg):
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool()],
                           tool_timeout=30, initialize_result=None,
                           _registered_tool_names=[], _sampling=None)


def _seed_tokens(home: Path, owner: str) -> None:
    """Pre-seed a completed OAuth flow for profile *owner* (what `hermes mcp login` would leave)."""
    tok = home / "mcp-tokens"
    tok.mkdir(parents=True, exist_ok=True)
    (tok / "cal.json").write_text(json.dumps({
        "access_token": f"tok-{owner}", "token_type": "Bearer",
        "expires_in": 3600, "expires_at": time.time() + 3600,
        "refresh_token": f"rt-{owner}", "scope": "calendar.events"}))
    (tok / "cal.client.json").write_text(json.dumps({
        "client_id": "shared-client",
        "redirect_uris": ["http://127.0.0.1:8377/oauth/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none"}))
    (tok / "cal.meta.json").write_text(json.dumps({
        "issuer": "https://accounts.example", "token_endpoint": "https://accounts.example/token"}))


def _sync_token_files(src_home: Path, dst_home: Path) -> None:
    dst = dst_home / "mcp-tokens"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("cal.json", "cal.client.json", "cal.meta.json"):
        shutil.copy(src_home / "mcp-tokens" / f, dst / f)


@pytest.fixture
def two_profiles(tmp_path, monkeypatch):
    """Multiplex on, clean MCP ledgers, a scope switcher for homes A and B; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    homes = {k: tmp_path / "profiles" / k for k in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_mcp_tool_server_names", "_orphaned_adopters")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    tokens = []

    def enter(which, seed=True):
        home = homes[which]
        if seed:
            _seed_tokens(home, which.upper())
        tokens.append(set_hermes_home_override(home))
        return hermes_home_key(home)

    yield enter, homes
    for tool_name in ("mcp__cal__whoami", "mcp__x__whoami"):
        for home in homes.values():
            registry.deregister(tool_name, scope=hermes_home_key(home))
    for token in reversed(tokens):
        reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def test_oauth_profiles_with_distinct_tokens_do_not_cross_adopt(two_profiles):
    """The #109422 acceptance case: two profiles, identical OAuth config, different token
    files. B's discovery must NOT adopt A's live connection — B stays a connect candidate
    so it opens its OWN connection authenticated as B's user."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    enter, _ = two_profiles
    scope_a = enter("a")
    srv_a = _server("cal", OAUTH_CFG)
    disc._adopt_server("cal", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("cal", srv_a, OAUTH_CFG)
    assert registry.snapshot_registration("mcp__cal__whoami", scope=scope_a) is not None

    scope_b = enter("b")
    # B must not record itself into A's scope and must not be healed onto A's connection.
    assert reg.register_connected_into_current_scope({"cal": OAUTH_CFG}) == 0
    with core._lock:
        assert scope_b not in core._server_tool_scopes.get((scope_a, "cal"), ())
        assert (scope_b, "cal") not in core._servers
    # B's own key stays a connect candidate: B will open its OWN connection (not shadowed by A).
    assert "cal" in disc._select_new_servers({"cal": OAUTH_CFG})
    assert registry.snapshot_registration("mcp__cal__whoami", scope=scope_b) is None


def test_oauth_profiles_with_identical_token_state_still_share(two_profiles):
    """Same token state on both sides is one identity: adoption still works — no regression
    on the sharing fast path."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    enter, homes = two_profiles
    scope_a = enter("a")
    srv_a = _server("cal", OAUTH_CFG)
    disc._adopt_server("cal", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("cal", srv_a, OAUTH_CFG)

    _sync_token_files(homes["a"], homes["b"])
    scope_b = enter("b", seed=False)
    assert reg.register_connected_into_current_scope({"cal": OAUTH_CFG}) == 1
    assert registry.snapshot_registration("mcp__cal__whoami", scope=scope_b) is not None
    with core._lock:
        assert scope_b in core._server_tool_scopes.get((scope_a, "cal"), ())


def test_non_oauth_shared_route_is_unchanged(two_profiles):
    """No ``auth: oauth``: header/static routes keep the legacy credential-free sharing."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "headers": {"Authorization": "Bearer shared"}}
    enter, _ = two_profiles
    scope_a = enter("a", seed=False)
    srv_a = _server("x", cfg)
    disc._adopt_server("x", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("x", srv_a, cfg)

    scope_b = enter("b", seed=False)
    assert reg.register_connected_into_current_scope({"x": cfg}) == 1
    assert registry.snapshot_registration("mcp__x__whoami", scope=scope_b) is not None
    with core._lock:
        assert scope_b in core._server_tool_scopes.get((scope_a, "x"), ())


def test_oauth_caller_without_token_file_still_adopts(two_profiles):
    """Caller has ``auth: oauth`` but no token file yet (first-use profile): no identity to
    compare, so the shared route is still adoptable — the pending flow behaves as before."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from tools.registry import registry

    enter, _ = two_profiles
    scope_a = enter("a")
    srv_a = _server("cal", OAUTH_CFG)
    disc._adopt_server("cal", srv_a)
    srv_a._registered_tool_names = reg._register_server_tools("cal", srv_a, OAUTH_CFG)

    scope_b = enter("b", seed=False)  # B has NO mcp-tokens at all
    assert reg.register_connected_into_current_scope({"cal": OAUTH_CFG}) == 1
    assert registry.snapshot_registration("mcp__cal__whoami", scope=scope_b) is not None


def test_oauth_credential_fingerprint_distinct_per_home(two_profiles):
    """Direct check: token material under different homes produces different fingerprints,
    and the same home compares stably (no per-call nonce)."""
    from tools import mcp_tool_registration as reg

    enter, homes = two_profiles
    home_a = enter("a")   # seeds A's mcp-tokens
    home_b = enter("b")   # seeds B's mcp-tokens (distinct token values)
    assert home_b != home_a
    fp_a = reg._oauth_credential_fingerprint("cal", "oauth", hermes_home=home_a)
    fp_b = reg._oauth_credential_fingerprint("cal", "oauth", hermes_home=home_b)
    assert fp_a and fp_b and fp_a != fp_b
    assert fp_a == reg._oauth_credential_fingerprint("cal", "oauth", hermes_home=home_a)
    # No auth method -> no fingerprint.
    assert reg._oauth_credential_fingerprint("cal", "", hermes_home=home_a) is None


def test_same_server_route_oauth_distinct_tokens(two_profiles):
    """Predicate-level check of the adoption gate itself."""
    from tools import mcp_tool_registration as reg

    enter, homes = two_profiles
    scope_a = enter("a")
    srv_a = _server("cal", OAUTH_CFG)
    enter("b")
    # Different token files on both sides -> not the same route (the #109422 blocker).
    assert reg._same_server_route(srv_a, OAUTH_CFG, "cal", owner_home=str(homes["a"])) is False
    # Identical token content on both sides -> shareable.
    _sync_token_files(homes["a"], homes["b"])
    assert reg._same_server_route(srv_a, OAUTH_CFG, "cal", owner_home=str(homes["a"])) is True
    # Different static route -> never shareable, regardless of token state.
    other_route = dict(OAUTH_CFG, url="https://mcp.example/other")
    assert reg._same_server_route(srv_a, other_route, "cal", owner_home=str(homes["a"])) is False
    # Deliberately drop the owner hint: caller-without-token-files case still shares.
    homes_b = homes["b"]
    for f in ("cal.json", "cal.client.json", "cal.meta.json"):
        (homes_b / "mcp-tokens" / f).unlink(missing_ok=True)
    assert reg._same_server_route(srv_a, OAUTH_CFG, "cal") is True


def test_single_profile_behavior_unchanged(tmp_path, monkeypatch):
    """No multiplexer: bare name keys, self-comparison of the identity always passes, and
    scoped no-ops stay no-ops — the legacy single-profile behavior, unchanged."""
    import tools.mcp_tool as core
    from tools import mcp_tool_discovery as disc, mcp_tool_registration as reg
    from hermes_constants import get_hermes_home

    home = tmp_path / "home"
    home.mkdir()
    _seed_tokens(home, "A")
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: False)
    tokens = [set_hermes_home_override(home)]
    try:
        srv = _server("cal", OAUTH_CFG)
        assert core._mcp_registry_scope() is None
        disc._adopt_server("cal", srv)
        assert "cal" in core._servers  # bare name key, not (scope, name)
        assert reg._same_server_route(srv, OAUTH_CFG, "cal") is True  # self-compare under own home
        assert reg.register_connected_into_current_scope({"cal": OAUTH_CFG}) == 0  # no scope: no-op
        assert str(get_hermes_home()) == str(home)
    finally:
        for token in reversed(tokens):
            reset_hermes_home_override(token)
        for key in ("cal", ("a", "cal"), ("b", "cal")):
            core._servers.pop(key, None)
            core._server_scope_keys.pop(key, None)
