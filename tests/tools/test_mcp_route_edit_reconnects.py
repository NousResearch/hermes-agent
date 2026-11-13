"""A url / header / auth edit to a live ``mcp_servers`` entry must RECONNECT that server, not leave
it up with zero registered tools.

The reconcile answered a changed ``_connection_identity`` on the profile's OWN live connection by
removing that profile's overlay and KEEPING the connection (``_remove_server_scope``: tools
deregistered, ``_core._servers`` entry and scope dropped, nothing logged). Discovery then never
rebuilt anything — a key already in ``_core._servers`` is not a connect candidate
(``_select_new_servers``), and the overlay pass skips this profile's own connection — so only a
restart or ``/reload-mcp`` cleared it. The route and the credentials are in the identity, so they
must replace the connection rather than detach from it.

An ADOPTED connection (owned by another profile) keeps the detach-only verdict: its session belongs
to its owner, whose config did not change.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override

_URL = "https://mcp.example/x"
_URL_MOVED = "https://mcp.example/y"


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name: str, cfg: dict, tools: tuple[str, ...] = ("t",)) -> SimpleNamespace:
    """A live ``MCPServerTask`` stand-in: session up, config as connected, tools as listed."""
    from tools.mcp_tool_registration import _adopter_identity_digest
    return SimpleNamespace(name=name, session=object(), _config=cfg, _tools=[_tool(t) for t in tools],
                           tool_timeout=30, initialize_result=None, _registered_tool_names=[],
                           _sampling=None, _resolved_identity=_adopter_identity_digest(name, cfg))


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """One served profile scope, clean MCP ledgers; restores everything."""
    import tools.mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools.registry import registry

    home = tmp_path / "profiles" / "hanna"
    (home / "cache").mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_lazy_server_fingerprints", "_lazy_server_tool_names", "_mcp_tool_server_names",
               "_orphaned_adopters", "_parallel_safe_servers", "_server_trust_levels",
               "_tool_read_only_hints", "_server_cost_policies")
    saved = {n: type(getattr(core, n))(getattr(core, n)) for n in ledgers}
    for n in ledgers:
        getattr(core, n).clear()
    registry.register_toolset_alias("x", "mcp-x")
    token = set_hermes_home_override(home)
    scope = hermes_home_key(home)

    yield SimpleNamespace(home=home, scope=scope, registry=registry, core=core)

    for tool_name in list(registry.get_tool_names_for_toolset("mcp-x")):
        registry.deregister(tool_name, scope=scope)
        registry.deregister(tool_name)
    registry._drop_toolset_aliases("mcp-x")
    reset_hermes_home_override(token)
    for n in ledgers:
        getattr(core, n).clear()
        getattr(core, n).update(saved[n])


def _config(url: str = _URL, headers: dict | None = None, include: list | None = None) -> dict:
    cfg = {"url": url, "headers": dict(headers or {"Authorization": "Bearer a"})}
    if include is not None:
        cfg["tools"] = {"include": list(include)}
    return cfg


def _record_shutdown(server, closed: list[str], key=None) -> None:
    """Attach the ``async def shutdown()`` a live task has (the close runs on the MCP loop).

    A real ``MCPServerTask.shutdown()`` also deregisters that connection's tools from EVERY scope that
    borrowed it (``_deregister_tools``), which the adopter case depends on: without that, a stand-in
    teardown leaves the borrower's overlay in place and hides whether the re-adoption rebuilt it. Pass
    *key* to mirror it."""
    async def _shutdown(self):
        closed.append(self.name)
        if key is None:
            return
        from tools import mcp_tool_registration as reg
        from tools.registry import registry
        scopes = set(reg._core._server_tool_scopes.get(key, ()))
        for tool_name in reg._native_tool_names(self.name, self):
            for scope in scopes:
                registry.deregister(tool_name, scope=scope)
    server.shutdown = _shutdown.__get__(server)


def _stub_connect(monkeypatch, disc, reg, connected: list) -> None:
    """Stand in for the transport: open a live connection with the config it is asked to run, and
    register its tools exactly as ``_discover_and_register_server`` does afterwards."""
    async def _connect(name: str, config: dict) -> list[str]:
        server = _server(name, config)
        disc._adopt_server(name, server)
        connected.append((name, config))
        return reg._register_server_tools(name, server, config)
    monkeypatch.setattr(disc, "_discover_and_register_server", _connect)


def test_editing_the_url_reconnects_and_reregisters_under_the_new_route(profile, monkeypatch):
    """Pointing a live server at a new endpoint must end with a connection on THAT endpoint and
    its tools registered. Before this fix the old session stayed up on the old endpoint with the
    profile's overlay gone, and nothing reconnected it for the life of the process."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg

    before, after = _config(), _config(url=_URL_MOVED)
    old = _server("x", before)
    disc._adopt_server("x", old)
    assert reg._register_server_tools("x", old, before) == ["mcp__x__t"]

    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)

    loop._ensure_mcp_loop()
    try:
        disc.register_mcp_servers({"x": after})
    finally:
        loop._stop_mcp_loop()

    assert closed == ["x"], "a connection whose route changed must be closed, not left serving the old one"
    assert connected == [("x", after)], "the same discovery pass must reconnect it with the new config"
    new = profile.core._servers[(profile.scope, "x")]
    assert new is not old
    assert new._config == after
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"], \
        "the tools must be registered again, on the connection that now runs"
    assert profile.scope in profile.core._server_tool_scopes[(profile.scope, "x")]


def test_the_reconcile_tick_reconnects_a_url_edited_server(profile, monkeypatch):
    """The housekeeping tick, and the Desktop repair verb behind it, go through
    ``reconcile_mcp_servers_with_config``. A url edit changes no NAME, so the tick used to see
    nothing to do: nothing to prune, nothing to add, discovery never ran, and the edit waited for a
    restart. It must count the server as needing a connect, which is what replaces the stale session."""
    from tools import mcp_tool_config as _config_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg

    before, after = _config(), _config(url=_URL_MOVED)
    old = _server("x", before)
    disc._adopt_server("x", old)
    assert reg._register_server_tools("x", old, before) == ["mcp__x__t"]

    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)
    monkeypatch.setattr(_config_mod, "_load_mcp_config", lambda: {"x": copy.deepcopy(after)})

    loop._ensure_mcp_loop()
    try:
        result = disc.reconcile_mcp_servers_with_config()
    finally:
        loop._stop_mcp_loop()

    assert result["added"] == ["x"], result
    assert closed == ["x"], "the tick must replace the connection, not leave it on the old endpoint"
    assert connected == [("x", after)]
    assert profile.core._servers[(profile.scope, "x")]._config == after
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_an_adopted_connection_is_still_only_detached(profile):
    """Another profile's live connection is not this profile's to close. When the borrower's config
    moves the route, its overlay is detached and the OWNER's session keeps running: closing it would
    cut the owner's own tools, and every other adopter's, over an edit the owner never made."""
    import tools.mcp_tool as core
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    owner_scope = "another-profile"
    before, after = _config(), _config(url=_URL_MOVED)
    owned = _server("x", before)
    key = (owner_scope, "x")
    with core._lock:
        core._servers[key] = owned
        core._server_scope_keys[key] = owner_scope
        core._server_tool_scopes[key] = {owner_scope, profile.scope}
    closed: list[str] = []
    _record_shutdown(owned, closed)
    for scope in (owner_scope, profile.scope):  # the owner's overlay, and the one this profile adopted
        registry.register(name="mcp__x__t", toolset="mcp-x",
                          schema={"name": "mcp__x__t", "parameters": {"type": "object", "properties": {}}},
                          handler=lambda args, **kwargs: "x", scope=scope)
    try:
        assert registry.snapshot_registration("mcp__x__t", scope=profile.scope) is not None

        reg.register_connected_into_current_scope({"x": after})

        assert registry.snapshot_registration("mcp__x__t", scope=profile.scope) is None, \
            "the borrower's overlay of the old route must go"
        assert registry.snapshot_registration("mcp__x__t", scope=owner_scope) is not None, \
            "removal stays scope-precise: the owner still serves the tool"
        assert core._servers[key] is owned, "the owner's connection is not this profile's to close"
        assert closed == [], "never shut down a connection owned by another profile"
        assert profile.scope not in core._server_tool_scopes[key]
    finally:
        with core._lock:
            core._servers.pop(key, None)
            core._server_scope_keys.pop(key, None)
            core._server_tool_scopes.pop(key, None)
        for scope in (owner_scope, profile.scope):
            registry.deregister("mcp__x__t", scope=scope)


def test_editing_the_auth_header_reconnects_too(profile, monkeypatch):
    """A rotated header is the same kind of edit as a moved endpoint: the live session carries the
    old credential, so it must be replaced, not detached from."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg

    before, after = _config(), _config(headers={"Authorization": "Bearer b"})
    old = _server("x", before)
    disc._adopt_server("x", old)
    assert reg._register_server_tools("x", old, before) == ["mcp__x__t"]

    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)

    loop._ensure_mcp_loop()
    try:
        disc.register_mcp_servers({"x": after})
    finally:
        loop._stop_mcp_loop()

    assert closed == ["x"] and connected == [("x", after)]
    assert profile.core._servers[(profile.scope, "x")]._config == after
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_a_stable_config_never_restarts_the_connection(profile, monkeypatch):
    """The verdict must be "the config moved", not "the config is not the live one": a comparison
    that fires on a settled config would respawn the server on every turn."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg

    cfg = _config(include=["t"])
    server = _server("x", cfg, tools=("t", "t2"))
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, cfg) == ["mcp__x__t"]

    closed, connected = [], []
    _record_shutdown(server, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)

    loop._ensure_mcp_loop()
    try:
        for _ in range(3):
            disc.register_mcp_servers({"x": cfg})
    finally:
        loop._stop_mcp_loop()

    assert closed == [] and connected == [], "a settled config must not restart the server"
    assert profile.core._servers[(profile.scope, "x")] is server
    assert profile.registry.get_tool_names_for_toolset("mcp-x") == ["mcp__x__t"]


def test_the_reconcile_tick_keeps_a_settled_config_quiet(profile, monkeypatch):
    """No drift, no edit: the tick must not enter discovery at all (it takes a cross-process lock)."""
    from tools import mcp_tool_config as _config_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = _config()
    server = _server("x", cfg)
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, cfg) == ["mcp__x__t"]
    discovered: list[dict] = []
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: discovered.append(cfg) or [])
    monkeypatch.setattr(_config_mod, "_load_mcp_config", lambda: {"x": copy.deepcopy(cfg)})

    assert disc.reconcile_mcp_servers_with_config() == {"removed": [], "added": [], "pending": []}
    assert discovered == []


def test_a_teardown_that_cannot_complete_is_retried_not_duplicated(profile, monkeypatch):
    """When the close cannot land (here: no MCP loop is running, so the scoped close never reaches
    the task), nothing may pretend the server came back: the old connection stays and no second one
    is dialled beside it. The next pass that CAN close it replaces it."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg

    before, after = _config(), _config(url=_URL_MOVED)
    old = _server("x", before)
    disc._adopt_server("x", old)
    old._registered_tool_names = reg._register_server_tools("x", old, before)
    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)

    disc.register_mcp_servers({"x": after})  # no MCP loop: the close cannot land

    assert profile.core._servers[(profile.scope, "x")] is old, "an unclosed session is never replaced silently"
    assert connected == [], "never dial a second connection beside the one that would not close"

    loop._ensure_mcp_loop()  # the loop the real process always has
    try:
        disc.register_mcp_servers({"x": after})
    finally:
        loop._stop_mcp_loop()

    assert closed == ["x"]
    assert connected == [("x", after)]
    assert profile.core._servers[(profile.scope, "x")]._config == after


def test_an_owner_reconnect_gives_an_adopting_profile_its_tools_back(profile, monkeypatch, tmp_path):
    """A served profile that BORROWED the edited connection must not be collateral: the owner's
    replacement closes that connection, and the adopter re-adopts the new one in the same pass
    (``_reregister_orphaned_adopters``), or its tools would vanish with the owner's edit."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as loop
    from tools import mcp_tool_registration as reg
    from tools.registry import registry

    borrower_home = tmp_path / "profiles" / "borrower"
    borrower_home.mkdir(parents=True)
    after = _config(url=_URL_MOVED)
    (borrower_home / "config.yaml").write_text(
        "mcp_servers:\n  x:\n"
        f"    url: {after['url']}\n"
        "    headers:\n"
        "      Authorization: Bearer a\n", encoding="utf-8")
    before = _config()
    old = _server("x", before)
    disc._adopt_server("x", old)
    assert reg._register_server_tools("x", old, before) == ["mcp__x__t"]
    key = (profile.scope, "x")

    token = set_hermes_home_override(borrower_home)
    borrower_scope = hermes_home_key(borrower_home)
    try:
        with profile.core._lock:
            profile.core._server_tool_scopes[key].add(borrower_scope)
        registry.register(name="mcp__x__t", toolset="mcp-x",
                          schema={"name": "mcp__x__t", "parameters": {"type": "object", "properties": {}}},
                          handler=lambda args, **kwargs: "x", scope=borrower_scope)
        assert registry.snapshot_registration("mcp__x__t", scope=borrower_scope) is not None
    finally:
        reset_hermes_home_override(token)

    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)
    loop._ensure_mcp_loop()
    try:
        disc.register_mcp_servers({"x": after})

        assert closed == ["x"] and connected == [("x", after)]
        assert profile.core._servers[key]._config == after
        assert borrower_scope in profile.core._server_tool_scopes[key], \
            "the borrower re-adopts the replacement"
        assert registry.snapshot_registration("mcp__x__t", scope=borrower_scope) is not None, \
            "the owner's edit must not leave the borrowing profile tool-less"
    finally:
        loop._stop_mcp_loop()
        registry.deregister("mcp__x__t", scope=borrower_scope)


def test_a_single_profile_process_is_left_to_reload_mcp(profile, monkeypatch):
    """The limitation, pinned so a later fix flips it on purpose: without a registry scope there is no
    overlay to rebuild, so the reconcile must NOT claim it replaced anything it cannot. A single-profile
    process (an interactive CLI/TUI) still needs ``/reload-mcp`` after a url edit, exactly as before."""
    from tools import mcp_tool_config as _config_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    monkeypatch.setattr(profile.core, "_mcp_registry_scope", lambda: None)
    cfg = _config()
    server = _server("x", cfg)
    disc._adopt_server("x", server)
    assert reg._register_server_tools("x", server, cfg) == ["mcp__x__t"]
    discovered: list[str] = []
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: discovered.append("ran") or [])
    monkeypatch.setattr(_config_mod, "_load_mcp_config",
                        lambda: {"x": _config(url=_URL_MOVED)})

    assert disc.reconcile_mcp_servers_with_config() == {"removed": [], "added": [], "pending": []}
    assert discovered == [], "a reconnect this path cannot make must not be announced"
    assert profile.core._servers["x"] is server


def test_a_replace_the_pass_could_not_land_is_reported_pending_not_added(profile, monkeypatch):
    """``added`` must say what the pass got, not what it meant to do. When the close cannot land, the
    configured route is still not live: that is a retry (``pending``), not a reconnect, and the caller
    must not be told the server came back."""
    from tools import mcp_tool_config as _config_mod
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    before, after = _config(), _config(url=_URL_MOVED)
    old = _server("x", before)
    disc._adopt_server("x", old)
    old._registered_tool_names = reg._register_server_tools("x", old, before)
    closed, connected = [], []
    _record_shutdown(old, closed, (profile.scope, "x"))
    _stub_connect(monkeypatch, disc, reg, connected)
    monkeypatch.setattr(_config_mod, "_load_mcp_config", lambda: {"x": copy.deepcopy(after)})

    result = disc.reconcile_mcp_servers_with_config()  # no MCP loop: the close cannot land

    assert result == {"removed": [], "added": [], "pending": ["x"]}, result
    assert profile.core._servers[(profile.scope, "x")] is old
    assert connected == [], "never a second connection beside the one that would not close"
