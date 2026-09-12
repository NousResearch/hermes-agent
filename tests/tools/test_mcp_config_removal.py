"""Live MCP tasks stop when another process removes their native config."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import tools.mcp_tool as mcp_tool
from tools import mcp_tool_config, mcp_tool_discovery


@pytest.mark.asyncio
async def test_removed_native_server_is_not_reconnected(monkeypatch):
    """A transport teardown after config removal must not spawn a successor."""
    configured = {"enabled": True}
    calls = 0
    server = mcp_tool.MCPServerTask("removed")

    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_config",
        lambda _name: (True, {"command": "fake"} if configured["enabled"] else None),
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        configured["enabled"] = False
        return "reconnect"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", lambda _server: None)
    with mcp_tool._lock:
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None

    try:
        await server.run(mcp_tool_config._MCPServerConfig(
            {"command": "fake"}, native_config_managed=True))

        assert calls == 1
        assert server._shutdown_event.is_set()
        assert server.name not in mcp_tool._servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)


@pytest.mark.asyncio
async def test_native_snapshot_removed_before_run_never_spawns(monkeypatch):
    """Discovery provenance survives removal before task initialization."""
    calls = 0
    server = mcp_tool.MCPServerTask("removed-before-run")
    snapshot = mcp_tool_config._MCPServerConfig(
        {"command": "fake"}, native_config_managed=True)

    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_config",
        lambda _name: (True, None),
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", lambda _server: None)
    with mcp_tool._lock:
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None

    try:
        await server.run(snapshot)

        assert calls == 0
        assert server._shutdown_event.is_set()
        assert server.name not in mcp_tool._servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)


def test_discovery_completes_when_native_snapshot_was_removed(monkeypatch):
    """Pre-spawn retirement is a clean discovery outcome, not a timeout."""
    name = "removed-during-discovery"
    calls = 0
    snapshot = mcp_tool_config._MCPServerConfig(
        {"command": "fake", "supports_parallel_tool_calls": True},
        native_config_managed=True,
    )

    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(
        mcp_tool_config,
        "_native_mcp_server_config",
        lambda _name: (True, None),
    )
    monkeypatch.setattr(
        mcp_tool_config,
        "_filter_suspicious_mcp_servers",
        lambda servers: servers,
    )

    async def transport(_server, _config):
        nonlocal calls
        calls += 1
        return "shutdown"

    monkeypatch.setattr(mcp_tool.MCPServerTask, "_run_stdio", transport)
    with mcp_tool._lock:
        mcp_tool._server_tool_scopes[name] = {"stale-scope"}
        mcp_tool._server_connect_errors[name] = "stale error"
        mcp_tool._server_connect_failures[name] = 2
        mcp_tool._server_connect_retry_after[name] = time.monotonic() - 1

    started = time.monotonic()
    try:
        assert mcp_tool_discovery.register_mcp_servers({name: snapshot}) == []
        assert time.monotonic() - started < 2
        assert calls == 0
        with mcp_tool._lock:
            assert name not in mcp_tool._servers
            assert name not in mcp_tool._server_scope_keys
            assert name not in mcp_tool._server_tool_scopes
            assert name not in mcp_tool._server_connecting
            assert name not in mcp_tool._server_connect_errors
            assert name not in mcp_tool._server_connect_failures
            assert name not in mcp_tool._server_connect_retry_after
            assert name not in mcp_tool._parallel_safe_servers
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(name, None)
            mcp_tool._server_scope_keys.pop(name, None)
            mcp_tool._server_tool_scopes.pop(name, None)
            mcp_tool._server_connecting.discard(name)
            mcp_tool._server_connect_errors.pop(name, None)
            mcp_tool._server_connect_failures.pop(name, None)
            mcp_tool._server_connect_retry_after.pop(name, None)
            mcp_tool._parallel_safe_servers.discard(name)


def test_managed_effective_config_controls_native_membership(tmp_path, monkeypatch):
    """The retirement probe reads the same managed overlay as discovery."""
    from hermes_cli import config as hermes_config, managed_scope

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_config = managed / "config.yaml"
    managed_config.write_text(
        "mcp_servers:\n  pinned:\n    command: managed-command\n",
        encoding="utf-8",
    )
    hermes_config._LOAD_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()

    known, config = mcp_tool_config._native_mcp_server_config("pinned")
    assert known is True
    assert config == {"command": "managed-command"}

    managed_config.write_text(
        "mcp_servers:\n  pinned:\n    command: managed-command\n    enabled: false\n",
        encoding="utf-8",
    )
    hermes_config._LOAD_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    assert mcp_tool_config._native_mcp_server_config("pinned") == (True, None)


def test_owner_removal_preserves_authorized_adopter(tmp_path, monkeypatch):
    """Revoking launch scope A keeps an identical connection callable in scope B."""
    from hermes_cli import config as hermes_config
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool_registration
    from tools.registry import registry

    homes = {name: tmp_path / name for name in ("a", "b")}
    cfg = {"url": "https://mcp.example/shared", "headers": {"Authorization": "Bearer shared"}}
    for home in homes.values():
        home.mkdir()
        (home / "config.yaml").write_text(
            "mcp_servers:\n  shared:\n    url: https://mcp.example/shared\n"
            "    headers:\n      Authorization: Bearer shared\n",
            encoding="utf-8",
        )
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    server = mcp_tool.MCPServerTask("shared")
    server._config = cfg
    server._native_config_managed = True
    server.session = object()
    server._tools = [SimpleNamespace(
        name="ping", description="", inputSchema={"type": "object", "properties": {}}, annotations=None)]
    server.initialize_result = None
    server._registered_tool_names = []

    tokens = []
    try:
        tokens.append(set_hermes_home_override(homes["a"]))
        scope_a = hermes_home_key(homes["a"])
        mcp_tool_discovery._adopt_server("shared", server)
        server._registered_tool_names = mcp_tool_registration._register_server_tools("shared", server, cfg)

        tokens.append(set_hermes_home_override(homes["b"]))
        scope_b = hermes_home_key(homes["b"])
        assert mcp_tool_registration.register_connected_into_current_scope({"shared": cfg}) == 1

        (homes["a"] / "config.yaml").write_text("{}\n", encoding="utf-8")
        hermes_config._LOAD_CONFIG_CACHE.clear()
        assert server._retire_if_removed_from_config() is False
        assert registry.snapshot_registration("mcp__shared__ping", scope=scope_a) is None
        assert registry.snapshot_registration("mcp__shared__ping", scope=scope_b) is not None
        key = (scope_a, "shared")
        assert mcp_tool._server_scope_keys[key] == scope_b
        assert mcp_tool._servers[key] is server

        (homes["b"] / "config.yaml").write_text("{}\n", encoding="utf-8")
        hermes_config._LOAD_CONFIG_CACHE.clear()
        assert server._retire_if_removed_from_config() is True
        assert registry.snapshot_registration("mcp__shared__ping", scope=scope_b) is None
        assert key not in mcp_tool._servers
    finally:
        for token in reversed(tokens):
            reset_hermes_home_override(token)
        with mcp_tool._lock:
            for ledger in (
                mcp_tool._servers, mcp_tool._server_scope_keys, mcp_tool._server_tool_scopes,
                mcp_tool._server_connect_errors, mcp_tool._server_connect_failures,
                mcp_tool._server_connect_retry_after, mcp_tool._server_error_counts,
                mcp_tool._server_breaker_opened_at, mcp_tool._server_errors_all_application,
                mcp_tool._server_trust_levels, mcp_tool._tool_read_only_hints,
            ):
                ledger.clear()
            mcp_tool._server_connecting.clear()


def test_adopter_removal_deregisters_its_filtered_tools(tmp_path, monkeypatch):
    """A revoked adopter loses tools not present in the launch owner's filter."""
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool_registration
    from tools.registry import registry

    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    cfg = {"url": "https://mcp.example/shared"}
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    server = mcp_tool.MCPServerTask("shared")
    server._config = cfg
    server._native_config_managed = True
    server.session = object()
    server._registered_tool_names = ["mcp__shared__owner", "mcp__shared__adopter"]
    tokens = []
    try:
        tokens.append(set_hermes_home_override(homes["a"]))
        scope_a = hermes_home_key(homes["a"])
        mcp_tool_discovery._adopt_server("shared", server)
        registry.register(
            name="mcp__shared__owner", toolset="mcp-shared",
            schema={"name": "mcp__shared__owner", "parameters": {"type": "object", "properties": {}}},
            handler=lambda _args: "ok", scope=scope_a)
        with mcp_tool._lock:
            key = (scope_a, "shared")
            mcp_tool._server_tool_scopes[key] = {scope_a}

        tokens.append(set_hermes_home_override(homes["b"]))
        scope_b = hermes_home_key(homes["b"])
        registry.register(
            name="mcp__shared__adopter", toolset="mcp-shared",
            schema={"name": "mcp__shared__adopter", "parameters": {"type": "object", "properties": {}}},
            handler=lambda _args: "ok", scope=scope_b)
        with mcp_tool._lock:
            mcp_tool._server_tool_scopes[key].add(scope_b)

        monkeypatch.setattr(
            mcp_tool_config, "_native_mcp_server_config",
            lambda _name: (True, cfg if mcp_tool._mcp_registry_scope() == scope_a else None))
        assert server._retire_if_removed_from_config() is False
        assert registry.snapshot_registration("mcp__shared__owner", scope=scope_a) is not None
        assert registry.snapshot_registration("mcp__shared__adopter", scope=scope_b) is None
    finally:
        for tool_name in server._registered_tool_names:
            for scope in (locals().get("scope_a"), locals().get("scope_b")):
                if scope is not None:
                    registry.deregister(tool_name, scope=scope)
        for token in reversed(tokens):
            reset_hermes_home_override(token)
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._server_scope_keys.clear()
            mcp_tool._server_tool_scopes.clear()


def test_scope_change_during_poll_defers_retirement(monkeypatch):
    """A concurrent adopter prevents retirement from a stale authority snapshot."""
    server = mcp_tool.MCPServerTask("shared")
    server._config = {"command": "fake"}
    server._native_config_managed = True
    with mcp_tool._lock:
        mcp_tool._servers[server.name] = server
        mcp_tool._server_scope_keys[server.name] = None

    def changed_authority(_name):
        with mcp_tool._lock:
            mcp_tool._server_tool_scopes[server.name] = {"new-scope"}
        return True, None

    monkeypatch.setattr(mcp_tool_config, "_native_mcp_server_config", changed_authority)
    try:
        assert server._retire_if_removed_from_config() is False
        assert server._retired_from_config is False
        assert mcp_tool._servers[server.name] is server
    finally:
        with mcp_tool._lock:
            mcp_tool._servers.pop(server.name, None)
            mcp_tool._server_scope_keys.pop(server.name, None)
            mcp_tool._server_tool_scopes.pop(server.name, None)


def test_discard_retired_lazy_candidate_removes_cached_overlay(monkeypatch):
    """Removal before first lazy spawn deletes its schema registration and ledgers."""
    from tools.registry import registry

    name = "lazy-removed"
    tool_name = "mcp__lazy_removed__ping"
    registry.register(
        name=tool_name, toolset=f"mcp-{name}",
        schema={"name": tool_name, "parameters": {"type": "object", "properties": {}}},
        handler=lambda _args: "ok")
    with mcp_tool._lock:
        mcp_tool._lazy_server_configs[name] = {"command": "fake"}
        mcp_tool._lazy_server_fingerprints[name] = "fingerprint"
        mcp_tool._lazy_server_tool_names[name] = [tool_name]
    try:
        mcp_tool_discovery._discard_retired_candidate(name, mcp_tool.MCPServerTask(name))
        assert registry.snapshot_registration(tool_name) is None
        assert name not in mcp_tool._lazy_server_configs
        assert name not in mcp_tool._lazy_server_fingerprints
        assert name not in mcp_tool._lazy_server_tool_names
    finally:
        registry.deregister(tool_name)


@pytest.mark.asyncio
async def test_parked_native_task_polls_config_without_waiting_for_revival(monkeypatch):
    """An untimed recycled/parked wait still observes native removal promptly."""
    server = mcp_tool.MCPServerTask("parked")
    server._config = {"command": "fake"}
    server._native_config_managed = True
    monkeypatch.setattr(mcp_tool, "_MCP_CONFIG_POLL_INTERVAL", 0.01)
    monkeypatch.setattr(mcp_tool_config, "_native_mcp_server_config", lambda _name: (True, None))
    monkeypatch.setattr(mcp_tool.MCPServerTask, "_deregister_tools", lambda _server: None)

    result = await asyncio.wait_for(server._wait_for_reconnect_or_shutdown(), timeout=0.5)
    assert result == "shutdown"
    assert server._retired_from_config is True


def test_adoption_revalidates_after_retirement_cas(tmp_path, monkeypatch):
    """A profile cannot publish an overlay after its selected task retires."""
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool_registration
    from tools.registry import registry

    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    cfg = {"url": "https://mcp.example/shared"}
    server = mcp_tool.MCPServerTask("shared")
    server._config = cfg
    server.session = object()
    server._tools = [SimpleNamespace(
        name="ping", description="", inputSchema={"type": "object", "properties": {}}, annotations=None)]
    server.initialize_result = None
    server._registered_tool_names = []
    selected = threading.Event()
    resume = threading.Event()

    class GateLock:
        def __enter__(self):
            selected.set()
            assert resume.wait(2)

        def __exit__(self, *_exc):
            return False

    token = set_hermes_home_override(homes["a"])
    scope_a = hermes_home_key(homes["a"])
    try:
        mcp_tool_discovery._adopt_server("shared", server)
        key = (scope_a, "shared")
        server._config_authority_lock = GateLock()
        result = []

        def adopt_b():
            worker_token = set_hermes_home_override(homes["b"])
            try:
                result.append(mcp_tool_registration.register_connected_into_current_scope({"shared": cfg}))
            finally:
                reset_hermes_home_override(worker_token)

        worker = threading.Thread(target=adopt_b)
        worker.start()
        assert selected.wait(2)
        with mcp_tool._lock:
            server._retired_from_config = True
            mcp_tool._servers.pop(key)
        resume.set()
        worker.join(2)
        assert not worker.is_alive()
        scope_b = hermes_home_key(homes["b"])
        assert result == [0]
        assert scope_b not in mcp_tool._server_tool_scopes.get(key, set())
        assert registry.snapshot_registration("mcp__shared__ping", scope=scope_b) is None
    finally:
        resume.set()
        reset_hermes_home_override(token)
        with mcp_tool._lock:
            mcp_tool._servers.clear()
            mcp_tool._server_scope_keys.clear()
            mcp_tool._server_tool_scopes.clear()
