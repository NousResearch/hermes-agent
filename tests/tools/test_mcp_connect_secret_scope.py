"""MCP connect resolves credentials under the connection owner's profile secret scope.

Discovery at gateway startup runs outside any per-turn scope. Under multiplexing an unscoped
``get_secret`` fails closed, so the stdio child-env build raised ``UnscopedSecretError`` and the
server registered zero tools (#113746).
"""
import asyncio
from contextlib import contextmanager

import pytest

from agent.secret_scope import current_secret_scope, set_multiplex_active
from hermes_cli import env_loader
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import mcp_tool_discovery as discovery
from tools.mcp_tool_config import _build_safe_env

TOKEN_NAME, TOKEN_VALUE = "EXAMPLE_TOKEN", "profile-own-value"


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """A profile home whose ``.env`` holds the credential an external source tagged."""
    home = tmp_path / "profile"
    home.mkdir()
    (home / ".env").write_text(f"{TOKEN_NAME}={TOKEN_VALUE}\n", encoding="utf-8")

    # An external secret source (secrets.command / bitwarden / 1password) tags names
    # process-wide; the VALUE must come from the active profile's scope.
    monkeypatch.setitem(env_loader._SECRET_SOURCES, TOKEN_NAME, "command")

    home_token = set_hermes_home_override(str(home))
    set_multiplex_active(True)
    try:
        yield home
    finally:
        set_multiplex_active(False)
        reset_hermes_home_override(home_token)


@pytest.fixture
def spawn_env(monkeypatch):
    """Stub server task: ``start()`` builds the stdio child env the way the transport does."""
    captured = {}

    class _StubServerTask:
        def __init__(self, name):
            self.name = name

        async def start(self, config):
            captured["config"] = config
            captured["env"] = _build_safe_env(config.get("env"))

        async def shutdown(self):
            pass

    # Origin state is read through _core; patch the origin module (see mcp_tool_discovery docstring).
    monkeypatch.setattr("tools.mcp_tool.MCPServerTask", _StubServerTask)
    return captured


def test_connect_resolves_the_owning_profiles_secret(profile_home, spawn_env, monkeypatch):
    """Red on base: UnscopedSecretError. The child env carries the OWNING profile's value — never
    the launch process env's — and the binding is the run task's, not the caller's."""
    monkeypatch.setenv(TOKEN_NAME, "launch-env-value")
    assert current_secret_scope() is None  # discovery runs unscoped

    asyncio.run(discovery._connect_server("demo", {"command": "true"}))

    assert spawn_env["env"][TOKEN_NAME] == TOKEN_VALUE
    assert current_secret_scope() is None


def test_single_profile_process_binds_nothing(tmp_path, monkeypatch, spawn_env):
    """No multiplexer, no home override (scope key None): unchanged, get_secret reads os.environ."""
    monkeypatch.setitem(env_loader._SECRET_SOURCES, TOKEN_NAME, "command")
    monkeypatch.setenv(TOKEN_NAME, "process-env-value")
    set_multiplex_active(False)

    asyncio.run(discovery._connect_server("demo", {"command": "true"}))

    assert spawn_env["env"][TOKEN_NAME] == "process-env-value"
    assert current_secret_scope() is None


def test_unscoped_discover_interpolates_header_refs_under_the_owners_scope(profile_home, monkeypatch):
    """Red on base: ``_load_mcp_config`` interpolates ``${VAR}`` refs BEFORE any connect and swallows
    the UnscopedSecretError into {}, so discover for a routed profile registered ZERO servers
    (stdio siblings included). The header carries the OWNING profile's value, never the launch env's."""
    monkeypatch.setenv(TOKEN_NAME, "launch-env-value")
    (profile_home / "config.yaml").write_text(
        "mcp_servers:\n"
        "  httpsrv:\n    url: https://example.invalid/mcp\n"
        f"    headers:\n      Authorization: 'Bearer ${{{TOKEN_NAME}}}'\n"
        "  stdiosrv:\n    command: 'true'\n", encoding="utf-8")
    handed_over = {}
    monkeypatch.setattr(discovery, "register_mcp_servers", lambda servers: handed_over.update(servers) or [])
    monkeypatch.setattr("tools.mcp_tool._ensure_mcp_sdk", lambda: True)
    assert current_secret_scope() is None

    discovery.discover_mcp_tools()

    assert set(handed_over) == {"httpsrv", "stdiosrv"}
    assert handed_over["httpsrv"]["headers"]["Authorization"] == f"Bearer {TOKEN_VALUE}"
    assert current_secret_scope() is None


def test_connect_scope_install_failure_releases_the_discovery_claim(monkeypatch, spawn_env):
    """Red on base: the scope install sat between the claim ``set(None)`` and the try, so a raise from
    hydration skipped ``_connect_server_claim.reset`` and the caller's claim stayed cleared."""
    async def _boom():
        raise RuntimeError("hydration failed")
    monkeypatch.setattr(discovery, "_install_owner_secret_scope", _boom)
    claim = lambda server: None  # noqa: E731

    async def _run():
        token = discovery._core._connect_server_claim.set(claim)
        try:
            with pytest.raises(RuntimeError, match="hydration failed"):
                await discovery._connect_server("demo", {"command": "true"})
            assert discovery._core._connect_server_claim.get() is claim
        finally:
            discovery._core._connect_server_claim.reset(token)

    asyncio.run(_run())


REMOTE = {"url": "https://example.invalid/mcp", "headers": {"Authorization": f"Bearer ${{{TOKEN_NAME}}}"}}


@contextmanager
def _boot_scope(home):
    """The gateway's boot-time ``_profile_runtime_scope`` shape: home override plus a secret scope
    SNAPSHOT built now — before the profile's secret source may have answered."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    home_token = set_hermes_home_override(str(home))
    token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(token)
        reset_hermes_home_override(home_token)


def _connect_under_boot_scope(home, config):
    async def _run():
        with _boot_scope(home):
            await discovery._connect_server("demo", dict(config, headers=dict(config["headers"])))
    asyncio.run(_run())


def test_connect_renders_remote_headers_under_the_owners_fresh_scope(tmp_path, spawn_env, monkeypatch):
    """Red on base (#119092): a caller-bound scope was trusted as-is, so a header rendered under the
    boot-time snapshot — taken before B's secret source answered — went out as the literal
    ``Bearer ${VAR}`` (401) instead of failing closed, and never healed. A→B→A: each home's own
    value, never the other's."""
    monkeypatch.setitem(env_loader._SECRET_SOURCES, TOKEN_NAME, "command")
    monkeypatch.setenv(TOKEN_NAME, "launch-env-value")
    home_a, home_b = tmp_path / "s6probe-a", tmp_path / "s6probe-b"
    for home in (home_a, home_b):
        home.mkdir()
    (home_a / ".env").write_text(f"{TOKEN_NAME}=value-a\n", encoding="utf-8")
    set_multiplex_active(True)
    try:
        _connect_under_boot_scope(home_a, REMOTE)
        assert spawn_env["config"]["headers"]["Authorization"] == "Bearer value-a"

        with pytest.raises(ValueError, match=f"'demo'.*{TOKEN_NAME}"):  # B: source not hydrated yet
            _connect_under_boot_scope(home_b, REMOTE)
        (home_b / ".env").write_text(f"{TOKEN_NAME}=value-b\n", encoding="utf-8")  # the source answers
        _connect_under_boot_scope(home_b, REMOTE)
        assert spawn_env["config"]["headers"]["Authorization"] == "Bearer value-b"

        _connect_under_boot_scope(home_a, REMOTE)
        assert spawn_env["config"]["headers"]["Authorization"] == "Bearer value-a"
        assert current_secret_scope() is None
    finally:
        set_multiplex_active(False)


def test_launch_profile_env_only_credential_survives_the_owner_rebuild(spawn_env, monkeypatch):
    """Red on the salvage: the owner rebuild used ``build_profile_secret_scope`` (files only) for the
    LAUNCH profile too, so a credential that lives only in the launch env (systemd ``Environment=``,
    ``op run``, Compose) — present in the ``launch_secret_scope`` mapping the caller bound — vanished,
    the header stayed the literal ``${VAR}`` and the fail-closed check parked a server that worked."""
    import os
    from pathlib import Path
    from tools import mcp_tool_config as _config
    from tui_gateway import launch_profile_policy
    launch_home = Path(os.environ["HERMES_HOME"])  # conftest's per-test process home
    (launch_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv(TOKEN_NAME, "tok-from-systemd")
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    launch_profile_policy.activate_multi_profile_hosting()  # freeze the launch env, fail closed, pin the home

    async def _run():
        with launch_profile_policy.launch_profile_runtime_scope(launch_home):
            await discovery._connect_server("demo", dict(REMOTE, headers=dict(REMOTE["headers"])))
            with discovery._owner_secret_scope():  # a later rebuild (reconnect / reconcile) under the same door
                return _config._interpolate_env_vars(dict(REMOTE["headers"]))

    try:
        rebuilt = asyncio.run(_run())
    finally:
        set_multiplex_active(False)
    assert spawn_env["config"]["headers"]["Authorization"] == "Bearer tok-from-systemd"
    assert rebuilt["Authorization"] == "Bearer tok-from-systemd"


def test_reconnect_rerenders_remote_headers_under_the_owners_fresh_scope(tmp_path, monkeypatch):
    """Red on base (#119092): the run task re-read config.yaml on every rebuild but rendered it under
    its copied connect-time scope snapshot, so a parked server retried the literal ``${VAR}`` header
    forever after the owner's secret source came up."""
    from tools.mcp_tool import MCPServerTask
    monkeypatch.setitem(env_loader._SECRET_SOURCES, TOKEN_NAME, "command")
    home_b = tmp_path / "s6probe-b"
    home_b.mkdir()
    (home_b / "config.yaml").write_text(
        "mcp_servers:\n  demo:\n    url: https://example.invalid/mcp\n"
        f"    headers:\n      Authorization: 'Bearer ${{{TOKEN_NAME}}}'\n", encoding="utf-8")
    seen: list = []

    class _Task(MCPServerTask):
        async def _prepare_run(self, config):
            self._config = config
            self._auth_type = ""
            return True

        async def _run_http(self, config):
            seen.append(config["headers"]["Authorization"])
            self._ready.set()
            self.session = object()
            self._session_proven = True
            if len(seen) == 1:
                (home_b / ".env").write_text(f"{TOKEN_NAME}=value-b\n", encoding="utf-8")  # the source answers
                return "reconnect"
            self._shutdown_event.set()
            return "shutdown"

    async def _scenario():
        with _boot_scope(home_b):  # snapshot taken before B's source answered
            await asyncio.wait_for(_Task("demo").run(dict(REMOTE, headers=dict(REMOTE["headers"]))), timeout=10)

    set_multiplex_active(True)
    try:
        asyncio.run(_scenario())
    finally:
        set_multiplex_active(False)
    assert seen == [f"Bearer ${{{TOKEN_NAME}}}", "Bearer value-b"]
