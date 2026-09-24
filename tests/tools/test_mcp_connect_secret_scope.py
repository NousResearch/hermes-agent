"""MCP connect resolves credentials under the connection owner's profile secret scope.

Discovery at gateway startup runs outside any per-turn scope. Under multiplexing an unscoped
``get_secret`` fails closed, so the stdio child-env build raised ``UnscopedSecretError`` and the
server registered zero tools (#113746).
"""
import asyncio

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
