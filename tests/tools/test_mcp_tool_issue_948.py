import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE
from tools.mcp_tool_errors import _format_connect_error
from tools.mcp_tool_config import _resolve_stdio_command
from tools.mcp_tool_config import _which_with_config_pathext

# Ensure the mcp module symbols exist for patching even when the SDK isn't installed
if not _MCP_AVAILABLE:
    import tools.mcp_tool as _mcp_mod
    if not hasattr(_mcp_mod, "StdioServerParameters"):
        _mcp_mod.StdioServerParameters = MagicMock
    if not hasattr(_mcp_mod, "stdio_client"):
        _mcp_mod.stdio_client = MagicMock
    if not hasattr(_mcp_mod, "ClientSession"):
        _mcp_mod.ClientSession = MagicMock


@pytest.mark.platforms("posix")
def test_resolve_stdio_command_keeps_the_child_path_order(tmp_path):
    """A command found later on the child's PATH must not pull its directory ahead of
    earlier entries: the child's other bare lookups (node, python3, git) follow the PATH
    order the user, or pm.activate(), chose. Hoisting it (#124792) handed a brew
    command's children brew's node/python3/git instead of the pinned store copies."""
    first, middle, later = tmp_path / "first", tmp_path / "middle", tmp_path / "later"
    for directory in (first, middle, later):
        directory.mkdir()
    tool = later / "mytool"
    tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool.chmod(0o755)
    path = os.pathsep.join([str(first), str(middle), str(later)])

    command, env = _resolve_stdio_command("mytool", {"PATH": path})

    assert command == str(tool)
    assert env["PATH"] == path


def test_resolve_stdio_command_skips_unknown_commands():
    """Bare command names outside the npx/npm/node/uv/uvx launcher set must NOT
    be matched against the fallback paths — that would rewrite ``command:
    my-tool`` into a coincidentally-named file at /opt/homebrew/bin/my-tool."""
    with patch("tools.mcp_tool_config.shutil.which", return_value=None), \
         patch("tools.mcp_tool_config.os.path.isfile", return_value=True), \
         patch("tools.mcp_tool_config.os.access", return_value=True):
        command, _env = _resolve_stdio_command("my-tool", {"PATH": "/usr/bin:/bin"})

    assert command == "my-tool"


def test_resolve_stdio_command_absent_path_is_a_miss(tmp_path, monkeypatch):
    """A server env without PATH must not resolve commands against the PARENT's PATH:
    the child would be spawned without it and the lookup would pass on an env the
    child never sees."""
    parent_bin = tmp_path / "parent-bin"
    parent_bin.mkdir()
    server_tool = parent_bin / "some-mcp-server"
    server_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(parent_bin))

    command, _env = _resolve_stdio_command("some-mcp-server", {"OTHER": "1"})

    # absent child PATH: honest miss, not an ambient hit
    assert command == "some-mcp-server"


def test_resolve_stdio_command_empty_path_is_a_miss(monkeypatch, tmp_path):
    """An explicitly empty child PATH keeps its cwd-only meaning (never the parent's PATH):
    ``which`` sees ``[""]`` -> cwd. The binary lives only in the parent's PATH dir, so the
    lookup must miss rather than silently inheriting the parent's directories."""
    parent_bin = tmp_path / "parent-bin"
    parent_bin.mkdir()
    server_tool = parent_bin / "other-mcp-server"
    server_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server_tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(parent_bin))

    command, _env = _resolve_stdio_command("other-mcp-server", {"PATH": ""})

    assert command == "other-mcp-server"  # cwd-only lookup: no ambient fallback


def test_config_pathext_lookup_never_touches_parent_environ(tmp_path, monkeypatch):
    """Resolving under a configured PATHEXT must not mutate the parent's ``os.environ``:
    a multiplexed gateway resolves servers for several profiles from one process, and
    any thread reading PATHEXT (or inheriting env for its own subprocess) inside the
    lookup window would otherwise see this server's per-profile value."""
    server_dir = tmp_path / "bin"
    server_dir.mkdir()
    (server_dir / "server.cmd").write_text("@echo off\r\n", encoding="utf-8")
    (server_dir / "server.cmd").chmod(0o755)
    monkeypatch.delenv("PATHEXT", raising=False)
    monkeypatch.setenv("PATH", str(server_dir))
    seen = {}

    import tools.mcp_tool_config as _cfg

    def _spy(cmd, path=None):
        seen["PATHEXT"] = os.environ.get("PATHEXT")
        raise AssertionError("shutil.which must not be the lookup engine here")

    with patch.object(_cfg.shutil, "which", side_effect=_spy):
        cfg_env = {"PATHEXT": ".cmd;.exe"}
        hit = _which_with_config_pathext("server", str(server_dir), cfg_env)

    assert hit == str(server_dir / "server.cmd")
    assert "PATHEXT" not in os.environ  # not written, not left behind
    assert seen == {}  # and never consulted mid-lookup either


# ---------------------------------------------------------------------------
# #29184: OSV malware preflight must not block the asyncio event loop, and a
# stalled check must time out fail-open rather than freezing MCP startup.
# ---------------------------------------------------------------------------


def _stdio_mocks():
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))
    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)
    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_stdio_cm, mock_session_cm


def test_run_stdio_malware_check_does_not_block_event_loop():
    """The blocking OSV check runs off the loop (asyncio.to_thread), so a
    concurrent coroutine keeps making progress while it runs."""
    import time
    mock_stdio_cm, mock_session_cm = _stdio_mocks()

    def slow_check(_command, _args):
        time.sleep(0.3)  # simulate a slow OSV HTTPS call
        return None

    ticks = {"n": 0}

    async def _ticker():
        # If the loop were blocked, these ticks would not advance during the
        # 0.3s check.
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks["n"] += 1

    async def _test():
        with patch("tools.osv_check.check_package_for_malware", side_effect=slow_check), \
             patch("tools.mcp_tool_config._managed_launcher", return_value=None), \
             patch("tools.mcp_tool.StdioServerParameters"), \
             patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm), \
             patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm):
            server = MCPServerTask("srv")
            ticker = asyncio.create_task(_ticker())
            await server.start({"command": "npx", "args": ["-y", "pkg"]})
            ticks_during = ticks["n"]
            await ticker
            await server.shutdown()
        # The loop kept ticking DURING the 0.3s blocking check -> not blocked.
        assert ticks_during >= 3, f"event loop appeared blocked (ticks={ticks_during})"

    asyncio.run(_test())


def test_run_stdio_malware_check_times_out_fail_open():
    """A check that hangs past the timeout must NOT freeze startup: it times
    out, logs, and proceeds (fail-open) so the server still starts."""
    import time
    mock_stdio_cm, mock_session_cm = _stdio_mocks()

    def hung_check(_command, _args):
        time.sleep(0.5)  # outlasts the 0.2s timeout 2.5x; short enough not to stall teardown
        return "MALWARE"  # would block startup if awaited to completion

    async def _test():
        with patch("tools.osv_check.check_package_for_malware", side_effect=hung_check), \
             patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.2), \
             patch("tools.mcp_tool_config._managed_launcher", return_value=None), \
             patch("tools.mcp_tool.StdioServerParameters"), \
             patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm), \
             patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm):
            server = MCPServerTask("srv")
            start = time.monotonic()
            await server.start({"command": "npx", "args": ["-y", "pkg"]})
            elapsed = time.monotonic() - start
            await server.shutdown()
        # Returned shortly after the 0.2s timeout (fail-open), not the 0.5s hang.
        assert elapsed < 1.0, f"startup did not fail-open promptly ({elapsed:.1f}s)"

    asyncio.run(_test())


def _toolchain_bin(root, *names):
    root.mkdir(parents=True)
    for name in names:
        for spelling in (name, name + ".cmd"):
            launcher = root / spelling
            launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            launcher.chmod(0o755)
    return root


def _pm_ships(monkeypatch, *, node_dirs=(), uv=None):
    import hermes_constants
    import pm

    monkeypatch.setattr(pm, "ensure", lambda name, **kw: None)
    monkeypatch.setattr(hermes_constants, "with_hermes_node_path",
                        lambda env: {**env, "PATH": os.pathsep.join(map(str, node_dirs))})
    monkeypatch.setattr(pm, "uv_launcher", lambda name: uv.with_name(name) if uv else None)


def test_bare_node_launchers_resolve_pm_node_ahead_of_the_users(tmp_path, monkeypatch):
    """The packaged-toolchain rule: a bare ``npx`` runs Hermes's PM npx, with PM's npm and node
    dirs first on the child PATH (npx's ``env node``), even when the user's Node sorts first."""
    user_bin = _toolchain_bin(tmp_path / "user-node", "npx", "node")
    npm_bin = _toolchain_bin(tmp_path / "store" / "npm" / "bin", "npx", "npm")
    node_bin = _toolchain_bin(tmp_path / "store" / "node" / "bin", "node")
    _pm_ships(monkeypatch, node_dirs=(npm_bin, node_bin))

    command, env = _resolve_stdio_command("npx", {"PATH": os.pathsep.join([str(user_bin), "/usr/bin"])})

    assert os.path.dirname(command) == str(npm_bin)
    assert env["PATH"].split(os.pathsep) == [str(npm_bin), str(node_bin), str(user_bin), "/usr/bin"]


def test_bare_uvx_resolves_pm_uv_and_an_absolute_command_stays_the_users(tmp_path, monkeypatch):
    user_bin = _toolchain_bin(tmp_path / "user" / ".local" / "bin", "uvx")
    uv_dir = _toolchain_bin(tmp_path / "store" / "uv-0.1", "uv", "uvx")
    _pm_ships(monkeypatch, uv=uv_dir / "uv")

    command, env = _resolve_stdio_command("uvx", {"PATH": str(user_bin)})
    assert os.path.dirname(command) == str(uv_dir)
    assert env["PATH"].split(os.pathsep)[0] == str(uv_dir)

    explicit = str(user_bin / "uvx")
    command, _env = _resolve_stdio_command(explicit, {"PATH": "/usr/bin"})
    assert command == explicit
