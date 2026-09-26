import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE
from tools.mcp_tool_errors import _format_connect_error
from tools.mcp_tool_config import _node_fallback, _resolve_stdio_command
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


def test_resolve_stdio_command_falls_back_to_hermes_node_bin(tmp_path):
    node_bin = tmp_path / "node" / "bin"
    node_bin.mkdir(parents=True)
    npx_path = node_bin / "npx"
    npx_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    npx_path.chmod(0o755)

    with patch("tools.mcp_tool_config.shutil.which", return_value=None), \
         patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}, clear=False):
        command, env = _resolve_stdio_command("npx", {"PATH": "/usr/bin"})

    assert command == str(npx_path)
    assert env["PATH"].split(os.pathsep)[0] == str(node_bin)


def test_windows_managed_node_root_prefers_cmd_launchers(tmp_path):
    """Managed Windows Node lives directly in ``<HERMES_HOME>/node`` (no ``bin``) as ``npx.cmd`` /
    ``npm.cmd`` / ``node.exe``; a bare ``command: npx`` must resolve to those launchers (#111937).
    The extensionless POSIX sibling is a shell script Windows cannot spawn, so it must never win."""
    node_root = tmp_path / "node"
    node_root.mkdir()
    for name in ("npx", "npm", "npx.cmd", "npm.cmd", "node.exe"):
        launcher = node_root / name
        launcher.write_text("@echo off\r\n", encoding="utf-8")
        launcher.chmod(0o755)

    with patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}, clear=False):
        assert _node_fallback("npx", windows=True) == str(node_root / "npx.cmd")
        assert _node_fallback("npm", windows=True) == str(node_root / "npm.cmd")
        assert _node_fallback("node", windows=True) == str(node_root / "node.exe")


def test_node_fallback_uses_active_profile_home(tmp_path, monkeypatch):
    """The managed-Node lookup follows ``get_hermes_home()`` (context override), not raw ``HERMES_HOME``:
    a multiplexed profile whose home differs from the launch env must find ITS managed Node."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profile"
    npx_path = profile_home / "node" / "bin" / "npx"
    npx_path.parent.mkdir(parents=True)
    npx_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    npx_path.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-home"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))  # keep a real ~/.local/bin/npx out of the picture

    token = set_hermes_home_override(profile_home)
    try:
        with patch("tools.mcp_tool_config.shutil.which", return_value=None):
            command, _env = _resolve_stdio_command("npx", {"PATH": "/usr/bin"})
    finally:
        reset_hermes_home_override(token)
    assert command == str(npx_path)


def test_resolve_stdio_command_falls_back_to_usr_local_bin():
    """When ``npx`` isn't on the filtered PATH and isn't under ``$HERMES_HOME/node/bin``
    or ``~/.local/bin``, the resolver should still locate it at ``/usr/local/bin/npx``.

    This is the canonical install location for Node on Linux from-source builds,
    the upstream ``node:bookworm-slim`` image (which the Hermes Docker image
    copies ``node + npm + corepack`` from since #4977), and macOS Homebrew on
    Intel. Without this candidate, MCP servers run with an ``env.PATH`` that
    omits ``/usr/local/bin`` (common when users hand-author PATH for sandboxing)
    fail with ENOENT at ``execvp``.
    """
    target = os.path.join(os.sep, "usr", "local", "bin", "npx")

    # Pretend ONLY the /usr/local/bin/npx candidate exists and is executable —
    # the other candidates ($HERMES_HOME/node/bin/npx and ~/.local/bin/npx)
    # should fail isfile() and the resolver must fall through to /usr/local/bin.
    def _fake_isfile(path):
        return path == target

    def _fake_access(path, _mode):
        return path == target

    with patch("tools.mcp_tool_config.shutil.which", return_value=None), \
         patch("tools.mcp_tool.os.path.isfile", side_effect=_fake_isfile), \
         patch("tools.mcp_tool.os.access", side_effect=_fake_access):
        command, env = _resolve_stdio_command("npx", {"PATH": "/opt/data/bin:/usr/bin:/bin"})

    assert command == target
    # /usr/local/bin must be prepended so npx's shebang (`/usr/bin/env node`)
    # can find node in the same directory.
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(target)


# ---------------------------------------------------------------------------
# #37589: Desktop/launchd processes inherit a minimal PATH on macOS that does
# not include ~/.local/bin, /opt/homebrew/bin, or /usr/local/bin. The resolver
# must locate bare uv/uvx (the dominant Python MCP-server runtime) under those
# locations instead of failing with ENOENT at execvp.
# ---------------------------------------------------------------------------


def test_resolve_stdio_command_finds_uvx_in_user_local_bin(tmp_path, monkeypatch):
    """uv's official installer drops uv/uvx at ``~/.local/bin/uvx`` on macOS and
    Linux. The resolver must pick it up when the GUI PATH doesn't include that
    directory (#37589)."""
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    uvx_path = local_bin / "uvx"
    uvx_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uvx_path.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HOME", str(tmp_path))

    with patch("tools.mcp_tool_config.shutil.which", return_value=None):
        command, env = _resolve_stdio_command("uvx", {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"})

    assert command == str(uvx_path)
    # The resolver prepended the chosen bin so uvx's sibling `uv` and its
    # shebang-resolved children resolve in the same directory.
    assert env["PATH"].split(os.pathsep)[0] == str(local_bin)


def test_resolve_stdio_command_uv_fallback_order(tmp_path, monkeypatch):
    """Bare uv/uvx probe the well-known install dirs in uv's install order:
    managed ``<HERMES_HOME>/bin`` first, then ``~/.local/bin``, then Homebrew
    (Apple Silicon, then Intel/from-source)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setattr("tools.mcp_tool_config.os.path.expanduser", lambda p: p.replace("~", str(tmp_path / "user")) if p.startswith("~") else p)

    candidates = [
        os.path.join(str(tmp_path / "hermes"), "bin", "uvx"),
        os.path.join(str(tmp_path / "user"), ".local", "bin", "uvx"),
        os.path.join(os.sep, "opt", "homebrew", "bin", "uvx"),
        os.path.join(os.sep, "usr", "local", "bin", "uvx"),
    ]
    seen = []

    def _fake_access(path, mode):
        assert mode == os.X_OK
        seen.append(path)
        return path == candidates[-1]  # only /usr/local/bin exists

    with patch("tools.mcp_tool_config.shutil.which", return_value=None), \
         patch("tools.mcp_tool_config.os.path.isfile", return_value=True), \
         patch("tools.mcp_tool_config.os.access", side_effect=_fake_access):
        command, _env = _resolve_stdio_command("uvx", {"PATH": "/usr/bin"})

    assert seen == candidates  # every dir probed, in install order
    assert command == candidates[-1]


def test_resolve_stdio_command_uvx_unchanged_when_already_on_path():
    """A shutil.which hit still takes precedence — don't double-resolve a working
    bare command on the child's own PATH into something else."""
    resolved_path = "/some/custom/bin/uvx"
    with patch("tools.mcp_tool_config.shutil.which", return_value=resolved_path):
        command, _env = _resolve_stdio_command("uvx", {"PATH": "/usr/bin"})

    assert command == resolved_path


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
    child never sees. Bare ``node`` still reaches the explicit well-known dirs."""
    parent_bin = tmp_path / "parent-bin"
    parent_bin.mkdir()
    server_tool = parent_bin / "some-mcp-server"
    server_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    server_tool.chmod(0o755)
    node_tool = tmp_path / "node" / "bin" / "node"
    node_tool.parent.mkdir(parents=True)
    node_tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    node_tool.chmod(0o755)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # the parent PATH contains BOTH names: an ambient hit would resolve either
    monkeypatch.setenv("PATH", str(parent_bin))

    command, _env = _resolve_stdio_command("some-mcp-server", {"OTHER": "1"})

    # absent child PATH: honest miss, not an ambient hit
    assert command == "some-mcp-server"

    with patch.dict("os.environ", {"PATH": str(parent_bin)}):
        command, _env = _resolve_stdio_command("node", {"OTHER": "1"})
    assert command == str(node_tool)  # the explicit Node fallback dirs stay reachable


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
