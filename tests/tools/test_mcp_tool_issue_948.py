import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
             patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
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
             patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
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


@pytest.mark.windows_only
def test_run_stdio_spawns_the_cached_windows_launcher(tmp_path, monkeypatch):
    """Exercise .npmrc cache resolution through the actual Windows stdio process boundary."""
    pytest.importorskip("mcp")

    monkeypatch.delenv("npm_config_cache", raising=False)
    monkeypatch.delenv("NPM_CONFIG_CACHE", raising=False)
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    child_cwd = tmp_path / "child-cwd"
    child_cwd.mkdir()
    (child_cwd / ".npmrc").write_text("cache=configured-npm-cache\n", encoding="utf-8")
    cache_root = child_cwd / "configured-npm-cache"
    entry = cache_root / "_npx" / "configured" / "node_modules"
    package = entry / "mcp-linear"
    package.mkdir(parents=True)
    (entry.parent / "package.json").write_text(
        '{"dependencies":{"mcp-linear":"^1.0.0"}}', encoding="utf-8")
    (package / "package.json").write_text(
        '{"bin":{"mcp-linear":"dist/index.js"}}', encoding="utf-8")
    bin_path = entry / ".bin" / "mcp-linear.cmd"
    bin_path.parent.mkdir()
    fixture_server = tmp_path / "fixture_mcp_server.py"
    fixture_server.write_text(
        """
import json
import os
import sys


def reply(request_id, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    request = json.loads(line)
    request_id = request.get("id")
    if request_id is None:
        continue
    if request.get("method") == "initialize":
        with open(os.environ["HERMES_EVENT_MARKER"], "a", encoding="utf-8") as event:
            event.write("server-initialize\\n")
        with open(os.environ["HERMES_FIXTURE_MARKER"], "w", encoding="utf-8") as marker:
            json.dump({
                "argv": sys.argv[1:],
                "cache": os.environ.get("NPM_CONFIG_CACHE"),
                "cwd": os.getcwd(),
                "hermes_home": os.environ.get("HERMES_HOME"),
            }, marker)
        reply(request_id, {
            "protocolVersion": request["params"]["protocolVersion"],
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fixture", "version": "1.0.0"},
        })
    elif request.get("method") == "tools/list":
        reply(request_id, {"tools": [{
            "name": "fixture_tool",
            "description": "fixture tool",
            "inputSchema": {"type": "object", "properties": {}},
        }]})
    elif request.get("method") == "ping":
        reply(request_id, {})
""".lstrip(),
        encoding="utf-8",
    )
    cache_marker = tmp_path / "cache-launch.txt"
    event_marker = tmp_path / "launch-events.txt"
    fixture_marker = tmp_path / "fixture-launch.json"
    bin_path.write_text(
        "@echo off\r\n"
        "> \"%HERMES_CACHE_MARKER%\" echo %*\r\n"
        ">> \"%HERMES_EVENT_MARKER%\" echo cached-launch\r\n"
        "\"%HERMES_TEST_PYTHON%\" -u \"%HERMES_FIXTURE_SERVER%\" %*\r\n",
        encoding="utf-8",
    )

    npx_dir = tmp_path / "npx-bin"
    npx_dir.mkdir()
    (npx_dir / "npx.cmd").write_text(
        "@echo off\r\n"
        ">> \"%HERMES_EVENT_MARKER%\" echo npx-fallback\r\n"
        "exit /b 1\r\n",
        encoding="utf-8",
    )
    npm_config_server = tmp_path / "fixture_npm_config.py"
    npm_config_server.write_text(
        """
import os
import re
import sys
from pathlib import Path


npmrc = (Path.cwd() / ".npmrc").read_text(encoding="utf-8")
cache = re.search(r"^\\s*cache\\s*=\\s*(.+?)\\s*$", npmrc, flags=re.MULTILINE)
if cache is None:
    raise SystemExit(2)
Path(os.environ["HERMES_NPM_CONFIG_MARKER"]).write_text(os.getcwd(), encoding="utf-8")
with open(os.environ["HERMES_EVENT_MARKER"], "a", encoding="utf-8") as marker:
    marker.write("npm-config\\n")
sys.stdout.write(cache.group(1) + "\\n")
""".lstrip(),
        encoding="utf-8",
    )
    (npx_dir / "node.cmd").write_text(
        "@echo off\r\n"
        "\"%HERMES_TEST_PYTHON%\" -u \"%HERMES_NPM_CONFIG_SERVER%\" %*\r\n",
        encoding="utf-8",
    )
    npm_cli = npx_dir / "node_modules" / "npm" / "bin" / "npm-cli.js"
    npm_cli.parent.mkdir(parents=True)
    npm_cli.write_text("// paired npm fixture\n", encoding="utf-8")
    npm_config_marker = tmp_path / "npm-config.txt"

    async def _test():
        server = MCPServerTask("windows-cached-launcher-fixture")
        try:
            await server.start({
                "command": "npx",
                "args": ["-y", "mcp-linear", "--fixture-arg"],
                "connect_timeout": 5,
                "env": {
                    "HERMES_CACHE_MARKER": str(cache_marker),
                    "HERMES_EVENT_MARKER": str(event_marker),
                    "HERMES_FIXTURE_MARKER": str(fixture_marker),
                    "HERMES_FIXTURE_SERVER": str(fixture_server),
                    "HERMES_HOME": str(hermes_home),
                    "HERMES_NPM_CONFIG_SERVER": str(npm_config_server),
                    "HERMES_NPM_CONFIG_MARKER": str(npm_config_marker),
                    "HERMES_TEST_PYTHON": sys.executable,
                    "PATHEXT": ".COM;.EXE;.BAT;.CMD",
                    "PATH": str(npx_dir) + os.pathsep + os.environ.get("PATH", ""),
                },
                "cwd": str(child_cwd),
            })
            launched = json.loads(fixture_marker.read_text(encoding="utf-8"))
            assert cache_marker.read_text(encoding="utf-8").strip() == "--fixture-arg"
            assert launched == {
                "argv": ["--fixture-arg"],
                "cache": str(cache_root),
                "cwd": str(child_cwd),
                "hermes_home": str(hermes_home),
            }
            assert npm_config_marker.read_text(encoding="utf-8") == str(child_cwd)
            assert event_marker.read_text(encoding="utf-8").splitlines() == [
                "npm-config", "cached-launch", "server-initialize",
            ]
            assert [tool.name for tool in server._tools] == ["fixture_tool"]
        finally:
            await server.shutdown()

    with patch("tools.osv_check.check_package_for_malware", return_value=None):
        asyncio.run(asyncio.wait_for(_test(), timeout=15))
