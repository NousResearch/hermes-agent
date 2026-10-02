import asyncio
import json
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
             patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
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
    import threading
    mock_stdio_cm, mock_session_cm = _stdio_mocks()

    check_started = threading.Event()
    release_check = threading.Event()
    check_finished = threading.Event()

    def hung_check(_command, _args):
        check_started.set()
        release_check.wait(10)  # explicit test release, not a scheduler-sensitive sleep
        check_finished.set()
        return "MALWARE"  # would block startup if awaited to completion

    async def _test():
        with patch("tools.osv_check.check_package_for_malware", side_effect=hung_check) as malware_check, \
             patch("tools.mcp_tool._OSV_MALWARE_CHECK_TIMEOUT_S", 0.2), \
             patch("tools.mcp_tool._effective_npx_cache_env", return_value=None), \
             patch("tools.mcp_tool_config._managed_launcher", return_value=("npx", [])), \
             patch("tools.mcp_tool.StdioServerParameters"), \
             patch("tools.mcp_tool.stdio_client", return_value=mock_stdio_cm), \
             patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm):
            server = MCPServerTask("srv")
            start = time.monotonic()
            try:
                await asyncio.wait_for(server.start({"command": "npx", "args": ["-y", "pkg"]}), timeout=2)
                elapsed = time.monotonic() - start
                assert check_started.is_set()
                assert not check_finished.is_set(), "startup waited for the blocked malware check"
                malware_check.assert_called_once_with("npx", ["-y", "pkg"])
            finally:
                release_check.set()
                await server.shutdown()
        # The 0.2s preflight timeout releases startup while the checker remains blocked.
        assert elapsed < 2.0, f"startup did not fail-open promptly ({elapsed:.1f}s)"

    asyncio.run(_test())


@pytest.mark.platforms('windows')
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
                "command": str(npx_dir / "npx.cmd"),
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
