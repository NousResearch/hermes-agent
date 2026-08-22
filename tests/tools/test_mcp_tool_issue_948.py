import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools.mcp_tool import MCPServerTask, _format_connect_error, _resolve_stdio_command, _MCP_AVAILABLE

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

    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch.dict("os.environ", {"HERMES_HOME": str(tmp_path)}, clear=False):
        command, env = _resolve_stdio_command("npx", {"PATH": "/usr/bin"})

    assert command == str(npx_path)
    assert env["PATH"].split(os.pathsep)[0] == str(node_bin)


def _resolver_home_env(user_home, hermes_home):
    return {
        "HOME": str(user_home),
        "USERPROFILE": str(user_home),
        "HERMES_HOME": str(hermes_home),
    }


def _launcher_filename(command_name):
    if sys.platform == "win32" and command_name in {"uv", "uvx"}:
        return f"{command_name}.exe"
    return command_name


def test_resolve_stdio_command_falls_back_to_user_uvx_bin(tmp_path):
    user_home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    user_bin = user_home / ".local" / "bin"
    user_bin.mkdir(parents=True)
    uvx_path = user_bin / _launcher_filename("uvx")
    uvx_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uvx_path.chmod(0o755)

    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch.dict("os.environ", _resolver_home_env(user_home, hermes_home), clear=False):
        command, env = _resolve_stdio_command("uvx", {"PATH": "/usr/bin"})

    assert command == str(uvx_path)
    assert env["PATH"].split(os.pathsep)[0] == str(user_bin)


def test_resolve_stdio_command_uses_scoped_hermes_home(tmp_path):
    process_home = tmp_path / "process-home"
    scoped_home = tmp_path / "profile-home"
    scoped_bin = scoped_home / "bin"
    scoped_bin.mkdir(parents=True)
    uvx_path = scoped_bin / _launcher_filename("uvx")
    uvx_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uvx_path.chmod(0o755)

    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch.dict("os.environ", {"HERMES_HOME": str(process_home)}, clear=False):
        token = set_hermes_home_override(scoped_home)
        try:
            command, env = _resolve_stdio_command("uvx", {"PATH": "/usr/bin"})
        finally:
            reset_hermes_home_override(token)

    assert command == str(uvx_path)
    assert env["PATH"].split(os.pathsep)[0] == str(scoped_bin)


@pytest.mark.parametrize(
    ("command_name", "selected_index"),
    [
        pytest.param("uv", 0, id="uv-managed"),
        pytest.param("uvx", 0, id="uvx-managed"),
        pytest.param("uv", 1, id="uv-user-local"),
        pytest.param("uvx", 1, id="uvx-user-local"),
        pytest.param("uv", 2, id="uv-apple-homebrew"),
        pytest.param("uvx", 2, id="uvx-apple-homebrew"),
        pytest.param("uv", 3, id="uv-usr-local"),
        pytest.param("uvx", 3, id="uvx-usr-local"),
    ],
)
def test_resolve_stdio_command_uv_fallback_order(tmp_path, command_name, selected_index):
    user_home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    candidate_name = _launcher_filename(command_name)
    expected_candidates = [
        os.path.join(hermes_home, "bin", candidate_name),
        os.path.join(user_home, ".local", "bin", candidate_name),
        os.path.join(os.sep, "opt", "homebrew", "bin", candidate_name),
        os.path.join(os.sep, "usr", "local", "bin", candidate_name),
    ]
    seen_candidates = []

    def _fake_access(path, mode):
        assert mode == os.X_OK
        seen_candidates.append(path)
        return path == expected_candidates[selected_index]

    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch("tools.mcp_tool.os.path.isfile", return_value=True), \
         patch("tools.mcp_tool.os.access", side_effect=_fake_access), \
         patch.dict("os.environ", _resolver_home_env(user_home, hermes_home), clear=False):
        command, env = _resolve_stdio_command(command_name, {"PATH": "/usr/bin"})

    assert seen_candidates == expected_candidates[:selected_index + 1]
    assert command == expected_candidates[selected_index]
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(command)


def test_resolve_stdio_command_uses_windows_uvx_executable(tmp_path):
    user_home = tmp_path / "home"
    hermes_home = tmp_path / "hermes"
    expected = os.path.join(hermes_home, "bin", "uvx.exe")

    with patch("tools.mcp_tool.sys.platform", "win32"), \
         patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch("tools.mcp_tool.os.path.isfile", return_value=True), \
         patch("tools.mcp_tool.os.access", return_value=True), \
         patch.dict("os.environ", _resolver_home_env(user_home, hermes_home), clear=False):
        command, env = _resolve_stdio_command("uvx", {"PATH": r"C:\Windows\System32"})

    assert command == expected
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(expected)


def test_resolve_stdio_command_prefers_filtered_path_hit():
    target = os.path.join(os.sep, "custom", "bin", "uvx")

    with patch("tools.mcp_tool.shutil.which", return_value=target) as mock_which, \
         patch("tools.mcp_tool.os.path.isfile") as mock_isfile:
        command, env = _resolve_stdio_command("uvx", {"PATH": "/custom/bin"})

    assert command == target
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(target)
    mock_which.assert_called_once_with("uvx", path="/custom/bin")
    mock_isfile.assert_not_called()


def test_resolve_stdio_command_does_not_fallback_for_unknown_command():
    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch("tools.mcp_tool.os.path.isfile") as mock_isfile:
        command, env = _resolve_stdio_command(
            "custom-mcp-launcher", {"PATH": "/usr/bin"}
        )

    assert command == "custom-mcp-launcher"
    assert env["PATH"] == "/usr/bin"
    mock_isfile.assert_not_called()


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

    with patch("tools.mcp_tool.shutil.which", return_value=None), \
         patch("tools.mcp_tool.os.path.isfile", side_effect=_fake_isfile), \
         patch("tools.mcp_tool.os.access", side_effect=_fake_access):
        command, env = _resolve_stdio_command("npx", {"PATH": "/opt/data/bin:/usr/bin:/bin"})

    assert command == target
    # /usr/local/bin must be prepended so npx's shebang (`/usr/bin/env node`)
    # can find node in the same directory.
    assert env["PATH"].split(os.pathsep)[0] == os.path.dirname(target)


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
