import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE
from tools.mcp_tool_errors import _format_connect_error
from tools.mcp_tool_common import _prepend_path
from tools.mcp_tool_config import _first_user_which_hit, _resolve_stdio_command
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




def _bare_exe(directory, name):
    """Platform-shaped fixture executable: Windows resolves bare names through PATHEXT
    and an extensionless file is not executable there, so the file carries `.exe` (the
    exec-bit chmod is skipped — os.access(X_OK) is an existence check on win32); POSIX
    keeps the bare name with 0o755."""
    if sys.platform == "win32":
        exe = directory / f"{name}.exe"
        exe.write_text("", encoding="utf-8")
        return exe
    exe = directory / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


def test_bare_python3_steps_past_the_managed_runtime_to_the_user_hit(tmp_path, monkeypatch):
    """Two PATH hits for a bare non-launcher command: the managed runtime's and the
    user's. The user's wins; the resolved env prepends the user's dir (helpers the
    server spawns resolve against the same interpreter). The user dir must sit OUTSIDE
    the hermes home — everything under it counts as managed."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home = tmp_path / "home"
    managed_bin = home / "bin"  # <HERMES_HOME>/bin — a managed dir by definition
    managed_bin.mkdir(parents=True)
    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    _bare_exe(managed_bin, "python3")
    user_exe = _bare_exe(user_bin, "python3")
    token = set_hermes_home_override(home)
    try:
        command, env = _resolve_stdio_command("python3", {
            "PATH": os.pathsep.join([str(managed_bin), str(user_bin)])})
    finally:
        reset_hermes_home_override(token)

    assert command == str(user_exe)
    # The user's dir is already on the child PATH, so it keeps its place: main's
    # invariant (test_resolve_stdio_command_keeps_the_child_path_order) is that a
    # resolved command never reorders the child's PATH.
    assert env["PATH"] == os.pathsep.join([str(managed_bin), str(user_bin)])


def test_bare_python3_keeps_the_managed_hit_when_the_user_has_none(tmp_path, monkeypatch):
    """A managed-only PATH keeps the managed hit: no user hit exists to prefer, and a
    resolved absolute path still beats an ENOENT at execvp. The trailing user dir is a
    real directory with no executables — not /usr/bin, which may genuinely carry one."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    home = tmp_path / "home"
    managed_bin = home / "bin"
    managed_bin.mkdir(parents=True)
    managed_exe = _bare_exe(managed_bin, "python3")
    empty_bin = tmp_path / "user-bin"
    empty_bin.mkdir()
    token = set_hermes_home_override(home)
    try:
        command, _env = _resolve_stdio_command("python3", {
            "PATH": os.pathsep.join([str(managed_bin), str(empty_bin)])})
    finally:
        reset_hermes_home_override(token)

    assert command == str(managed_exe)


def test_first_user_which_hit_follows_the_child_env_pathext(tmp_path):
    """Candidate extensions come from the child env's PATHEXT (``shutil.which`` reads the
    PARENT's), so a user install that only ships a ``.bat`` wrapper — the usual
    conda/npm shape — is a hit, not just the ``.cmd``/``.exe`` pair the npx cache layout
    guarantees. A command that already carries a PATHEXT suffix is matched as written."""
    user_bin = tmp_path / "user-bin"
    user_bin.mkdir()
    # srv.BAT spelled as PATHEXT spells it: a bare `srv` hits the PATHEXT-spelled join
    # (real Windows filesystems are case-insensitive; a case-sensitive host needs the
    # exact spelling to keep this branch observable). wrap.cmd keeps the lowercase,
    # as-written spelling its own assertion looks up.
    bat = user_bin / "srv.BAT"
    bat.write_text("@echo off\r\n", encoding="utf-8")
    bat.chmod(0o755)  # real exec-bit check on POSIX test hosts; a no-op on Windows
    wrapper = user_bin / "wrap.cmd"
    wrapper.write_text("@echo off\r\n", encoding="utf-8")
    wrapper.chmod(0o755)
    pathext = {"PATHEXT": ".COM;.EXE;.BAT;.CMD"}

    assert _first_user_which_hit("srv", str(user_bin), pathext, windows=True) == str(bat)
    # No double extension (wrap.cmd.exe) for a command that already carries a suffix
    assert _first_user_which_hit("wrap.cmd", str(user_bin), pathext, windows=True) == str(wrapper)
    assert _first_user_which_hit("missing", str(user_bin), pathext, windows=True) is None


def test_bare_launcher_commands_keep_the_managed_first_resolution(tmp_path, monkeypatch):
    """The launcher family is exempt from the user-hit step: a bare ``uvx``/``npx``
    resolves through PM's managed tree (``_managed_launcher``), never through
    ``_first_user_which_hit`` — even when a user copy of the launcher exists on the
    child PATH and sorts first (#37589, #111937)."""
    user_bin = _toolchain_bin(tmp_path / "user-bin", "uvx")
    uv_dir = _toolchain_bin(tmp_path / "store" / "uv-0.1", "uv", "uvx")
    _pm_ships(monkeypatch, uv=uv_dir / "uv")

    command, _env = _resolve_stdio_command("uvx", {"PATH": os.pathsep.join([str(user_bin)])})

    assert command == str(uv_dir / "uvx")


def test_tail_server_stderr_scopes_to_the_named_server(tmp_path, monkeypatch):
    """The connect-failure log line quotes the child's last stderr lines; a server that
    never started has no segment and gets nothing (not another server's output)."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import mcp_tool_config as cfg

    token = set_hermes_home_override(tmp_path)
    try:
        cfg._close_mcp_stderr_logs()
        cfg._write_stderr_log_header("beta")
        fh = cfg._get_mcp_stderr_log()
        fh.write("beta noise\n")
        fh.flush()
        cfg._write_stderr_log_header("alpha")
        fh.write("ModuleNotFoundError: No module named 'requests'\n")
        fh.flush()

        assert "ModuleNotFoundError" in cfg._tail_server_stderr("alpha")
        assert "beta noise" not in cfg._tail_server_stderr("alpha")
        assert cfg._tail_server_stderr("never-started") == ""
    finally:
        cfg._close_mcp_stderr_logs()
        reset_hermes_home_override(token)
