"""Regression tests for #75093: mcp_servers.*.args JSON-quoted YAML scalar.

When a config writer serialises ``args`` as a JSON string instead of a proper
YAML sequence, ``_run_stdio`` previously passed the entire JSON string as a
single argv element. The MCP SDK's ``StdioServerParameters`` then iterates
the string (pydantic ``list[str]`` coercion), the spawned server receives a
malformed argv, exits immediately, and the caller sees
``McpError: Connection closed``.

These tests exercise the real production path: they drive ``_run_stdio`` with
the broken shapes and capture what actually reaches ``StdioServerParameters``.
The argv the watchdog wrapper prepends is constant
(``[watchdog_path, --ppid, <pid>, --, <command>]``), so we extract the
trailing user-args segment and assert on THAT. On ``upstream/main`` the
JSON string ends up split into individual characters by pydantic; with the
fix it splits correctly into the original argv elements.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from tools.mcp_tool import MCPServerTask, _MCP_AVAILABLE

# Ensure the mcp module symbols exist for patching even when the SDK isn't installed
if not _MCP_AVAILABLE:
    import tools.mcp_tool as _mcp_mod
    if not hasattr(_mcp_mod, "StdioServerParameters"):
        _mcp_mod.StdioServerParameters = MagicMock
    if not hasattr(_mcp_mod, "stdio_client"):
        _mcp_mod.stdio_client = MagicMock
    if not hasattr(_mcp_mod, "ClientSession"):
        _mcp_mod.ClientSession = MagicMock


# ---------------------------------------------------------------------------
# Production-path capture: drive _run_stdio just far enough to inspect the
# ``StdioServerParameters`` it builds, without actually spawning anything.
# ---------------------------------------------------------------------------


def _capture_user_args(config: dict) -> list:
    """Return the user-supplied argv segment that reaches
    ``StdioServerParameters`` after the watchdog wrapper has prepended its
    fixed prefix ``[watchdog_path, --ppid, <pid>, --, <command>]``.

    Returns the trailing slice (everything after the literal ``"--"`` token
    minus the command itself), so tests can compare directly against the
    expected user-supplied argv.
    """
    captured: dict = {}

    def fake_params(*, command, args, env=None, **_kwargs):
        captured["command"] = command
        captured["args"] = list(args)
        captured["env"] = env
        return SimpleNamespace(command=command, args=args, env=env)

    class _FakeStdioClient:
        def __init__(self, params, errlog=None):
            pass

        async def __aenter__(self):
            return (MagicMock(), MagicMock())

        async def __aexit__(self, *_args):
            return False

    task = MCPServerTask(name="regression-75093")
    task._config = config

    async def _drive():
        with patch("tools.mcp_tool.StdioServerParameters", side_effect=fake_params), \
             patch("tools.mcp_tool.stdio_client", _FakeStdioClient), \
             patch("tools.mcp_tool._kill_orphaned_mcp_children", return_value=None), \
             patch("tools.mcp_tool._snapshot_child_pids", return_value=set()), \
             patch("tools.mcp_tool._write_stderr_log_header"), \
             patch("tools.mcp_tool._get_mcp_stderr_log", return_value=None), \
             patch("tools.mcp_tool.asyncio.to_thread",
                   new=AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))):
            try:
                await task._run_stdio(config)
            except Exception:
                # _run_stdio calls into ClientSession which we never wired up.
                # We only care about captured args.
                pass

    asyncio.run(_drive())
    # Watchdog wraps argv as: [watchdog_path, --ppid, <pid>, --, <command>, *user_args]
    full = captured.get("args", [])
    try:
        sep_idx = full.index("--")
    except ValueError:
        return full  # no watchdog wrap (Windows); return as-is
    # After `--` we get the original command followed by the user args.
    return full[sep_idx + 2:]


# ---------------------------------------------------------------------------
# Test cases (RED on upstream/main; GREEN with the fix)
# ---------------------------------------------------------------------------


def test_json_quoted_args_string_is_split_into_list():
    """The canonical #75093 shape: ``'["-y", "@modelcontextprotocol/server-filesystem"]'``.

    On upstream/main this string is split into individual characters by
    pydantic's ``list[str]`` coercion, then forwarded verbatim to the server.
    After the fix the OSV preflight and ``StdioServerParameters`` see
    ``['-y', '@modelcontextprotocol/server-filesystem', '/home/hermes']``.
    """
    config = {
        "command": "npx",
        "args": '["-y", "@modelcontextprotocol/server-filesystem", "/home/hermes"]',
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/hermes",
    ], f"expected JSON-quoted args to split, got {user_args!r}"


def test_json_array_with_non_string_elements_is_coerced_to_strings():
    """JSON arrays may contain ints/floats/null — coerce each to str so the
    watchdog wrapper and ``StdioServerParameters`` receive a uniform
    ``list[str]``. (YAML sequences with non-strings are already a list, but a
    JSON-quoted scalar with non-string elements exercises the same path.)
    """
    config = {
        "command": "uvx",
        "args": '[1, 2.5, "pkg", null]',
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == ["1", "2.5", "pkg", "None"], (
        f"expected non-string JSON elements to coerce to str, got {user_args!r}"
    )


def test_list_args_pass_through_unchanged():
    """The well-formed path: a real YAML sequence of strings must NOT be
    re-normalised — preserve element order and content exactly.
    """
    config = {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/hermes"],
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/hermes",
    ], f"expected list to pass through, got {user_args!r}"


def test_empty_list_args_passes_through():
    """``args: []`` is a legitimate config — don't trip on it."""
    config = {"command": "npx", "args": [], "env": {"PATH": "/usr/bin"}}
    user_args = _capture_user_args(config)
    assert user_args == [], f"expected [] to pass through, got {user_args!r}"


def test_empty_string_args_normalises_to_empty_list():
    """``args: ''`` (rare but possible from a hand-edited config) should
    behave like ``args: []`` rather than producing ``['']`` (or worse, a list
    of single characters from pydantic coercion).
    """
    config = {"command": "npx", "args": "", "env": {"PATH": "/usr/bin"}}
    user_args = _capture_user_args(config)
    assert user_args == [], f"expected '' to normalise to [], got {user_args!r}"


def test_shlex_quoted_scalar_is_split():
    """A POSIX shell-quoted scalar (``'-y "@pkg/path"'``) should split via
    ``shlex.split`` rather than being passed verbatim. This catches hand-edited
    configs that look like shell commands.
    """
    config = {
        "command": "npx",
        "args": '-y "@modelcontextprotocol/server-filesystem"',
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == [
        "-y",
        "@modelcontextprotocol/server-filesystem",
    ], f"expected shlex split, got {user_args!r}"


def test_single_json_string_scalar_wraps_into_single_element_list():
    """A JSON scalar that's a single string (not an array) should wrap to a
    one-element list. Without this, the user loses the value entirely.
    """
    config = {
        "command": "uvx",
        "args": '"some-package-name"',
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == ["some-package-name"], (
        f"expected single JSON string to wrap, got {user_args!r}"
    )


def test_malformed_string_args_falls_through_as_single_element():
    """Garbage that fails both JSON and shlex parsing must not crash the
    gateway and must not be split into characters by pydantic. We pass it
    through as a single argv element and let the OSV preflight / spawned
    process deal with the malformed input.
    """
    config = {
        "command": "npx",
        "args": "garbage[invalid ' unmatched",
        "env": {"PATH": "/usr/bin"},
    }
    user_args = _capture_user_args(config)
    assert user_args == ["garbage[invalid ' unmatched"], (
        f"expected malformed string to fall through as single element, got {user_args!r}"
    )


def test_args_key_absent_defaults_to_empty_list():
    """``config.get('args', [])`` is the canonical default — when ``args`` is
    omitted entirely the normalised value must still be ``[]``.
    """
    config = {"command": "npx", "env": {"PATH": "/usr/bin"}}
    user_args = _capture_user_args(config)
    assert user_args == [], f"expected missing args to default to [], got {user_args!r}"
