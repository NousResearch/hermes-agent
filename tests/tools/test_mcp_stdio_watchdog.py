"""Contract tests for the direct POSIX stdio MCP child watchdog."""

import os
import sys

import pytest

from tools import mcp_stdio_watchdog, mcp_tool


def test_is_orphaned_is_false_while_direct_parent_is_unchanged():
    original_ppid = 1234

    assert mcp_stdio_watchdog._is_orphaned(
        original_ppid,
        getppid=lambda: original_ppid,
    ) is False


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_wrap_command_uses_stable_parent_pid_and_preserves_command_tail():
    parent_pid = os.getpid()
    command = "/opt/hermes/bin/mcp-server"
    command_args = ["--label", "value with spaces", "--", "literal-tail"]

    wrapped_command, wrapped_args = mcp_tool._wrap_command_with_watchdog(
        command,
        command_args,
    )

    assert wrapped_command == sys.executable
    assert wrapped_args == [
        os.path.join(os.path.dirname(mcp_tool.__file__), "mcp_stdio_watchdog.py"),
        "--ppid",
        str(parent_pid),
        "--",
        command,
        *command_args,
    ]
    assert "--create-time" not in wrapped_args


def test_coerce_mcp_stdio_args_parses_json_and_python_repr():
    """A YAML-quoted list must not be unpacked as characters (#79519)."""
    assert mcp_tool._coerce_mcp_stdio_args(["-y", "pkg"]) == ["-y", "pkg"]
    assert mcp_tool._coerce_mcp_stdio_args('["-y", "hermes-atlas-mcp"]') == [
        "-y",
        "hermes-atlas-mcp",
    ]
    assert mcp_tool._coerce_mcp_stdio_args("['-y', 'hermes-atlas-mcp']") == [
        "-y",
        "hermes-atlas-mcp",
    ]
    assert mcp_tool._coerce_mcp_stdio_args(None) == []
    assert mcp_tool._coerce_mcp_stdio_args("-y") == ["-y"]


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
def test_wrap_does_not_split_stringified_args_into_characters():
    coerced = mcp_tool._coerce_mcp_stdio_args('["-y", "hermes-atlas-mcp"]')
    _command, wrapped_args = mcp_tool._wrap_command_with_watchdog("npx", coerced)
    assert wrapped_args[-3:] == ["npx", "-y", "hermes-atlas-mcp"]
    assert "[" not in wrapped_args
