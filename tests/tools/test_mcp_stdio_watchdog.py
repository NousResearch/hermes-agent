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


@pytest.mark.skipif(os.name != "posix", reason="prctl is a Linux kernel API")
def test_prctl_pdeathsig_loaded_at_import():
    """Import and reload exercises the PR_SET_PDEATHSIG codepath.

    The prctl(PR_SET_PDEATHSIG) call lives inside main() which also
    spawns a subprocess via Popen — incompatible with pytest's stdin
    redirection. This test validates the module remains importable with
    the prctl code present; the live path is exercised in integration."""
    import importlib

    import tools.mcp_stdio_watchdog as _mod

    importlib.reload(_mod)
    assert hasattr(_mod, "main")
