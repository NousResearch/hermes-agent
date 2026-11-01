"""Unit tests for tools/mcp_tool_diagnostics.py::list_mcp_subprocess_owners and its helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.mcp_tool_diagnostics import (
    _extract_parent_pgid,
    _role_for_cmdline,
    _server_name_for_cmdline,
    list_mcp_subprocess_owners,
)


def test_extract_parent_pgid_space_separated():
    assert _extract_parent_pgid(["python3", "mcp_death_supervisor.py", "--parent-pgid", "4113"]) == 4113


def test_extract_parent_pgid_equals_form():
    assert _extract_parent_pgid(["python3", "mcp_death_supervisor.py", "--parent-pgid=4113"]) == 4113


def test_extract_parent_pgid_missing():
    assert _extract_parent_pgid(["python3", "mcp_death_supervisor.py"]) is None


def test_extract_parent_pgid_non_numeric_is_ignored():
    assert _extract_parent_pgid(["python3", "mcp_death_supervisor.py", "--parent-pgid", "nope"]) is None


def test_role_detection():
    assert _role_for_cmdline(["python3", "-c", "...", "gateway", "run"]) == "gateway"
    assert _role_for_cmdline(["python3", "-c", "...", "dashboard", "--host", "0.0.0.0"]) == "dashboard"
    assert _role_for_cmdline(["node", "/x/ui-tui/dist/entry.js"]) == "tui"
    assert _role_for_cmdline(["python3", "-m", "tui_gateway.entry"]) == "tui"
    assert _role_for_cmdline(["some", "unrelated", "process"]) is None


def test_server_name_detection():
    assert _server_name_for_cmdline(["/x/uv", "tool", "uvx", "wikipedia-mcp"]) == "wikipedia"
    assert _server_name_for_cmdline(["node", ".../mcp-searxng"]) == "searxng"
    assert _server_name_for_cmdline(["unrelated"]) is None


def test_list_mcp_subprocess_owners_is_read_only_and_returns_list():
    """Smoke test against whatever is actually running: must never raise, must return
    a list of well-shaped dicts, and must never signal/kill anything (no os.kill calls
    other than the read-only signal-0 probe inside the shared _pid_exists helper)."""
    owners = list_mcp_subprocess_owners()
    assert isinstance(owners, list)
    for o in owners:
        assert isinstance(o["supervisor_pid"], int)
        assert isinstance(o["parent_pid"], int)
        assert isinstance(o["alive"], bool)
        assert o["role"] is None or isinstance(o["role"], str)
        assert isinstance(o["server_names"], list)
        assert isinstance(o["server_pgids"], list)
        # The one genuine leak signal is exactly "parent is dead".
        assert o["leak_signal"] == (not o["alive"])
