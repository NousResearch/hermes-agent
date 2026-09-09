"""TUI/Desktop plugin command context: methods_tools._plugin_command_invocation.

The TUI server and the Desktop app share the tui_gateway backend; both route
plugin slash commands through the same helper, so it is tested directly here.

Coverage targets:
  - Session-less dispatch yields an empty fail-closed context.
  - A fresh session record (no agent yet) still carries provenance (surface,
    platform, cwd) but refuses tool dispatch.
  - A session with a live agent authorizes dispatch against the agent's effective
    tool set and reports the session identity.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tui_gateway.methods_tools import _plugin_command_invocation


class _FakeAgent:
    session_id = "sess-t1"
    platform = "tui"
    valid_tool_names = ["read_file", "terminal"]


def test_sessionless_dispatch_context_fails_closed():
    invocation = _plugin_command_invocation(None)
    assert invocation is not None
    assert invocation.authorized is False
    assert invocation.session_id == ""
    assert invocation.session_key == ""
    assert invocation.tool_names == frozenset()
    with pytest.raises(PermissionError, match="not authorized"):
        invocation.dispatch_tool("read_file", {"path": "x"})


def test_fresh_session_carries_provenance_but_no_authority():
    invocation = _plugin_command_invocation({
        "session_key": "key-t1", "source": "desktop", "cwd": "C:/work/project",
        "agent": None,
    })
    assert invocation.surface == "desktop"
    assert invocation.platform == "desktop"
    assert invocation.session_key == "key-t1"
    assert invocation.cwd == Path("C:/work/project")
    assert invocation.authorized is False
    with pytest.raises(PermissionError, match="not authorized"):
        invocation.dispatch_tool("read_file", {"path": "x"})


def test_live_agent_session_authorizes_against_effective_tool_set():
    invocation = _plugin_command_invocation({
        "session_key": "key-t2", "source": "tui", "cwd": "C:/work/project",
        "agent": _FakeAgent(),
    })
    assert invocation.surface == "tui"
    assert invocation.session_id == "sess-t1"
    assert invocation.authorized is True
    assert invocation.tool_names == frozenset({"read_file", "terminal"})
    # Refused before any tool execution: outside the session's effective set.
    with pytest.raises(PermissionError, match="not available"):
        invocation.dispatch_tool("web_search", {"query": "x"})
    with pytest.raises(TypeError, match="dictionary"):
        invocation.dispatch_tool("read_file", ["path"])
