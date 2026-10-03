"""Late MCP refresh must keep the routed session profile across its thread hop."""

import threading
import types

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools import mcp_tool_agent
from tui_gateway import entry, server


def test_late_mcp_refresh_keeps_routed_profile_context(tmp_path, monkeypatch):
    launch_home = tmp_path / "launch"
    served_home = tmp_path / "served"
    launch_home.mkdir()
    served_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch_home))

    monkeypatch.setattr(entry, "mcp_discovery_in_flight", lambda: True)
    monkeypatch.setattr(entry, "join_mcp_discovery", lambda timeout=None: True)

    observed = []
    called = threading.Event()

    def _observe_refresh(agent, *, quiet_mode=True):
        observed.append(get_hermes_home())
        called.set()
        return False

    monkeypatch.setattr(mcp_tool_agent, "refresh_agent_mcp_tools", _observe_refresh)

    agent = types.SimpleNamespace(_user_turn_count=0, _api_call_count=0)
    sid = "profile-scoped-late-mcp"
    server._sessions[sid] = {"agent": agent, "profile_home": str(served_home)}
    token = set_hermes_home_override(served_home)
    try:
        server._schedule_mcp_late_refresh(sid, agent)
    finally:
        reset_hermes_home_override(token)

    try:
        assert called.wait(5), "late MCP refresh worker did not reach the refresh boundary"
        for thread in list(threading.enumerate()):
            if thread.name == f"tui-mcp-late-refresh-{sid}":
                thread.join(timeout=5)
        assert observed == [served_home]
    finally:
        server._sessions.pop(sid, None)
