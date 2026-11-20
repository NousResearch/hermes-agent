"""An MCP server's ``notifications/progress`` during ``tools/call`` reaches the agent's
``tool_progress_callback`` as throttled ``tool.progress`` events for that call id. Real stdio child,
real config, real SDK progress token round trip."""

import json
import sys
import textwrap

import pytest

SERVER = textwrap.dedent('''\
    from mcp.server.mcpserver import MCPServer, Context
    app = MCPServer("prog")

    @app.tool()
    async def slow(ctx: Context) -> str:
        for i in range(1, 21):  # back to back: the throttle must drop intermediates, never the last
            await ctx.report_progress(i * 5, 100, f"stage {i}")
        return "done"

    app.run()
''')


@pytest.mark.no_isolate
@pytest.mark.platforms("posix")
def test_mcp_progress_notifications_reach_tool_progress_callback(monkeypatch, tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    home = tmp_path / "home"
    home.mkdir()
    (home / "server.py").write_text(SERVER, encoding="utf-8")
    (home / "config.yaml").write_text(
        f"mcp_servers:\n  prog:\n    command: {json.dumps(sys.executable)}\n    args: [{json.dumps(str(home / 'server.py'))}]\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.tool_progress_relay import bound_tool_progress
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_loop as _loop
    from tools.registry import registry

    events = []
    try:
        assert "mcp__prog__slow" in disc.discover_mcp_tools()
        with bound_tool_progress(lambda *a, **kw: events.append((a, kw)), "mcp__prog__slow", "call-7"):
            result = registry.dispatch("mcp__prog__slow", {})
        assert "done" in result
    finally:
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        shutdown_mcp_servers()
        _loop._stop_mcp_loop()

    assert events and all(a[0] == "tool.progress" and a[1] == "mcp__prog__slow" for a, _ in events)
    assert {kw["tool_call_id"] for _, kw in events} == {"call-7"}
    # 20 updates sent back to back: the throttle drops intermediates, never the final 100%. Exact
    # pacing is pinned below on a controlled clock.
    assert len(events) < 20
    assert events[-1][1]["percent"] == 100 and events[-1][0][2] == "stage 20"


def test_progress_throttle_is_per_call_and_always_passes_first_and_final(monkeypatch):
    from agent import tool_progress_relay as relay

    clock = [0.0]
    monkeypatch.setattr(relay.time, "monotonic", lambda: 100.0 + clock[0] / 8)
    events = []
    with relay.bound_tool_progress(lambda *a, **kw: events.append(kw["progress"]), "t", "call-1"):
        report = relay.current_tool_progress()
        for step in range(1, 11):  # one update every 1/8 s (exact in binary floats), total 100
            report(step * 10.0, 100.0, None)
            clock[0] += 1

    # First (10), then one per 500 ms (50: 4/8 s later), and the final 100 regardless of the interval.
    assert events == [10.0, 50.0, 90.0, 100.0]
